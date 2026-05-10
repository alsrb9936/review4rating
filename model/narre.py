# pyright: reportAny=false, reportDeprecated=false, reportImplicitOverride=false, reportIncompatibleMethodOverride=false, reportUnknownMemberType=false, reportUnnecessaryCast=false, reportUnnecessaryComparison=false, reportUnusedCallResult=false

from collections.abc import Mapping
from typing import Dict, Tuple, Union, cast, final

import torch
import torch.nn as nn

from .abstract import AbstractRec


@final
class NARRE(AbstractRec):
    def __init__(self, configs: Mapping[str, object], train_dataset: object) -> None:
        super().__init__()
        self.configs = configs
        self.train_dataset = train_dataset

        self.review_length = self._get_int_config("review_length", 40)
        self.review_count = self._get_int_config("review_count", 10)
        self.word_dim = self._get_int_config("word_dim", 50)
        self.kernel_count = self._get_int_config("kernel_count", 100)
        self.kernel_size = self._get_int_config("kernel_size", 3)
        self.id_dim = self._get_int_config("id_dim", 32)
        self.attention_size = self._get_int_config("attention_size", 32)
        self.dropout_prob = self._get_float_config("dropout_prob", 0.5)

        self.num_users = int(getattr(train_dataset, "num_users"))
        self.num_items = int(getattr(train_dataset, "num_items"))
        self.pad_idx = int(getattr(train_dataset, "pad_idx", 0))

        embedding_weight = torch.as_tensor(
            getattr(train_dataset, "embedding_matrix"),
            dtype=torch.float32,
        )
        if embedding_weight.size(1) != self.word_dim:
            raise ValueError(
                "Configured word_dim={} does not match embedding dim={}".format(
                    self.word_dim,
                    embedding_weight.size(1),
                )
            )

        conv_padding = (self.kernel_size - 1) // 2

        self.word_embedding = nn.Embedding.from_pretrained(
            embedding_weight,
            freeze=bool(configs.get("freeze_word_embedding", False)),
            padding_idx=self.pad_idx,
        )
        # Keep large embedding on CPU to avoid OOM with 300d GoogleNews
        self.word_embedding_cpu = bool(configs.get("word_embedding_cpu", True))
        if self.word_embedding_cpu:
            self.word_embedding.cpu()

        self.user_conv = nn.Conv1d(
            in_channels=self.word_dim,
            out_channels=self.kernel_count,
            kernel_size=self.kernel_size,
            padding=conv_padding,
        )
        self.item_conv = nn.Conv1d(
            in_channels=self.word_dim,
            out_channels=self.kernel_count,
            kernel_size=self.kernel_size,
            padding=conv_padding,
        )

        self.user_review_fc = nn.Linear(self.kernel_count, self.attention_size)
        self.item_review_fc = nn.Linear(self.kernel_count, self.attention_size)
        self.user_id_attention_fc = nn.Linear(self.id_dim, self.attention_size)
        self.item_id_attention_fc = nn.Linear(self.id_dim, self.attention_size)
        self.user_attention_fc = nn.Linear(self.attention_size, 1)
        self.item_attention_fc = nn.Linear(self.attention_size, 1)

        self.user_feature_fc = nn.Linear(self.kernel_count, self.id_dim)
        self.item_feature_fc = nn.Linear(self.kernel_count, self.id_dim)

        self.user_id_embedding = nn.Embedding(self.num_users, self.id_dim)
        self.item_id_embedding = nn.Embedding(self.num_items, self.id_dim)
        self.user_review_item_id_embedding = nn.Embedding(
            self.num_items + 1,
            self.id_dim,
            padding_idx=self.num_items,
        )
        self.item_review_user_id_embedding = nn.Embedding(
            self.num_users + 1,
            self.id_dim,
            padding_idx=self.num_users,
        )

        self.user_bias = nn.Embedding(self.num_users, 1)
        self.item_bias = nn.Embedding(self.num_items, 1)
        self.predict_layer = nn.Linear(self.id_dim, 1)

        ratings_tensor = torch.as_tensor(
            getattr(train_dataset, "ratings"),
            dtype=torch.float32,
        )
        self.global_bias = nn.Parameter(ratings_tensor.mean())

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_prob)
        self.loss_fn = nn.MSELoss()

        self.init_weights()

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if self.word_embedding_cpu:
            self.word_embedding.cpu()
        return self

    def _get_int_config(self, key: str, default: int) -> int:
        return int(cast(Union[int, float, str], self.configs.get(key, default)))

    def _get_float_config(self, key: str, default: float) -> float:
        return float(cast(Union[int, float, str], self.configs.get(key, default)))

    def init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding) and module is not self.word_embedding:
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].fill_(0.0)

    def _lookup_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.word_embedding_cpu:
            return cast(torch.Tensor, self.word_embedding(tokens.cpu())).to(tokens.device)
        return cast(torch.Tensor, self.word_embedding(tokens))

    def _encode_reviews(self, review_tokens: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
        if review_tokens.dim() != 3:
            raise ValueError("Expected review tokens with shape [B, R, L].")

        batch_size, review_count, review_length = review_tokens.shape
        if review_count != self.review_count:
            raise ValueError(
                "Expected review_count={}, got {}".format(self.review_count, review_count)
            )
        if review_length != self.review_length:
            raise ValueError(
                "Expected review_length={}, got {}".format(self.review_length, review_length)
            )

        review_input = review_tokens.reshape(batch_size * review_count, review_length)
        embedded = self._lookup_embedding(review_input)
        embedded = cast(torch.Tensor, embedded.transpose(1, 2))
        conv_out = cast(torch.Tensor, self.relu(conv(embedded)))
        pooled = cast(torch.Tensor, torch.amax(conv_out, dim=2))
        return pooled.reshape(batch_size, review_count, self.kernel_count)

    def _masked_attention(
        self,
        review_features: torch.Tensor,
        review_id_features: torch.Tensor,
        review_fc: nn.Linear,
        id_fc: nn.Linear,
        attention_fc: nn.Linear,
        review_mask: torch.Tensor,
    ) -> torch.Tensor:
        review_proj = cast(torch.Tensor, review_fc(review_features))
        id_proj = cast(torch.Tensor, id_fc(review_id_features))
        attention_input = cast(torch.Tensor, self.relu(review_proj + id_proj))
        attention_logits = cast(torch.Tensor, attention_fc(attention_input))

        mask = review_mask.unsqueeze(-1)
        attention_logits = attention_logits.masked_fill(~mask, -1e9)
        attention = cast(torch.Tensor, torch.softmax(attention_logits, dim=1))
        attention = attention * mask.to(dtype=attention.dtype)
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-8)

        attended = cast(torch.Tensor, torch.sum(attention * review_features, dim=1))
        return attended

    def forward(
        self,
        user_id: torch.Tensor,
        item_id: torch.Tensor,
        user_review: torch.Tensor,
        item_review: torch.Tensor,
        user_review_item_ids: torch.Tensor,
        item_review_user_ids: torch.Tensor,
    ) -> torch.Tensor:
        user_review_features = self._encode_reviews(user_review, self.user_conv)
        item_review_features = self._encode_reviews(item_review, self.item_conv)

        user_review_item_ids = user_review_item_ids.clamp(min=0, max=self.num_items)
        item_review_user_ids = item_review_user_ids.clamp(min=0, max=self.num_users)

        user_review_item_features = cast(
            torch.Tensor,
            self.user_review_item_id_embedding(user_review_item_ids),
        )
        item_review_user_features = cast(
            torch.Tensor,
            self.item_review_user_id_embedding(item_review_user_ids),
        )

        user_review_mask = user_review.ne(self.pad_idx).any(dim=2)
        item_review_mask = item_review.ne(self.pad_idx).any(dim=2)
        user_review_mask = user_review_mask & user_review_item_ids.ne(self.num_items)
        item_review_mask = item_review_mask & item_review_user_ids.ne(self.num_users)

        attended_user = self._masked_attention(
            review_features=user_review_features,
            review_id_features=user_review_item_features,
            review_fc=self.user_review_fc,
            id_fc=self.user_id_attention_fc,
            attention_fc=self.user_attention_fc,
            review_mask=user_review_mask,
        )
        attended_item = self._masked_attention(
            review_features=item_review_features,
            review_id_features=item_review_user_features,
            review_fc=self.item_review_fc,
            id_fc=self.item_id_attention_fc,
            attention_fc=self.item_attention_fc,
            review_mask=item_review_mask,
        )

        user_feature = cast(torch.Tensor, self.user_feature_fc(self.dropout(attended_user)))
        item_feature = cast(torch.Tensor, self.item_feature_fc(self.dropout(attended_item)))

        user_feature = user_feature + cast(torch.Tensor, self.user_id_embedding(user_id))
        item_feature = item_feature + cast(torch.Tensor, self.item_id_embedding(item_id))

        interaction = self.relu(user_feature * item_feature)
        rating = cast(torch.Tensor, self.predict_layer(self.dropout(interaction)))
        rating = rating + cast(torch.Tensor, self.user_bias(user_id))
        rating = rating + cast(torch.Tensor, self.item_bias(item_id))
        rating = rating + self.global_bias
        return rating

    def cal_loss(
        self,
        batch_data: Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        (
            user_id,
            item_id,
            user_review,
            item_review,
            user_review_item_ids,
            item_review_user_ids,
            ratings,
        ) = batch_data

        predictions = self.forward(
            user_id=user_id,
            item_id=item_id,
            user_review=user_review,
            item_review=item_review,
            user_review_item_ids=user_review_item_ids,
            item_review_user_ids=item_review_user_ids,
        )
        mse_loss = self.loss_fn(predictions, ratings.view(-1, 1).float())
        loss_value = float(mse_loss.detach().item())
        return mse_loss, {"mse_loss": loss_value, "total_loss": loss_value}

    def predict_scores(
        self,
        user_id: torch.Tensor,
        item_id: torch.Tensor,
        user_review: torch.Tensor,
        item_review: torch.Tensor,
        user_review_item_ids: torch.Tensor,
        item_review_user_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(
            user_id=user_id,
            item_id=item_id,
            user_review=user_review,
            item_review=item_review,
            user_review_item_ids=user_review_item_ids,
            item_review_user_ids=item_review_user_ids,
        )
