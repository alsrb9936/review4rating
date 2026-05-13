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
        self.id_dim = self._get_int_config("id_dim", 32)
        self.attention_size = self._get_int_config("attention_size", 32)
        self.dropout_prob = self._get_float_config("dropout_prob", 0.5)
        self.l2_reg_lambda = self._get_float_config("l2_reg_lambda", 0.0)
        self.freeze_word_embedding = self._get_bool_config("freeze_word_embedding", False)
        self.mask_padding_attention = self._get_bool_config("mask_padding_attention", False)

        filter_sizes_raw = configs.get("filter_sizes")
        if filter_sizes_raw is not None:
            if isinstance(filter_sizes_raw, (list, tuple)):
                self.filter_sizes = [int(x) for x in filter_sizes_raw]
            else:
                self.filter_sizes = [int(x.strip()) for x in str(filter_sizes_raw).split(",")]
        else:
            self.filter_sizes = [self._get_int_config("kernel_size", 3)]

        self.conv_output_dim = self.kernel_count * len(self.filter_sizes)

        self.num_users = int(getattr(train_dataset, "num_users"))
        self.num_items = int(getattr(train_dataset, "num_items"))
        self.pad_idx = int(getattr(train_dataset, "pad_idx", 0))

        user_embedding_weight = torch.as_tensor(
            getattr(train_dataset, "user_embedding_matrix"),
            dtype=torch.float32,
        )
        item_embedding_weight = torch.as_tensor(
            getattr(train_dataset, "item_embedding_matrix"),
            dtype=torch.float32,
        )
        if user_embedding_weight.size(1) != self.word_dim:
            raise ValueError(
                "Configured word_dim={} does not match embedding dim={}".format(
                    self.word_dim,
                    user_embedding_weight.size(1),
                )
            )
        if item_embedding_weight.size(1) != self.word_dim:
            raise ValueError(
                "Configured word_dim={} does not match embedding dim={}".format(
                    self.word_dim,
                    item_embedding_weight.size(1),
                )
            )

        fallback_vocab = getattr(train_dataset, "word_to_idx")
        user_vocab_size = len(getattr(train_dataset, "user_word_to_idx", fallback_vocab))
        item_vocab_size = len(getattr(train_dataset, "item_word_to_idx", fallback_vocab))
        assert user_embedding_weight.size(0) == user_vocab_size, (
            "User embedding rows ({}) != vocab size ({})".format(
                user_embedding_weight.size(0), user_vocab_size
            )
        )
        assert item_embedding_weight.size(0) == item_vocab_size, (
            "Item embedding rows ({}) != vocab size ({})".format(
                item_embedding_weight.size(0), item_vocab_size
            )
        )

        if len(self.filter_sizes) == 1:
            self.user_conv = nn.Conv1d(
                in_channels=self.word_dim,
                out_channels=self.kernel_count,
                kernel_size=self.filter_sizes[0],
                padding=0,
            )
            self.item_conv = nn.Conv1d(
                in_channels=self.word_dim,
                out_channels=self.kernel_count,
                kernel_size=self.filter_sizes[0],
                padding=0,
            )
        else:
            self.user_conv = nn.ModuleList([
                nn.Conv1d(
                    in_channels=self.word_dim,
                    out_channels=self.kernel_count,
                    kernel_size=ks,
                    padding=0,
                )
                for ks in self.filter_sizes
            ])
            self.item_conv = nn.ModuleList([
                nn.Conv1d(
                    in_channels=self.word_dim,
                    out_channels=self.kernel_count,
                    kernel_size=ks,
                    padding=0,
                )
                for ks in self.filter_sizes
            ])

        self.user_word_embedding = nn.Embedding.from_pretrained(
            user_embedding_weight,
            freeze=self.freeze_word_embedding,
            padding_idx=self.pad_idx,
        )
        self.item_word_embedding = nn.Embedding.from_pretrained(
            item_embedding_weight,
            freeze=self.freeze_word_embedding,
            padding_idx=self.pad_idx,
        )

        self.user_review_fc = nn.Linear(self.conv_output_dim, self.attention_size)
        self.item_review_fc = nn.Linear(self.conv_output_dim, self.attention_size)
        self.user_id_attention_fc = nn.Linear(self.id_dim, self.attention_size)
        self.item_id_attention_fc = nn.Linear(self.id_dim, self.attention_size)
        self.user_attention_fc = nn.Linear(self.attention_size, 1)
        self.item_attention_fc = nn.Linear(self.attention_size, 1)

        self.user_feature_fc = nn.Linear(self.conv_output_dim, self.id_dim)
        self.item_feature_fc = nn.Linear(self.conv_output_dim, self.id_dim)

        self.user_id_embedding = nn.Embedding(self.num_users, self.id_dim)
        self.item_id_embedding = nn.Embedding(self.num_items, self.id_dim)
        self.user_review_item_id_embedding = nn.Embedding(
            self.num_items + 2,
            self.id_dim,
        )
        self.item_review_user_id_embedding = nn.Embedding(
            self.num_users + 2,
            self.id_dim,
        )

        self.user_bias = nn.Embedding(self.num_users, 1)
        self.item_bias = nn.Embedding(self.num_items, 1)
        self.predict_layer = nn.Linear(self.id_dim, 1)

        self.global_bias = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_prob)
        self.last_user_attention: torch.Tensor | None = None
        self.last_item_attention: torch.Tensor | None = None

        self.init_weights()

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        user_embed_params = self.user_word_embedding.weight.numel()
        user_embed_trainable = self.user_word_embedding.weight.requires_grad
        item_embed_params = self.item_word_embedding.weight.numel()
        item_embed_trainable = self.item_word_embedding.weight.requires_grad
        print(
            (
                "NARRE parameters: total={}, trainable={}, "
                "user_word_embedding={} (trainable={}), "
                "item_word_embedding={} (trainable={})"
            ).format(
                total_params,
                trainable_params,
                user_embed_params,
                user_embed_trainable,
                item_embed_params,
                item_embed_trainable,
            )
        )

    def _get_int_config(self, key: str, default: int) -> int:
        return int(cast(Union[int, float, str], self.configs.get(key, default)))

    def _get_float_config(self, key: str, default: float) -> float:
        return float(cast(Union[int, float, str], self.configs.get(key, default)))

    def _get_bool_config(self, key: str, default: bool) -> bool:
        value = self.configs.get(key, default)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.trunc_normal_(module.weight, mean=0.0, std=0.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.1)
            elif isinstance(module, nn.Linear):
                nn.init.uniform_(module.weight, -0.1, 0.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.1)
            elif isinstance(module, nn.Embedding) and module not in {
                self.user_word_embedding,
                self.item_word_embedding,
            }:
                nn.init.uniform_(module.weight, -0.1, 0.1)
        nn.init.constant_(self.user_bias.weight, 0.1)
        nn.init.constant_(self.item_bias.weight, 0.1)

    def _encode_reviews(
        self,
        review_tokens: torch.Tensor,
        conv,
        word_embedding: nn.Embedding,
    ) -> torch.Tensor:
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
        embedded = cast(torch.Tensor, word_embedding(review_input))
        embedded = cast(torch.Tensor, embedded.transpose(1, 2))

        if isinstance(conv, nn.ModuleList):
            conv_outs = [
                cast(torch.Tensor, torch.amax(self.relu(c(embedded)), dim=2))
                for c in conv
            ]
            pooled = cast(torch.Tensor, torch.cat(conv_outs, dim=1))
        else:
            conv_out = cast(torch.Tensor, self.relu(conv(embedded)))
            pooled = cast(torch.Tensor, torch.amax(conv_out, dim=2))

        return pooled.reshape(batch_size, review_count, self.conv_output_dim)

    def _masked_attention(
        self,
        review_features: torch.Tensor,
        review_id_features: torch.Tensor,
        review_fc: nn.Linear,
        id_fc: nn.Linear,
        attention_fc: nn.Linear,
        review_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        review_proj = cast(torch.Tensor, review_fc(review_features))
        # TensorFlow original applies ReLU immediately after the id embedding
        # lookup before the attention projection.
        id_proj = cast(torch.Tensor, id_fc(self.relu(review_id_features)))
        attention_input = cast(torch.Tensor, self.relu(review_proj + id_proj))
        attention_logits = cast(torch.Tensor, attention_fc(attention_input))

        if self.mask_padding_attention:
            mask = review_mask.unsqueeze(-1)
            attention_logits = attention_logits.masked_fill(~mask, -1e9)
            attention = cast(torch.Tensor, torch.softmax(attention_logits, dim=1))
            attention = attention * mask.to(dtype=attention.dtype)
            attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-8)
        else:
            attention = cast(torch.Tensor, torch.softmax(attention_logits, dim=1))

        attended = cast(torch.Tensor, torch.sum(attention * review_features, dim=1))
        return attended, attention

    def forward(
        self,
        user_id: torch.Tensor,
        item_id: torch.Tensor,
        user_review: torch.Tensor,
        item_review: torch.Tensor,
        user_review_item_ids: torch.Tensor,
        item_review_user_ids: torch.Tensor,
    ) -> torch.Tensor:
        user_vocab_size = self.user_word_embedding.num_embeddings
        item_vocab_size = self.item_word_embedding.num_embeddings
        assert int(user_review.max()) < user_vocab_size, (
            "user_review token max ({}) >= user vocab size ({})".format(
                int(user_review.max()), user_vocab_size
            )
        )
        assert int(item_review.max()) < item_vocab_size, (
            "item_review token max ({}) >= item vocab size ({})".format(
                int(item_review.max()), item_vocab_size
            )
        )

        user_review_features = self._encode_reviews(
            user_review,
            self.user_conv,
            self.user_word_embedding,
        )
        item_review_features = self._encode_reviews(
            item_review,
            self.item_conv,
            self.item_word_embedding,
        )

        user_review_item_ids = user_review_item_ids.clamp(min=0, max=self.num_items + 1)
        item_review_user_ids = item_review_user_ids.clamp(min=0, max=self.num_users + 1)

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
        user_review_mask = user_review_mask & user_review_item_ids.ne(self.num_items + 1)
        item_review_mask = item_review_mask & item_review_user_ids.ne(self.num_users + 1)

        attended_user, user_attention = self._masked_attention(
            review_features=user_review_features,
            review_id_features=user_review_item_features,
            review_fc=self.user_review_fc,
            id_fc=self.user_id_attention_fc,
            attention_fc=self.user_attention_fc,
            review_mask=user_review_mask,
        )
        attended_item, item_attention = self._masked_attention(
            review_features=item_review_features,
            review_id_features=item_review_user_features,
            review_fc=self.item_review_fc,
            id_fc=self.item_id_attention_fc,
            attention_fc=self.item_attention_fc,
            review_mask=item_review_mask,
        )
        self.last_user_attention = user_attention
        self.last_item_attention = item_attention

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
        # Original TensorFlow code uses tf.nn.l2_loss(prediction - label), i.e.
        # one half of the summed squared error for the mini-batch.
        residual = predictions - ratings.view(-1, 1).float()
        rating_loss = 0.5 * torch.sum(residual ** 2)

        attention_l2_loss = torch.tensor(0.0, device=rating_loss.device)
        if self.l2_reg_lambda > 0:
            attention_l2_loss = (
                0.5 * torch.sum(self.user_review_fc.weight ** 2)
                + 0.5 * torch.sum(self.user_id_attention_fc.weight ** 2)
                + 0.5 * torch.sum(self.item_review_fc.weight ** 2)
                + 0.5 * torch.sum(self.item_id_attention_fc.weight ** 2)
            )

        total_loss = rating_loss + self.l2_reg_lambda * attention_l2_loss

        loss_value = float(total_loss.detach().item())
        rating_loss_value = float(rating_loss.detach().item())
        l2_value = float(attention_l2_loss.detach().item())
        return total_loss, {
            "rating_l2_loss": rating_loss_value,
            "attention_l2_loss": l2_value,
            "total_loss": loss_value,
        }

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


def run_narre_sanity_check() -> dict[str, object]:
    """Run a tiny CPU-only check of the TensorFlow-compatible NARRE path."""

    class DummyDataset:
        num_users = 4
        num_items = 5
        pad_idx = 0
        word_to_idx = {"<pad>": 0, "good": 1, "bad": 2, "ok": 3}
        user_embedding_matrix = torch.empty(4, 8).uniform_(-1.0, 1.0)
        item_embedding_matrix = torch.empty(4, 8).uniform_(-1.0, 1.0)
        ratings = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float32)

    configs: dict[str, object] = {
        "review_length": 6,
        "review_count": 3,
        "word_dim": 8,
        "kernel_count": 4,
        "kernel_size": 3,
        "id_dim": 5,
        "attention_size": 7,
        "dropout_prob": 0.5,
        "l2_reg_lambda": 0.001,
        "mask_padding_attention": False,
    }
    model = NARRE(configs, DummyDataset())
    model.train()

    batch_size = 2
    user_id = torch.tensor([0, 1], dtype=torch.long)
    item_id = torch.tensor([2, 3], dtype=torch.long)
    user_review = torch.tensor(
        [
            [[1, 2, 3, 0, 0, 0], [2, 3, 1, 2, 0, 0], [0, 0, 0, 0, 0, 0]],
            [[3, 1, 2, 3, 1, 0], [1, 1, 2, 2, 3, 0], [2, 2, 3, 3, 1, 0]],
        ],
        dtype=torch.long,
    )
    item_review = user_review.flip(dims=[1])
    user_review_item_ids = torch.tensor([[0, 1, 6], [2, 3, 4]], dtype=torch.long)
    item_review_user_ids = torch.tensor([[1, 2, 5], [0, 3, 1]], dtype=torch.long)
    ratings = torch.tensor([4.0, 2.0], dtype=torch.float32)

    prediction = model.forward(
        user_id,
        item_id,
        user_review,
        item_review,
        user_review_item_ids,
        item_review_user_ids,
    )
    loss, loss_dict = model.cal_loss((
        user_id,
        item_id,
        user_review,
        item_review,
        user_review_item_ids,
        item_review_user_ids,
        ratings,
    ))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        eval_prediction = model.forward(
            user_id,
            item_id,
            user_review,
            item_review,
            user_review_item_ids,
            item_review_user_ids,
        )

    return {
        "user_review_shape": tuple(user_review.shape),
        "item_review_shape": tuple(item_review.shape),
        "user_attention_shape": tuple(model.last_user_attention.shape) if model.last_user_attention is not None else None,
        "item_attention_shape": tuple(model.last_item_attention.shape) if model.last_item_attention is not None else None,
        "prediction_shape": tuple(prediction.shape),
        "eval_prediction_shape": tuple(eval_prediction.shape),
        "loss_is_finite": bool(torch.isfinite(loss.detach()).item()),
        "loss": loss_dict,
        "batch_size": batch_size,
    }
