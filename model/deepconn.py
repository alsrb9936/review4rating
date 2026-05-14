import torch
import torch.nn as nn
from collections.abc import Mapping
from typing import Union, cast, final

from .abstract import AbstractRec


class FactorizationMachine(nn.Module):
    def __init__(self, input_dim: int, fm_k: int) -> None:
        super().__init__()
        self.linear: nn.Linear = nn.Linear(input_dim, 1)
        self.V: nn.Parameter = nn.Parameter(torch.randn(input_dim, fm_k) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear_part = cast(torch.Tensor, self.linear(x))
        inter_part1 = cast(torch.Tensor, torch.mm(x, self.V) ** 2)
        inter_part2 = cast(torch.Tensor, torch.mm(x ** 2, self.V ** 2))
        interaction = cast(
            torch.Tensor,
            0.5 * torch.sum(inter_part1 - inter_part2, dim=1, keepdim=True),
        )
        return cast(torch.Tensor, linear_part + interaction)


class DeepCoNN(AbstractRec):
    def __init__(self, configs: Mapping[str, object], train_dataset: object) -> None:
        super().__init__()
        self.configs = configs
        self.train_dataset = train_dataset

        self.review_length = self._get_int_config("review_length", 40)
        self.review_count = self._get_int_config("review_count", 10)
        self.word_dim = self._get_int_config("word_dim", 50)
        self.kernel_count = self._get_int_config("kernel_count", 100)
        self.kernel_size = self._get_int_config("kernel_size", 3)
        self.latent_dim = self._get_int_config("cnn_out_dim", 50)
        self.cnn_out_dim = self.latent_dim
        self.fm_k = self._get_int_config("fm_k", 10)
        self.dropout_prob = self._get_float_config("dropout_prob", 0.5)
        self.pad_idx = int(getattr(train_dataset, "pad_idx", 0))

        embedding_weight: torch.Tensor = cast(torch.Tensor, getattr(train_dataset, "embedding_matrix"))
        if embedding_weight.size(1) != self.word_dim:
            raise ValueError(
                f"Configured word_dim={self.word_dim} does not match loaded embedding dim={embedding_weight.size(1)}"
            )

        self.cnn_u = nn.ModuleDict(
            {
                "embedding": nn.Embedding.from_pretrained(
                    embedding_weight,
                    freeze=True,
                    padding_idx=self.pad_idx,
                ),
                "conv": nn.Conv1d(
                    in_channels=self.word_dim,
                    out_channels=self.kernel_count,
                    kernel_size=self.kernel_size,
                    padding=(self.kernel_size - 1) // 2,
                ),
                "fc": nn.Linear(self.review_count * self.kernel_count, self.cnn_out_dim),
            }
        )

        self.cnn_i = nn.ModuleDict(
            {
                "embedding": nn.Embedding.from_pretrained(
                    embedding_weight,
                    freeze=True,
                    padding_idx=self.pad_idx,
                ),
                "conv": nn.Conv1d(
                    in_channels=self.word_dim,
                    out_channels=self.kernel_count,
                    kernel_size=self.kernel_size,
                    padding=(self.kernel_size - 1) // 2,
                ),
                "fc": nn.Linear(self.review_count * self.kernel_count, self.cnn_out_dim),
            }
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_prob)
        self.predict_layer = FactorizationMachine(self.cnn_out_dim * 2, self.fm_k)

        self.loss_fn = nn.MSELoss()

    def _get_int_config(self, key: str, default: int) -> int:
        return int(cast(Union[int, float, str], self.configs.get(key, default)))

    def _get_float_config(self, key: str, default: float) -> float:
        return float(cast(Union[int, float, str], self.configs.get(key, default)))

    def _encode_reviews(
        self,
        review_tokens: torch.Tensor,
        embedding: nn.Embedding,
        conv: nn.Conv1d,
        fc: nn.Linear,
    ) -> torch.Tensor:
        assert review_tokens.dim() == 3, f"Expected [B, R, L], got {tuple(review_tokens.shape)}"

        batch_size, review_count, review_length = review_tokens.shape
        assert review_count == self.review_count, (
            f"Expected review_count={self.review_count}, got {review_count}"
        )
        assert review_length == self.review_length, (
            f"Expected review_length={self.review_length}, got {review_length}"
        )

        x = review_tokens.reshape(batch_size * review_count, review_length)
        assert x.shape == (batch_size * self.review_count, self.review_length), (
            f"Expected reshaped tokens {(batch_size * self.review_count, self.review_length)}, got {tuple(x.shape)}"
        )

        x = cast(torch.Tensor, embedding(x))
        assert x.shape == (batch_size * self.review_count, self.review_length, self.word_dim), (
            f"Expected embedded shape {(batch_size * self.review_count, self.review_length, self.word_dim)}, got {tuple(x.shape)}"
        )

        x = cast(torch.Tensor, x.transpose(1, 2))
        x = cast(torch.Tensor, self.relu(conv(x)))
        x = cast(torch.Tensor, torch.amax(x, dim=2))
        assert x.shape == (batch_size * self.review_count, self.kernel_count), (
            f"Expected conv+pool shape {(batch_size * self.review_count, self.kernel_count)}, got {tuple(x.shape)}"
        )

        x = x.reshape(batch_size, self.review_count * self.kernel_count)
        assert x.shape == (batch_size, self.review_count * self.kernel_count), (
            f"Expected review concat shape {(batch_size, self.review_count * self.kernel_count)}, got {tuple(x.shape)}"
        )

        x = cast(torch.Tensor, fc(x))
        assert x.shape == (batch_size, self.cnn_out_dim), (
            f"Expected FC output shape {(batch_size, self.cnn_out_dim)}, got {tuple(x.shape)}"
        )

        x = cast(torch.Tensor, self.relu(x))
        x = cast(torch.Tensor, self.dropout(x))
        return x

    def forward(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        if args:
            user_review, item_review = args[:2]
        else:
            user_review = kwargs["user_review"]
            item_review = kwargs["item_review"]

        user_latent = self._encode_reviews(
            user_review,
            cast(nn.Embedding, self.cnn_u["embedding"]),
            cast(nn.Conv1d, self.cnn_u["conv"]),
            cast(nn.Linear, self.cnn_u["fc"]),
        )
        item_latent = self._encode_reviews(
            item_review,
            cast(nn.Embedding, self.cnn_i["embedding"]),
            cast(nn.Conv1d, self.cnn_i["conv"]),
            cast(nn.Linear, self.cnn_i["fc"]),
        )

        x = cast(torch.Tensor, torch.cat([user_latent, item_latent], dim=1))
        return cast(torch.Tensor, self.predict_layer(x))

    def cal_loss(self, *args: object, **kwargs: object) -> tuple[torch.Tensor, dict[str, float]]:
        batch_data = args[0] if args else kwargs["batch_data"]
        user_review, item_review, ratings = cast(tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_data)

        predictions = self.forward(user_review, item_review)
        mse_loss = self.loss_fn(predictions, ratings.view(-1, 1))

        loss_dict = {
            "mse_loss": float(mse_loss.detach().item()),
            "total_loss": float(mse_loss.detach().item()),
        }
        return mse_loss, loss_dict

    def predict_scores(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        if args:
            user_review, item_review = args[:2]
        else:
            user_review = cast(torch.Tensor, kwargs["user_review"])
            item_review = cast(torch.Tensor, kwargs["item_review"])
        return self.forward(user_review, item_review)
