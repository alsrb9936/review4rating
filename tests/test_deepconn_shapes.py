from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.deepconn import DeepCoNN


class MockDataset:
    embedding_matrix: torch.Tensor
    pad_idx: int

    def __init__(self, vocab_size: int = 10000, word_dim: int = 50, pad_idx: int = 0):
        self.embedding_matrix = torch.randn(vocab_size, word_dim)
        self.pad_idx = pad_idx


def build_model() -> tuple[DeepCoNN, dict[str, int | float]]:
    _ = torch.manual_seed(42)
    configs: dict[str, int | float] = {
        "review_length": 40,
        "review_count": 10,
        "word_dim": 50,
        "kernel_count": 100,
        "kernel_size": 3,
        "cnn_out_dim": 50,
        "fm_k": 10,
        "dropout_prob": 0.5,
    }
    train_dataset = MockDataset()
    model = DeepCoNN(configs, train_dataset)

    return model, configs


def test_forward_shape(model: DeepCoNN, configs: dict[str, int | float]) -> None:
    review_count = int(configs["review_count"])
    review_length = int(configs["review_length"])
    batch_size = 4
    user_review = torch.randint(
        0,
        10000,
        (batch_size, review_count, review_length),
        dtype=torch.long,
    )
    item_review = torch.randint(
        0,
        10000,
        (batch_size, review_count, review_length),
        dtype=torch.long,
    )

    with torch.no_grad():
        output = model(user_review, item_review)

    assert output.shape == (batch_size, 1), f"Expected [4, 1], got {tuple(output.shape)}"


def test_fm_v_shape(model: DeepCoNN, configs: dict[str, int | float]) -> None:
    cnn_out_dim = int(configs["cnn_out_dim"])
    fm_k = int(configs["fm_k"])
    v_param = model.predict_layer.V
    v_shape = tuple(v_param.size())
    assert v_shape == (2 * cnn_out_dim, fm_k), (
        f"Expected V shape [100, 10], got {v_shape}"
    )


def test_cnn_encoder_internal_flow(model: DeepCoNN, configs: dict[str, int | float]) -> None:
    word_dim = int(configs["word_dim"])
    review_count = int(configs["review_count"])
    review_length = int(configs["review_length"])
    kernel_count = int(configs["kernel_count"])
    cnn_out_dim = int(configs["cnn_out_dim"])
    batch_size = 4

    review_tokens = torch.randint(
        0,
        10000,
        (batch_size, review_count, review_length),
        dtype=torch.long,
    )

    # Expected flow requested by the task.
    x = review_tokens.reshape(batch_size * review_count, review_length)
    assert tuple(x.shape) == (batch_size * review_count, review_length)

    x = model.cnn_u["embedding"](x)
    assert tuple(x.shape) == (batch_size * review_count, review_length, word_dim)

    x = x.transpose(1, 2)
    assert tuple(x.shape) == (batch_size * review_count, word_dim, review_length)

    x = model.relu(model.cnn_u["conv"](x))
    assert tuple(x.shape) == (batch_size * review_count, kernel_count, review_length)

    x = torch.amax(x, dim=2)
    assert tuple(x.shape) == (batch_size * review_count, kernel_count)

    x = x.reshape(batch_size, review_count * kernel_count)
    assert tuple(x.shape) == (batch_size, review_count * kernel_count)

    x = model.cnn_u["fc"](x)
    assert tuple(x.shape) == (batch_size, cnn_out_dim)


def main() -> None:
    model, configs = build_model()
    test_forward_shape(model, configs)
    test_fm_v_shape(model, configs)
    test_cnn_encoder_internal_flow(model, configs)
    print("All shape tests passed!")


if __name__ == "__main__":
    main()
