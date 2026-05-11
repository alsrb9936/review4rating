import math
from typing import Any, cast

import numpy as np
import pandas as pd

from main import diagnose_and_filter_cold_start
from trainer.base_trainer import BaseTrainer


def run_evaluation_protocol_sanity_check():
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.configs = {"eval_clip": True}
    trainer.min_rating = 1.0
    trainer.max_rating = 5.0
    trainer.eval_clip = True

    predictions = np.array([0.0, 6.0, 3.0], dtype=np.float64)
    ratings = np.array([1.0, 5.0, 3.0], dtype=np.float64)
    metrics = trainer._build_eval_metrics(predictions, ratings, phase="sanity")
    if not math.isclose(metrics["rmse"], 0.0, abs_tol=1e-12):
        raise AssertionError(f"Expected clipped RMSE 0.0, got {metrics['rmse']}")
    if metrics["unclipped_prediction_min"] != 0.0 or metrics["unclipped_prediction_max"] != 6.0:
        raise AssertionError("Unclipped prediction range was not recorded correctly.")
    if metrics["clipped_prediction_min"] != 1.0 or metrics["clipped_prediction_max"] != 5.0:
        raise AssertionError("Clipped prediction range was not recorded correctly.")

    train_df = pd.DataFrame({"user_id": [0, 1], "item_id": [10, 11], "rating": [5.0, 4.0]})
    valid_df = pd.DataFrame({"user_id": [0, 2], "item_id": [10, 11], "rating": [5.0, 4.0]})
    test_df = pd.DataFrame({"user_id": [1, 1], "item_id": [11, 12], "rating": [4.0, 3.0]})
    configs: dict[str, object] = {"drop_cold_start_eval": True}
    filtered_valid, filtered_test = diagnose_and_filter_cold_start(train_df, valid_df, test_df, configs)
    cold_summary = cast(dict[str, dict[str, Any]], configs["cold_start_eval_summary"])
    if len(filtered_valid) != 1 or len(filtered_test) != 1:
        raise AssertionError(
            f"Expected cold-start filtering to leave one valid/test row, got {len(filtered_valid)}/{len(filtered_test)}"
        )
    if cold_summary["valid"]["removed_count"] != 1:
        raise AssertionError("Valid cold-start removed_count should be 1.")
    if cold_summary["test"]["removed_count"] != 1:
        raise AssertionError("Test cold-start removed_count should be 1.")

    return {
        "eval_clip_rmse": metrics["rmse"],
        "valid_samples_after_filter": len(filtered_valid),
        "test_samples_after_filter": len(filtered_test),
        "passed": True,
    }


if __name__ == "__main__":
    print(run_evaluation_protocol_sanity_check())
