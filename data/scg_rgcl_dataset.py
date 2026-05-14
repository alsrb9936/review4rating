import math
from typing import Optional

import numpy as np
import torch

from .abstract_dataset import RecDataset


class SCGRGCLDataset(RecDataset):
    """Full-graph SCG-RGCL dataset with train-only sentiment calibration."""

    def __init__(self, df, configs, split="train"):
        super().__init__(df, configs, split)
        self.rating_vals = [1, 2, 3, 4, 5]
        self.review_dim = int(configs.get("review_dim", 384))
        self.use_review_feat = bool(configs.get("use_review", True))
        self.symm = bool(configs.get("symm", True))
        self.min_rating = float(configs.get("min_rating", 1.0))
        self.max_rating = float(configs.get("max_rating", 5.0))
        self.eps = 1e-8

        self.calibrate_sentiment_to_rating = self._get_bool_config("calibrate_sentiment_to_rating", True)
        self.use_raw_sentiment_distance = self._get_bool_config("use_raw_sentiment_distance", False)
        self.use_sentiment_confidence = self._get_bool_config("use_sentiment_confidence", True)
        self.sentiment_label_col = configs.get("sentiment_label_col")
        self.sentiment_prob_cols = configs.get("sentiment_prob_cols")

        self.encoder_data = {}
        self.norm_factors = {}
        self.decoder_user_ids = None
        self.decoder_item_ids = None
        self.decoder_review_feat = None
        self.labels = None
        self.ratings = None
        self.sentiment_calibration_table = []
        self.train_edge_distance_stats = {}

        self._build_decoder_tensors(self.df)
        if self.split == "train":
            self._build_encoder_graph(self.df)

    def _get_bool_config(self, key: str, default: bool = False) -> bool:
        value = self.configs.get(key, default)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def _extract_review_tensor(self, frame) -> torch.Tensor:
        if not self.use_review_feat or "review_embedding" not in frame.columns:
            return torch.zeros((len(frame), self.review_dim), dtype=torch.float32)

        review_vectors = []
        for review_embedding in frame["review_embedding"].tolist():
            if isinstance(review_embedding, torch.Tensor):
                review_vectors.append(review_embedding.float().view(-1))
            elif review_embedding is None or (isinstance(review_embedding, float) and np.isnan(review_embedding)):
                review_vectors.append(torch.zeros(self.review_dim, dtype=torch.float32))
            else:
                review_vectors.append(torch.tensor(review_embedding, dtype=torch.float32).view(-1))
        return torch.stack(review_vectors, dim=0)

    def _build_decoder_tensors(self, frame) -> None:
        ratings = frame["rating"].to_numpy(dtype=np.float32)
        self.decoder_user_ids = torch.tensor(frame["user_id"].to_numpy(dtype=np.int64), dtype=torch.long)
        self.decoder_item_ids = torch.tensor(frame["item_id"].to_numpy(dtype=np.int64), dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)
        self.labels = torch.tensor(np.searchsorted(np.array(self.rating_vals, dtype=np.float32), ratings), dtype=torch.long)
        self.decoder_review_feat = self._extract_review_tensor(frame)

    def _split_prob_cols(self):
        if self.sentiment_prob_cols is None:
            return None
        if isinstance(self.sentiment_prob_cols, str):
            return [col.strip() for col in self.sentiment_prob_cols.split(",") if col.strip()]
        return list(self.sentiment_prob_cols)

    def _detect_prob_matrix(self, frame) -> Optional[np.ndarray]:
        prob_cols = self._split_prob_cols()
        if prob_cols:
            if all(col in frame.columns for col in prob_cols):
                return frame[prob_cols].to_numpy(dtype=np.float32)
            missing = [col for col in prob_cols if col not in frame.columns]
            print(f"[SCG_RGCL] sentiment_prob_cols missing {missing}; falling back to auto detection.")

        numbered_candidates = []
        for base in ("sent_prob", "sentiment_prob", "review_sent_prob"):
            zero_based = [f"{base}_{idx}" for idx in range(5)]
            one_based = [f"{base}_{idx}" for idx in range(1, 6)]
            if all(col in frame.columns for col in zero_based):
                numbered_candidates.append(zero_based)
            if all(col in frame.columns for col in one_based):
                numbered_candidates.append(one_based)
        if numbered_candidates:
            return frame[numbered_candidates[0]].to_numpy(dtype=np.float32)

        for col in ("review_score", "sentiment_score", "sentiment_probs", "sent_probs"):
            if col not in frame.columns:
                continue
            values = frame[col].tolist()
            rows = []
            for value in values:
                if isinstance(value, torch.Tensor):
                    arr = value.detach().cpu().float().view(-1).numpy()
                elif isinstance(value, (list, tuple, np.ndarray)):
                    arr = np.asarray(value, dtype=np.float32).reshape(-1)
                else:
                    return None
                rows.append(arr)
            if rows and all(row.size == rows[0].size for row in rows) and rows[0].size >= 2:
                matrix = np.stack(rows, axis=0).astype(np.float32)
                row_sums = matrix.sum(axis=1, keepdims=True)
                return matrix / np.maximum(row_sums, self.eps)
        return None

    def _detect_label_col(self, frame) -> Optional[str]:
        if self.sentiment_label_col and self.sentiment_label_col in frame.columns:
            return self.sentiment_label_col
        for col in ("sent_label", "sentiment_label", "review_sentiment", "sentiment", "sentiment_class"):
            if col in frame.columns:
                return col
        return None

    def _parse_label_index(self, value) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, float) and np.isnan(value):
            return None
        text = str(value).strip().lower().replace("_", "-")
        text_to_index = {
            "strong negative": 0,
            "strong-negative": 0,
            "very negative": 0,
            "negative": 1,
            "neutral": 2,
            "positive": 3,
            "strong positive": 4,
            "strong-positive": 4,
            "very positive": 4,
        }
        if text in text_to_index:
            return text_to_index[text]
        try:
            number = int(float(text))
        except ValueError:
            return None
        if 1 <= number <= 5:
            return number - 1
        if 0 <= number <= 4:
            return number
        return None

    def _detect_label_indices(self, frame) -> Optional[np.ndarray]:
        label_col = self._detect_label_col(frame)
        if label_col is None:
            return None
        labels = [self._parse_label_index(value) for value in frame[label_col].tolist()]
        if all(label is None for label in labels):
            return None
        return np.asarray([-1 if label is None else label for label in labels], dtype=np.int64)

    def _fallback_missing_table_values(self, table: np.ndarray, counts: np.ndarray, global_mean: float) -> np.ndarray:
        filled = table.copy()
        missing = counts <= self.eps
        if not missing.any():
            return filled
        for idx in np.where(missing)[0]:
            left = idx - 1
            while left >= 0 and missing[left]:
                left -= 1
            right = idx + 1
            while right < len(table) and missing[right]:
                right += 1
            neighbors = []
            if left >= 0:
                neighbors.append(filled[left])
            if right < len(table):
                neighbors.append(filled[right])
            filled[idx] = float(np.mean(neighbors)) if neighbors else global_mean
        return filled

    def _calibrate_from_probs(self, probs: np.ndarray, ratings: np.ndarray) -> np.ndarray:
        weighted_sum = (probs * ratings.reshape(-1, 1)).sum(axis=0)
        counts = probs.sum(axis=0)
        table = weighted_sum / np.maximum(counts, self.eps)
        return self._fallback_missing_table_values(table, counts, float(ratings.mean()))

    def _calibrate_from_labels(self, labels: np.ndarray, ratings: np.ndarray, num_classes: int = 5) -> np.ndarray:
        counts = np.zeros(num_classes, dtype=np.float32)
        sums = np.zeros(num_classes, dtype=np.float32)
        valid = (labels >= 0) & (labels < num_classes)
        for label, rating in zip(labels[valid], ratings[valid]):
            counts[int(label)] += 1.0
            sums[int(label)] += float(rating)
        table = sums / np.maximum(counts, self.eps)
        return self._fallback_missing_table_values(table, counts, float(ratings.mean()))

    def _compute_confidence(self, probs: np.ndarray) -> np.ndarray:
        if not self.use_sentiment_confidence:
            return np.ones(probs.shape[0], dtype=np.float32)
        entropy = -(probs * np.log(np.maximum(probs, self.eps))).sum(axis=1)
        max_entropy = math.log(max(probs.shape[1], 2))
        confidence = 1.0 - entropy / max_entropy
        return np.clip(confidence, 0.0, 1.0).astype(np.float32)

    def _compute_sentiment_features(self, frame):
        ratings = frame["rating"].to_numpy(dtype=np.float32)
        probs = self._detect_prob_matrix(frame)
        labels = self._detect_label_indices(frame)

        if probs is None and labels is None:
            table = np.full(5, float(ratings.mean()) if len(ratings) else 3.0, dtype=np.float32)
            zeros = np.zeros(len(frame), dtype=np.float32)
            ones = np.ones(len(frame), dtype=np.float32)
            print("[SCG_RGCL] No sentiment columns detected; edge_misalign_score is set to 0.")
            return table, table[0] + zeros, ones, zeros, zeros

        if probs is not None:
            probs = probs.astype(np.float32)
            probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), self.eps)
            if self.calibrate_sentiment_to_rating:
                table = self._calibrate_from_probs(probs, ratings)
                sent_scores = probs @ table
            elif self.use_raw_sentiment_distance:
                table = np.linspace(self.min_rating, self.max_rating, probs.shape[1], dtype=np.float32)
                sent_scores = probs @ table
            else:
                table = self._calibrate_from_probs(probs, ratings)
                sent_scores = probs @ table
            confidence = self._compute_confidence(probs)
            return table.astype(np.float32), sent_scores.astype(np.float32), confidence, None, None

        assert labels is not None
        num_classes = 5
        if self.calibrate_sentiment_to_rating:
            table = self._calibrate_from_labels(labels, ratings, num_classes=num_classes)
        elif self.use_raw_sentiment_distance:
            table = np.linspace(self.min_rating, self.max_rating, num_classes, dtype=np.float32)
        else:
            table = self._calibrate_from_labels(labels, ratings, num_classes=num_classes)
        safe_labels = np.clip(labels, 0, num_classes - 1)
        sent_scores = table[safe_labels]
        confidence = np.ones(len(frame), dtype=np.float32)
        return table.astype(np.float32), sent_scores.astype(np.float32), confidence, None, None

    def _build_encoder_graph(self, frame) -> None:
        user_ids = frame["user_id"].to_numpy(dtype=np.int64)
        item_ids = frame["item_id"].to_numpy(dtype=np.int64)
        ratings = frame["rating"].to_numpy(dtype=np.float32)
        review_feat = self._extract_review_tensor(frame)
        table, sent_scores, confidence, _, _ = self._compute_sentiment_features(frame)

        denom = max(self.max_rating - self.min_rating, self.eps)
        y_norm = np.clip((ratings - self.min_rating) / denom, 0.0, 1.0)
        s_norm = np.clip((sent_scores - self.min_rating) / denom, 0.0, 1.0)
        rating_sent_distance = np.abs(y_norm - s_norm).astype(np.float32)
        edge_misalign_score = (confidence * rating_sent_distance).astype(np.float32)

        self.sentiment_calibration_table = [float(value) for value in table.tolist()]
        self.train_edge_distance_stats = self._summarize(edge_misalign_score)
        self._print_calibration_and_distance_stats()

        self.encoder_data = {}
        for rating in self.rating_vals:
            mask = ratings == float(rating)
            if not np.any(mask):
                continue
            rating_key = str(rating)
            mask_tensor = torch.from_numpy(mask)
            self.encoder_data[rating_key] = {
                "user_ids": torch.tensor(user_ids[mask], dtype=torch.long),
                "item_ids": torch.tensor(item_ids[mask], dtype=torch.long),
                "review_feat": review_feat[mask_tensor],
                # Shape: (num_train_edges_for_rating,). Used only for train-graph propagation gates.
                "edge_misalign_score": torch.tensor(edge_misalign_score[mask], dtype=torch.float32),
                "sent_calibrated_score": torch.tensor(sent_scores[mask], dtype=torch.float32),
                "sent_confidence": torch.tensor(confidence[mask], dtype=torch.float32),
                "rating_sent_distance": torch.tensor(rating_sent_distance[mask], dtype=torch.float32),
            }

        self.norm_factors = self._compute_norm_factors(self.encoder_data)

    def _summarize(self, values: np.ndarray) -> dict[str, float]:
        if values.size == 0:
            return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
        return {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    def _print_calibration_and_distance_stats(self) -> None:
        print(f"[SCG_RGCL:{self.split}] sentiment calibration table:")
        names = ["strong negative", "negative", "neutral", "positive", "strong positive"]
        for idx, value in enumerate(self.sentiment_calibration_table):
            label = names[idx] if len(self.sentiment_calibration_table) == 5 else f"class {idx}"
            print(f"  class {idx} {label} -> mean rating {value:.4f}")
        stats = self.train_edge_distance_stats
        print(
            f"[SCG_RGCL:{self.split}] train edge distance: "
            f"mean={stats['mean']:.4f}, std={stats['std']:.4f}, min={stats['min']:.4f}, max={stats['max']:.4f}"
        )

    def _compute_norm_factors(self, encoder_data):
        user_in = torch.zeros(self.num_users, dtype=torch.float32)
        user_out = torch.zeros(self.num_users, dtype=torch.float32)
        item_in = torch.zeros(self.num_items, dtype=torch.float32)
        item_out = torch.zeros(self.num_items, dtype=torch.float32)

        for edge_data in encoder_data.values():
            edge_users = edge_data["user_ids"]
            edge_items = edge_data["item_ids"]
            user_in.index_add_(0, edge_users, torch.ones_like(edge_users, dtype=torch.float32))
            item_in.index_add_(0, edge_items, torch.ones_like(edge_items, dtype=torch.float32))
            if self.symm:
                user_out.index_add_(0, edge_users, torch.ones_like(edge_users, dtype=torch.float32))
                item_out.index_add_(0, edge_items, torch.ones_like(edge_items, dtype=torch.float32))

        def normalize(degrees: torch.Tensor) -> torch.Tensor:
            degrees = degrees.clone()
            degrees[degrees == 0] = float("inf")
            return (1.0 / torch.sqrt(degrees)).unsqueeze(1)

        return {
            "user": {
                "ci": normalize(user_in),
                "cj": normalize(user_out) if self.symm else torch.ones(self.num_users, 1, dtype=torch.float32),
            },
            "item": {
                "ci": normalize(item_in),
                "cj": normalize(item_out) if self.symm else torch.ones(self.num_items, 1, dtype=torch.float32),
            },
        }

    def _setup_evaluation(self, train_df, valid_df, test_df):
        if self.split in {"valid", "test"}:
            self._build_encoder_graph(train_df)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {
            "encoder_data": self.encoder_data,
            "norm_factors": self.norm_factors,
            "decoder_user_ids": self.decoder_user_ids,
            "decoder_item_ids": self.decoder_item_ids,
            "decoder_review_feat": self.decoder_review_feat,
            "labels": self.labels,
            "ratings": self.ratings,
        }


def scg_rgcl_collate_fn(batch):
    return batch[0]
