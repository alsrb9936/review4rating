import numpy as np
import torch

from .scg_rgcl_dataset import SCGRGCLDataset


class MARGCLDataset(SCGRGCLDataset):
    """Full-graph MA-RGCL dataset with soft rating-review alignment features."""

    def _build_history_feature_tensors(self, target_frame, history_frame, exclude_self: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        history_review = self._extract_review_tensor(history_frame)
        history_users = history_frame["user_id"].to_numpy(dtype=np.int64)
        history_items = history_frame["item_id"].to_numpy(dtype=np.int64)
        target_users = target_frame["user_id"].to_numpy(dtype=np.int64)
        target_items = target_frame["item_id"].to_numpy(dtype=np.int64)

        user_sum: dict[int, torch.Tensor] = {}
        item_sum: dict[int, torch.Tensor] = {}
        user_count: dict[int, int] = {}
        item_count: dict[int, int] = {}
        for pos, (user_id, item_id) in enumerate(zip(history_users, history_items)):
            review_vec = history_review[pos]
            user_key = int(user_id)
            item_key = int(item_id)
            user_sum[user_key] = user_sum.get(user_key, torch.zeros_like(review_vec)) + review_vec
            item_sum[item_key] = item_sum.get(item_key, torch.zeros_like(review_vec)) + review_vec
            user_count[user_key] = user_count.get(user_key, 0) + 1
            item_count[item_key] = item_count.get(item_key, 0) + 1

        target_review = self._extract_review_tensor(target_frame) if exclude_self else None
        user_history = []
        item_history = []
        empty_mask = []
        for pos, (user_id, item_id) in enumerate(zip(target_users, target_items)):
            user_key = int(user_id)
            item_key = int(item_id)
            user_total = user_sum.get(user_key, torch.zeros(self.review_dim, dtype=torch.float32)).clone()
            item_total = item_sum.get(item_key, torch.zeros(self.review_dim, dtype=torch.float32)).clone()
            user_n = user_count.get(user_key, 0)
            item_n = item_count.get(item_key, 0)

            if exclude_self and target_review is not None:
                user_total = user_total - target_review[pos]
                item_total = item_total - target_review[pos]
                user_n -= 1
                item_n -= 1

            user_empty = user_n <= 0
            item_empty = item_n <= 0
            if user_empty:
                user_history.append(torch.zeros(self.review_dim, dtype=torch.float32))
            else:
                user_history.append(user_total / float(user_n))
            if item_empty:
                item_history.append(torch.zeros(self.review_dim, dtype=torch.float32))
            else:
                item_history.append(item_total / float(item_n))
            empty_mask.append(bool(user_empty or item_empty))

        if not user_history:
            return (
                torch.empty((0, self.review_dim), dtype=torch.float32),
                torch.empty((0, self.review_dim), dtype=torch.float32),
                torch.empty((0,), dtype=torch.bool),
            )
        return torch.stack(user_history), torch.stack(item_history), torch.tensor(empty_mask, dtype=torch.bool)

    def _rating_and_sentiment_features(self, frame):
        ratings = frame["rating"].to_numpy(dtype=np.float32)
        _, sent_scores, confidence, _, _ = self._compute_sentiment_features(frame)

        denom = max(self.max_rating - self.min_rating, self.eps)
        normalized_rating = np.clip((ratings - self.min_rating) / denom, 0.0, 1.0).astype(np.float32)
        sentiment_score = np.clip((sent_scores - self.min_rating) / denom, 0.0, 1.0).astype(np.float32)

        if not self._has_sentiment_signal(frame):
            # Placeholder path: no sentiment is available, so keep every edge fully aligned.
            sentiment_score = normalized_rating.copy()
            confidence = np.ones(len(frame), dtype=np.float32)

        raw_misalign = np.abs(normalized_rating - sentiment_score).astype(np.float32)
        edge_misalign_score = (confidence.astype(np.float32) * raw_misalign).astype(np.float32)
        return normalized_rating, sentiment_score, edge_misalign_score

    def _has_sentiment_signal(self, frame) -> bool:
        return self._detect_prob_matrix(frame) is not None or self._detect_label_indices(frame) is not None

    def _decoder_review_sentiment_score(self, frame) -> np.ndarray:
        probs = self._detect_prob_matrix(frame)
        if probs is not None:
            probs = probs.astype(np.float32)
            probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), self.eps)
            raw_scale = np.linspace(0.0, 1.0, probs.shape[1], dtype=np.float32)
            return np.clip(probs @ raw_scale, 0.0, 1.0).astype(np.float32)

        labels = self._detect_label_indices(frame)
        if labels is not None:
            safe_labels = np.clip(labels, 0, 4).astype(np.float32)
            return np.clip(safe_labels / 4.0, 0.0, 1.0).astype(np.float32)

        # No review sentiment signal exists for this split; use a neutral placeholder, not the rating.
        return np.full(len(frame), 0.5, dtype=np.float32)

    def _build_decoder_tensors(self, frame) -> None:
        super()._build_decoder_tensors(frame)
        sentiment_score = self._decoder_review_sentiment_score(frame)
        # Shape: (num_decoder_edges,). This is review-derived only; decoder ratings are never exposed.
        self.decoder_sentiment_score = torch.tensor(sentiment_score, dtype=torch.float32)
        (
            self.decoder_user_history_feat,
            self.decoder_item_history_feat,
            self.decoder_history_empty_mask,
        ) = self._build_history_feature_tensors(frame, frame, exclude_self=True)

    def _build_encoder_graph(self, frame) -> None:
        user_ids = frame["user_id"].to_numpy(dtype=np.int64)
        item_ids = frame["item_id"].to_numpy(dtype=np.int64)
        ratings = frame["rating"].to_numpy(dtype=np.float32)
        review_feat = self._extract_review_tensor(frame)
        normalized_rating, sentiment_score, edge_misalign_score = self._rating_and_sentiment_features(frame)

        self.train_edge_distance_stats = self._summarize(edge_misalign_score)
        stats = self.train_edge_distance_stats
        print(
            f"[MA_RGCL:{self.split}] edge misalignment: "
            f"mean={stats['mean']:.4f}, std={stats['std']:.4f}, min={stats['min']:.4f}, max={stats['max']:.4f}"
        )

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
                # Shapes below: (num_train_edges_for_rating,). No edge is hard dropped.
                "normalized_rating": torch.tensor(normalized_rating[mask], dtype=torch.float32),
                "sentiment_score": torch.tensor(sentiment_score[mask], dtype=torch.float32),
                "edge_misalign_score": torch.tensor(edge_misalign_score[mask], dtype=torch.float32),
            }

        self.norm_factors = self._compute_norm_factors(self.encoder_data)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item.update(
            {
                "decoder_sentiment_score": self.decoder_sentiment_score,
                "decoder_user_history_feat": self.decoder_user_history_feat,
                "decoder_item_history_feat": self.decoder_item_history_feat,
                "decoder_history_empty_mask": self.decoder_history_empty_mask,
            }
        )
        return item

    def _setup_evaluation(self, train_df, valid_df, test_df):
        super()._setup_evaluation(train_df, valid_df, test_df)
        (
            self.decoder_user_history_feat,
            self.decoder_item_history_feat,
            self.decoder_history_empty_mask,
        ) = self._build_history_feature_tensors(self.df, train_df, exclude_self=False)


def ma_rgcl_collate_fn(batch):
    return batch[0]
