import numpy as np
import torch

from .iard_rm_dataset import IARDRMDataset


class IARMRMSentiDataset(IARDRMDataset):
    def __init__(self, df, configs, split="train"):
        self._sentiment_frame = df.reset_index(drop=True)
        super().__init__(df, configs, split)
        self._build_sentiment_anchor_tensors(self._sentiment_frame)

    @staticmethod
    def _safe_sentiment_scores(value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return [0.0, 1.0, 0.0]
        try:
            scores = list(value)
        except TypeError:
            return [0.0, 1.0, 0.0]
        if len(scores) not in {3, 5}:
            return [0.0, 1.0, 0.0]
        try:
            return [float(score) for score in scores]
        except (TypeError, ValueError):
            return [0.0, 1.0, 0.0]

    @staticmethod
    def _scores_to_three_class(scores):
        if scores.size(1) == 3:
            return scores
        if scores.size(1) == 5:
            negative = scores[:, 0] + scores[:, 1]
            neutral = scores[:, 2]
            positive = scores[:, 3] + scores[:, 4]
            return torch.stack([negative, neutral, positive], dim=1)
        raise ValueError(f"Expected review_score with 3 or 5 columns, got {int(scores.size(1))}.")

    @staticmethod
    def _normalize_sentiment(value):
        normalized = str(value).strip().lower()
        if normalized in {"negative", "neutral", "positive"}:
            return normalized
        if normalized in {"1", "2"}:
            return "negative"
        if normalized == "3":
            return "neutral"
        if normalized in {"4", "5"}:
            return "positive"
        return "neutral"

    def _build_sentiment_anchor_tensors(self, frame):
        if "review_score" not in frame.columns:
            raise ValueError(
                "iarm_rm_senti requires existing sentiment features. "
                "Enable use_sentiment=True so review_score/review_sentiment/rating_sentiment/is_consistent are available."
            )

        if len(frame) == 0:
            self.sent_neg = torch.empty(0, dtype=torch.float32)
            self.sent_neu = torch.empty(0, dtype=torch.float32)
            self.sent_pos = torch.empty(0, dtype=torch.float32)
            self.sent_score = torch.empty(0, dtype=torch.float32)
            self.is_consistent = torch.empty(0, dtype=torch.bool)
            self.q_agree = torch.empty(0, dtype=torch.float32)
            self.q_inc = torch.empty(0, dtype=torch.float32)
            return

        raw_scores = torch.tensor(
            [self._safe_sentiment_scores(value) for value in frame["review_score"].tolist()],
            dtype=torch.float32,
        )
        scores = self._scores_to_three_class(raw_scores)
        self.sent_neg = scores[:, 0]
        self.sent_neu = scores[:, 1]
        self.sent_pos = scores[:, 2]
        self.sent_score = self.sent_pos - self.sent_neg

        if "is_consistent" in frame.columns:
            consistent_values = frame["is_consistent"].fillna(False).astype(bool).to_numpy()
        else:
            consistent_values = np.zeros(len(frame), dtype=bool)
        self.is_consistent = torch.as_tensor(consistent_values, dtype=torch.bool)

        review_sentiments = [self._normalize_sentiment(value) for value in frame.get("review_sentiment", ["neutral"] * len(frame))]
        rating_sentiments = [self._normalize_sentiment(value) for value in frame.get("rating_sentiment", ["neutral"] * len(frame))]
        ratings = torch.as_tensor(frame["rating"].to_numpy(dtype=np.float32), dtype=torch.float32)

        neutral_or_three = torch.tensor(
            [
                review_sentiment == "neutral" or rating_sentiment == "neutral" or float(rating) == 3.0
                for review_sentiment, rating_sentiment, rating in zip(review_sentiments, rating_sentiments, ratings.tolist())
            ],
            dtype=torch.bool,
        )
        consistent = self.is_consistent.float()
        base_q = torch.where(self.is_consistent, torch.full_like(consistent, 0.8), torch.full_like(consistent, 0.2))
        neutral_q = torch.where(self.is_consistent, torch.full_like(consistent, 0.6), torch.full_like(consistent, 0.5))
        base_q = torch.where(neutral_or_three, neutral_q, base_q)

        confidence = scores.max(dim=1).values.clamp(0.0, 1.0)
        non_neutral_strength = (1.0 - self.sent_neu).clamp(0.0, 1.0)
        reliability = (confidence * non_neutral_strength).clamp(0.0, 1.0)
        q_agree = 0.5 + (base_q - 0.5) * reliability
        self.q_agree = q_agree.clamp(0.05, 0.95)
        self.q_inc = 1.0 - self.q_agree

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item.update(
            {
                "sent_neg": self.sent_neg[idx].clone().detach(),
                "sent_neu": self.sent_neu[idx].clone().detach(),
                "sent_pos": self.sent_pos[idx].clone().detach(),
                "sent_score": self.sent_score[idx].clone().detach(),
                "is_consistent": self.is_consistent[idx].clone().detach(),
                "q_agree": self.q_agree[idx].clone().detach(),
                "q_inc": self.q_inc[idx].clone().detach(),
            }
        )
        return item
