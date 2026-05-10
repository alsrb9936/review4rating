import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .abstract_dataset import RecDataset


# retain_rui is controlled via config (default: false in deepconn.yaml).
# Set retain_rui: true for DeepCoNN paper reproduction mode.
# Keep retain_rui: false for fair evaluation to avoid target-review leakage.

class DeepCoNNDataset(RecDataset):
    """DeepCoNN review-context dataset.

    Two modes are supported via ``retain_rui``:
    - ``True``: Original DeepCoNN reproduction mode. The target review is kept
      in the user/item context, which matches the paper but introduces data leakage.
    - ``False``: Fair evaluation mode. The target review is excluded from the
      context to prevent leakage and is the recommended default.
    """

    _glove_cache: Dict[str, Tuple[Dict[str, int], torch.Tensor, int]] = {}

    def __init__(self, df, configs, split="train"):
        super().__init__(df, configs, split)
        self.review_length = int(configs.get("review_length", 40))
        self.review_count = int(configs.get("review_count", 10))
        self.retain_rui = bool(configs.get("retain_rui", False))
        self.word_dim = int(configs.get("word_dim", 50))
        self.glove_path = configs.get(
            "glove_path",
            "/home/infolab/mnt/mingyu/review_rec/rating4review/cached/others/glove.6B.50d.txt",
        )

        self.user_ids = torch.as_tensor(df["user_id"].to_numpy(dtype=np.int64), dtype=torch.long)
        self.item_ids = torch.as_tensor(df["item_id"].to_numpy(dtype=np.int64), dtype=torch.long)
        self.ratings = torch.as_tensor(df["rating"].to_numpy(dtype=np.float32), dtype=torch.float32)

        vocab, embedding_matrix, self.pad_idx = self._load_glove(self.glove_path, self.word_dim)
        self.word_to_idx = vocab
        self.embedding_matrix = embedding_matrix

        self.interactions: List[Tuple[int, int, List[int]]] = []
        for row in df.itertuples(index=False):
            normalized_text = self._normalize_text(getattr(row, "reviewText"))
            self.interactions.append(
                (
                    int(getattr(row, "user_id")),
                    int(getattr(row, "item_id")),
                    self._tokens_to_ids(normalized_text),
                )
            )

        self.review_frames_by_user = self._build_review_lookups(self.interactions, use_user_key=True)
        self.review_frames_by_item = self._build_review_lookups(self.interactions, use_user_key=False)
        self.user_review_tensors: Optional[torch.Tensor] = None
        self.item_review_tensors: Optional[torch.Tensor] = None

        if split == "train":
            self.user_review_tensors, self.item_review_tensors = self._build_context_tensors(
                self.interactions,
                self.review_frames_by_user,
                self.review_frames_by_item,
            )

    @classmethod
    def _load_glove(cls, glove_path: str, word_dim: int) -> tuple[dict[str, int], torch.Tensor, int]:
        cache_key = f"{glove_path}:{word_dim}"
        if cache_key in cls._glove_cache:
            return cls._glove_cache[cache_key]

        word_to_idx: dict[str, int] = {"<pad>": 0}
        vectors: list[list[float]] = [[0.0] * word_dim]

        with open(glove_path, "r", encoding="utf-8") as glove_file:
            for line in glove_file:
                parts = line.rstrip().split()
                if len(parts) != word_dim + 1:
                    continue
                token = parts[0]
                if token in word_to_idx:
                    continue
                values = [float(value) for value in parts[1:]]
                word_to_idx[token] = len(vectors)
                vectors.append(values)

        embedding_matrix = torch.tensor(vectors, dtype=torch.float32)
        cached_value = (word_to_idx, embedding_matrix, 0)
        cls._glove_cache[cache_key] = cached_value
        return cached_value

    @staticmethod
    def _normalize_text(text: object) -> str:
        if text is None:
            return ""
        if isinstance(text, float) and np.isnan(text):
            return ""
        return str(text).strip().lower()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9']+", text)

    def _tokens_to_ids(self, text: str) -> list[int]:
        return [self.word_to_idx.get(token, self.pad_idx) for token in self._tokenize(text)]

    @staticmethod
    def _build_review_lookups(
        interactions: List[Tuple[int, int, List[int]]],
        use_user_key: bool,
    ) -> Dict[int, List[Tuple[int, List[int]]]]:
        review_lookup: Dict[int, List[Tuple[int, List[int]]]] = {}
        for user_id, item_id, review_tokens in interactions:
            query_id = user_id if use_user_key else item_id
            other_id = item_id if use_user_key else user_id
            review_lookup.setdefault(query_id, []).append((other_id, review_tokens))
        return review_lookup

    def _adjust_review_list(self, reviews: List[List[int]]) -> List[List[int]]:
        adjusted = reviews[: self.review_count]
        if len(adjusted) < self.review_count:
            adjusted = adjusted + [[self.pad_idx] * self.review_length for _ in range(self.review_count - len(adjusted))]
        return [review[: self.review_length] + [self.pad_idx] * max(0, self.review_length - len(review)) for review in adjusted]

    def _load_reviews(self, review_lookup: Dict[int, List[Tuple[int, List[int]]]], query_id: int, exclude_id: int) -> torch.Tensor:
        if self.retain_rui:
            # Reproduction mode: keep every review for the query entity,
            # including the current user-item interaction being predicted.
            selected_reviews = [tokens for _, tokens in review_lookup.get(int(query_id), [])]
        else:
            # Fair mode: remove the target interaction's review from the context
            # so the model only sees historical reviews.
            selected_reviews = [tokens for other_id, tokens in review_lookup.get(int(query_id), []) if other_id != int(exclude_id)]
        return torch.tensor(self._adjust_review_list(selected_reviews), dtype=torch.long)

    def _build_context_tensors(
        self,
        target_interactions: List[Tuple[int, int, List[int]]],
        review_frames_by_user: Dict[int, List[Tuple[int, List[int]]]],
        review_frames_by_item: Dict[int, List[Tuple[int, List[int]]]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        user_reviews = []
        item_reviews = []
        for user_id, item_id, _ in target_interactions:
            user_reviews.append(self._load_reviews(review_frames_by_user, query_id=user_id, exclude_id=item_id))
            item_reviews.append(self._load_reviews(review_frames_by_item, query_id=item_id, exclude_id=user_id))
        return torch.stack(user_reviews), torch.stack(item_reviews)

    def _setup_evaluation(self, train_df, valid_df, test_df):
        if self.split not in {"valid", "test"}:
            return

        history_source = train_df
        history_interactions: List[Tuple[int, int, List[int]]] = []
        for row in history_source.itertuples(index=False):
            history_interactions.append(
                (
                    int(getattr(row, "user_id")),
                    int(getattr(row, "item_id")),
                    self._tokens_to_ids(self._normalize_text(getattr(row, "reviewText"))),
                )
            )

        review_frames_by_user = self._build_review_lookups(history_interactions, use_user_key=True)
        review_frames_by_item = self._build_review_lookups(history_interactions, use_user_key=False)
        self.user_review_tensors, self.item_review_tensors = self._build_context_tensors(
            self.interactions,
            review_frames_by_user,
            review_frames_by_item,
        )

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        if self.user_review_tensors is None or self.item_review_tensors is None:
            raise RuntimeError("Review tensors must be initialized before accessing DeepCoNN samples.")

        return {
            "user_id": self.user_ids[idx].clone().detach(),
            "item_id": self.item_ids[idx].clone().detach(),
            "rating": self.ratings[idx].clone().detach(),
            "user_review": self.user_review_tensors[idx].clone().detach(),
            "item_review": self.item_review_tensors[idx].clone().detach(),
        }
