from __future__ import annotations

import re
from collections.abc import Mapping

import numpy as np
import pandas as pd
import torch

from .abstract_dataset import RecDataset


Interaction = tuple[int, int, list[int], list[int]]
ReviewEntry = tuple[int, list[int]]
ReviewLookup = dict[int, list[ReviewEntry]]


class NARREDataset(RecDataset):
    """NARRE review-context dataset.

    Two modes are supported via ``retain_rui``:
    - ``True``: Reproduction mode. The target user-item review is kept inside
      the user/item context, matching the original GitHub implementation.
    - ``False``: Fair evaluation mode. The target review is excluded from the
      context so the model only observes historical reviews.
    """

    _glove_cache: dict[tuple[str, int, frozenset[str]], torch.Tensor] = {}

    review_length: int
    review_count: int
    word_dim: int
    glove_path: str
    retain_rui: bool
    user_review_padding_id: int
    item_review_padding_id: int
    user_ids: torch.Tensor
    item_ids: torch.Tensor
    ratings: torch.Tensor
    pad_idx: int
    word_to_idx: dict[str, int]
    user_word_to_idx: dict[str, int]
    item_word_to_idx: dict[str, int]
    user_embedding_matrix: torch.Tensor
    item_embedding_matrix: torch.Tensor
    embedding_matrix: torch.Tensor
    interactions: list[Interaction]
    review_lookup_by_user: ReviewLookup
    review_lookup_by_item: ReviewLookup
    user_review_tensors: torch.Tensor | None
    item_review_tensors: torch.Tensor | None
    user_review_item_ids: torch.Tensor | None
    item_review_user_ids: torch.Tensor | None

    def __init__(
        self,
        df: pd.DataFrame,
        configs: Mapping[str, object],
        split: str = "train",
        train_dataset: NARREDataset | None = None,
    ) -> None:
        super().__init__(df, configs, split)
        self.review_length = self._coerce_int(configs.get("review_length", 40), 40)
        self.review_count = self._coerce_int(configs.get("review_count", 10), 10)
        self.word_dim = self._coerce_int(configs.get("word_dim", 50), 50)
        self.glove_path = self._coerce_str(
            configs.get(
            "glove_path",
            "/home/infolab/mnt/mingyu/review_rec/cached/others/GoogleNews-vectors-negative300.txt",
            ),
            "/home/infolab/mnt/mingyu/review_rec/cached/others/GoogleNews-vectors-negative300.txt",
        )
        self.retain_rui = self._coerce_bool(configs.get("retain_rui", True), True)

        self.user_review_padding_id = self.num_users + 1
        self.item_review_padding_id = self.num_items + 1

        self.user_ids = torch.tensor(df["user_id"].tolist(), dtype=torch.long)
        self.item_ids = torch.tensor(df["item_id"].tolist(), dtype=torch.long)
        self.ratings = torch.tensor(df["rating"].tolist(), dtype=torch.float32)

        # Vocab & embedding: train builds, valid/test share from train
        if train_dataset is not None:
            self.pad_idx = train_dataset.pad_idx
            self.word_to_idx = train_dataset.word_to_idx
            self.user_word_to_idx = train_dataset.user_word_to_idx
            self.item_word_to_idx = train_dataset.item_word_to_idx
            self.user_embedding_matrix = train_dataset.user_embedding_matrix.clone()
            self.item_embedding_matrix = train_dataset.item_embedding_matrix.clone()
            self.embedding_matrix = self.user_embedding_matrix
        else:
            user_vocab_tokens, item_vocab_tokens = self._collect_side_vocab_tokens(df)
            self.pad_idx = 0
            self.user_word_to_idx = self._build_word_to_idx(user_vocab_tokens)
            self.item_word_to_idx = self._build_word_to_idx(item_vocab_tokens)
            # Backward-compatible alias for callers that only need a vocabulary
            # object; the model now reads side-specific vocabularies directly.
            self.word_to_idx = self.user_word_to_idx

            self.user_embedding_matrix = self._load_glove(
                self.glove_path,
                self.word_dim,
                user_vocab_tokens,
                self.user_word_to_idx,
            ).clone()
            self.item_embedding_matrix = self._load_glove(
                self.glove_path,
                self.word_dim,
                item_vocab_tokens,
                self.item_word_to_idx,
            )
            self.embedding_matrix = self.user_embedding_matrix

        self.interactions = []
        for row in df.itertuples(index=False):
            review_text = self._normalize_text(getattr(row, "reviewText", ""))
            self.interactions.append(
                (
                    int(getattr(row, "user_id")),
                    int(getattr(row, "item_id")),
                    self._tokens_to_ids(review_text, self.user_word_to_idx),
                    self._tokens_to_ids(review_text, self.item_word_to_idx),
                )
            )

        self.review_lookup_by_user = self._build_review_lookups(self.interactions, use_user_key=True)
        self.review_lookup_by_item = self._build_review_lookups(self.interactions, use_user_key=False)

        self.user_review_tensors = None
        self.item_review_tensors = None
        self.user_review_item_ids = None
        self.item_review_user_ids = None

        if split == "train":
            (
                self.user_review_tensors,
                self.item_review_tensors,
                self.user_review_item_ids,
                self.item_review_user_ids,
            ) = self._build_context_tensors(
                self.interactions,
                self.review_lookup_by_user,
                self.review_lookup_by_item,
            )
        else:
            # valid/test: build context using TRAIN review lookups + side-specific vocabularies
            self._setup_evaluation_from_train(train_dataset)

    @classmethod
    def _load_glove(
        cls,
        glove_path: str,
        word_dim: int,
        vocab_tokens: set[str],
        word_to_idx: dict[str, int],
    ) -> torch.Tensor:
        cache_key = (glove_path, word_dim, frozenset(vocab_tokens))
        if cache_key in cls._glove_cache:
            return cls._glove_cache[cache_key]

        # Original TensorFlow NARRE initializes the word tables uniformly in
        # [-1, 1], then overwrites words found in the external word2vec file.
        embedding_matrix = torch.empty(len(word_to_idx), word_dim, dtype=torch.float32).uniform_(-1.0, 1.0)

        with open(glove_path, "r", encoding="utf-8") as glove_file:
            for line_idx, line in enumerate(glove_file):
                parts = line.rstrip().split()
                if line_idx == 0 and len(parts) == 2 and all(part.lstrip("+-").isdigit() for part in parts):
                    continue
                if len(parts) != word_dim + 1:
                    continue
                token = parts[0]
                if token not in vocab_tokens:
                    continue
                embedding_matrix[word_to_idx[token]] = torch.tensor([float(value) for value in parts[1:]], dtype=torch.float32)

        cls._glove_cache[cache_key] = embedding_matrix
        return embedding_matrix

    @classmethod
    def _collect_vocab_tokens(cls, df: pd.DataFrame) -> set[str]:
        vocab_tokens: set[str] = set()
        for row in df.itertuples(index=False):
            review_text = cls._normalize_text(getattr(row, "reviewText", ""))
            vocab_tokens.update(cls._tokenize(review_text))
        return vocab_tokens

    @classmethod
    def _collect_side_vocab_tokens(cls, df: pd.DataFrame) -> tuple[set[str], set[str]]:
        # The original TensorFlow preprocessing builds independent user-review
        # and item-review vocabularies. A review text appears in both corpora,
        # but each side still owns its own word-index mapping and embedding table.
        vocab_tokens = cls._collect_vocab_tokens(df)
        return set(vocab_tokens), set(vocab_tokens)

    @staticmethod
    def _build_word_to_idx(vocab_tokens: set[str]) -> dict[str, int]:
        # Original data_pro.py pads with the literal token "<PAD/>" and then
        # builds an alphabetically sorted vocabulary from padded corpora. The
        # token sorts before alphabetic words, so it remains index 0 here.
        pad_token = "<PAD/>"
        ordered_tokens = [pad_token]
        ordered_tokens.extend(token for token in sorted(vocab_tokens) if token != pad_token)
        word_to_idx = {}
        for idx, token in enumerate(ordered_tokens):
            word_to_idx[token] = idx
        return word_to_idx

    @staticmethod
    def _coerce_int(value: object, default: int) -> int:
        if value is None:
            return default
        return int(value)

    @staticmethod
    def _coerce_bool(value: object, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    @staticmethod
    def _coerce_str(value: object, default: str) -> str:
        if value is None:
            return default
        return str(value)

    @staticmethod
    def _normalize_text(text: object) -> str:
        if text is None:
            return ""
        if isinstance(text, (float, np.floating)) and np.isnan(text):
            return ""
        normalized = re.sub(r"[^A-Za-z]", " ", str(text))
        normalized = re.sub(r"\'s", " \'s", normalized)
        normalized = re.sub(r"\'ve", " \'ve", normalized)
        normalized = re.sub(r"n\'t", " n\'t", normalized)
        normalized = re.sub(r"\'re", " \'re", normalized)
        normalized = re.sub(r"\'d", " \'d", normalized)
        normalized = re.sub(r"\'ll", " \'ll", normalized)
        normalized = re.sub(r",", " , ", normalized)
        normalized = re.sub(r"!", " ! ", normalized)
        normalized = re.sub(r"\(", " \\( ", normalized)
        normalized = re.sub(r"\)", " \\) ", normalized)
        normalized = re.sub(r"\?", " \\? ", normalized)
        normalized = re.sub(r"\s{2,}", " ", normalized)
        return normalized.strip().lower()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        if not text:
            return []
        return text.split(" ")

    def _tokens_to_ids(self, text: str, word_to_idx: dict[str, int]) -> list[int]:
        pad_idx = word_to_idx.get("<PAD/>", self.pad_idx)
        return [word_to_idx.get(token, pad_idx) for token in self._tokenize(text)]

    @staticmethod
    def _build_review_lookups(interactions: list[Interaction], use_user_key: bool) -> ReviewLookup:
        review_lookup: ReviewLookup = {}
        for user_id, item_id, user_review_tokens, item_review_tokens in interactions:
            query_id = user_id if use_user_key else item_id
            other_id = item_id if use_user_key else user_id
            review_tokens = user_review_tokens if use_user_key else item_review_tokens
            review_lookup.setdefault(query_id, []).append((other_id, review_tokens))
        return review_lookup

    def _adjust_review_tokens(self, reviews: list[list[int]]) -> list[list[int]]:
        adjusted = reviews[: self.review_count]
        if len(adjusted) < self.review_count:
            adjusted = adjusted + [[self.pad_idx] * self.review_length for _ in range(self.review_count - len(adjusted))]
        return [review[: self.review_length] + [self.pad_idx] * max(0, self.review_length - len(review)) for review in adjusted]

    def _adjust_side_ids(self, side_ids: list[int], padding_id: int) -> list[int]:
        adjusted = side_ids[: self.review_count]
        if len(adjusted) < self.review_count:
            adjusted = adjusted + [padding_id] * (self.review_count - len(adjusted))
        return adjusted

    def _select_review_entries(
        self,
        lookup: ReviewLookup,
        query_id: int,
        exclude_id: int,
    ) -> list[ReviewEntry]:
        entries = lookup.get(int(query_id), [])
        if self.retain_rui:
            return list(entries)
        return [entry for entry in entries if entry[0] != int(exclude_id)]

    def _load_reviews(self, lookup: ReviewLookup, query_id: int, exclude_id: int) -> torch.Tensor:
        selected_entries = self._select_review_entries(lookup, query_id, exclude_id)
        selected_reviews = [tokens for _, tokens in selected_entries]
        return torch.tensor(self._adjust_review_tokens(selected_reviews), dtype=torch.long)

    def _load_review_ids(
        self, lookup: ReviewLookup, query_id: int, exclude_id: int, padding_id: int
    ) -> torch.Tensor:
        selected_entries = self._select_review_entries(lookup, query_id, exclude_id)
        selected_ids = [other_id for other_id, _ in selected_entries]
        return torch.tensor(self._adjust_side_ids(selected_ids, padding_id), dtype=torch.long)

    def _build_context_tensors(
        self,
        target_interactions: list[Interaction],
        review_lookup_by_user: ReviewLookup,
        review_lookup_by_item: ReviewLookup,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        user_reviews: list[torch.Tensor] = []
        item_reviews: list[torch.Tensor] = []
        user_review_item_ids: list[torch.Tensor] = []
        item_review_user_ids: list[torch.Tensor] = []

        for user_id, item_id, _, _ in target_interactions:
            user_reviews.append(self._load_reviews(review_lookup_by_user, query_id=user_id, exclude_id=item_id))
            item_reviews.append(self._load_reviews(review_lookup_by_item, query_id=item_id, exclude_id=user_id))
            user_review_item_ids.append(
                self._load_review_ids(
                    review_lookup_by_user,
                    query_id=user_id,
                    exclude_id=item_id,
                    padding_id=self.item_review_padding_id,
                )
            )
            item_review_user_ids.append(
                self._load_review_ids(
                    review_lookup_by_item,
                    query_id=item_id,
                    exclude_id=user_id,
                    padding_id=self.user_review_padding_id,
                )
            )

        return (
            torch.stack(user_reviews),
            torch.stack(item_reviews),
            torch.stack(user_review_item_ids),
            torch.stack(item_review_user_ids),
        )

    def _setup_evaluation_from_train(self, train_dataset: NARREDataset | None) -> None:
        if self.split not in {"valid", "test"} or train_dataset is None:
            return

        (
            self.user_review_tensors,
            self.item_review_tensors,
            self.user_review_item_ids,
            self.item_review_user_ids,
        ) = self._build_context_tensors(
            self.interactions,
            train_dataset.review_lookup_by_user,
            train_dataset.review_lookup_by_item,
        )

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if (
            self.user_review_tensors is None
            or self.item_review_tensors is None
            or self.user_review_item_ids is None
            or self.item_review_user_ids is None
        ):
            raise RuntimeError("Review context tensors must be initialized before accessing NARRE samples.")

        return {
            "user_id": self.user_ids[idx].clone().detach(),
            "item_id": self.item_ids[idx].clone().detach(),
            "rating": self.ratings[idx].clone().detach(),
            "user_review": self.user_review_tensors[idx].clone().detach(),
            "item_review": self.item_review_tensors[idx].clone().detach(),
            "user_review_item_ids": self.user_review_item_ids[idx].clone().detach(),
            "item_review_user_ids": self.item_review_user_ids[idx].clone().detach(),
        }
