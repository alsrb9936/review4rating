from __future__ import annotations

import re
from collections.abc import Mapping, MutableMapping

import numpy as np
import pandas as pd
import torch

from .abstract_dataset import RecDataset


Interaction = tuple[int, int, list[int], list[int]]
ReviewEntry = tuple[int, list[int]]
ReviewLookup = dict[int, list[ReviewEntry]]


class NARREDataset(RecDataset):
    """NARRE review-context dataset.

    This PyTorch dataset reproduces the preprocessing pipeline of the original
    TensorFlow NARRE implementation (chenchongthu/NARRE, WWW'18):

    1. ``loaddata.py`` logic  →  ``_build_original_text_corpora``,
       ``_compute_original_review_shape_from_corpora``
    2. ``data_pro.py::clean_str``  →  ``_normalize_text``
    3. ``data_pro.py::pad_sentences``  →  ``_adjust_review_tokens``
    4. ``data_pro.py::pad_reviewid``  →  ``_adjust_side_ids``
    5. ``data_pro.py::build_vocab``  →  ``_build_word_to_idx``
    6. ``data_pro.py::build_input_data``  →  ``_tokens_to_ids``
    7. ``data_pro.py::load_data_and_labels`` cold-start handling  →
       ``_select_review_entries`` with ``original_cold_start_dummy=True``
    8. Word2vec/GloVe loading  →  ``_load_glove``

    Two modes are supported via ``retain_rui``:
    - ``True``: Reproduction mode. The target user-item review is kept inside
      the user/item context, matching the original GitHub implementation.
    - ``False``: Fair evaluation mode. The target review is excluded from the
      context so the model only observes historical reviews.
    """

    _glove_cache: dict[tuple[str, int, frozenset[str]], torch.Tensor] = {}

    review_length: int
    review_count: int
    user_review_length: int
    item_review_length: int
    user_review_count: int
    item_review_count: int
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
        fit_df: pd.DataFrame | None = None,
    ) -> None:
        super().__init__(df, configs, split)
        self.review_length = self._coerce_int(configs.get("review_length", 40), 40)
        self.review_count = self._coerce_int(configs.get("review_count", 10), 10)
        self.user_review_length = self._coerce_int(configs.get("user_review_length", self.review_length), self.review_length)
        self.item_review_length = self._coerce_int(configs.get("item_review_length", self.review_length), self.review_length)
        self.user_review_count = self._coerce_int(configs.get("user_review_count", self.review_count), self.review_count)
        self.item_review_count = self._coerce_int(configs.get("item_review_count", self.review_count), self.review_count)
        self.word_dim = self._coerce_int(configs.get("word_dim", 50), 50)
        self.max_filter_size = self._get_max_filter_size(configs)
        self.preprocess_mode = self._coerce_str(configs.get("narre_preprocess_mode", "original"), "original")
        self.original_cold_start_dummy = self._coerce_bool(configs.get("original_cold_start_dummy", True), True)
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

        auto_review_shape = self._coerce_bool(configs.get("auto_review_shape", True), True)

        # Vocab & embedding: train builds, valid/test share from train
        if train_dataset is not None:
            self.pad_idx = train_dataset.pad_idx
            self.review_count = train_dataset.review_count
            self.review_length = train_dataset.review_length
            self.user_review_count = train_dataset.user_review_count
            self.item_review_count = train_dataset.item_review_count
            self.user_review_length = train_dataset.user_review_length
            self.item_review_length = train_dataset.item_review_length
            self.word_to_idx = train_dataset.word_to_idx
            self.user_word_to_idx = train_dataset.user_word_to_idx
            self.item_word_to_idx = train_dataset.item_word_to_idx
            self.user_embedding_matrix = train_dataset.user_embedding_matrix.clone()
            self.item_embedding_matrix = train_dataset.item_embedding_matrix.clone()
            self.embedding_matrix = self.user_embedding_matrix
        else:
            fit_frame = fit_df if fit_df is not None else df
            if auto_review_shape:
                if self.preprocess_mode == "original":
                    (
                        self.user_review_count,
                        self.item_review_count,
                        self.user_review_length,
                        self.item_review_length,
                    ) = self._compute_original_review_shape_from_corpora(df, fit_df, self.max_filter_size)
                else:
                    (
                        self.user_review_count,
                        self.item_review_count,
                        self.user_review_length,
                        self.item_review_length,
                    ) = self._compute_original_review_shape(fit_frame, self.max_filter_size, self.preprocess_mode)
                self.review_count = max(self.user_review_count, self.item_review_count)
                self.review_length = max(self.user_review_length, self.item_review_length)
                self._sync_review_shape_to_configs(configs)

            if self.preprocess_mode == "original":
                user_vocab_tokens, item_vocab_tokens = self._collect_original_side_vocab_tokens(
                    df,
                    fit_df,
                    self.user_review_count,
                    self.item_review_count,
                    self.user_review_length,
                    self.item_review_length,
                )
            else:
                user_vocab_tokens, item_vocab_tokens = self._collect_side_vocab_tokens(fit_frame, self.preprocess_mode)
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
            self._print_preprocess_summary(split, cold_start_rows=0)
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
    def _collect_vocab_tokens(cls, df: pd.DataFrame, preprocess_mode: str = "original") -> set[str]:
        vocab_tokens: set[str] = set()
        for row in df.itertuples(index=False):
            review_text = cls._normalize_text(getattr(row, "reviewText", ""), preprocess_mode)
            vocab_tokens.update(cls._tokenize(review_text))
        return vocab_tokens

    @classmethod
    def _collect_side_vocab_tokens(cls, df: pd.DataFrame, preprocess_mode: str = "original") -> tuple[set[str], set[str]]:
        # The original TensorFlow preprocessing builds independent user-review
        # and item-review vocabularies. A review text appears in both corpora,
        # but each side still owns its own word-index mapping and embedding table.
        vocab_tokens = cls._collect_vocab_tokens(df, preprocess_mode)
        return set(vocab_tokens), set(vocab_tokens)

    @classmethod
    def _compute_original_review_shape(cls, df: pd.DataFrame, min_review_length: int = 1, preprocess_mode: str = "original") -> tuple[int, int, int, int]:
        def percentile_90(values: list[int], default: int) -> int:
            if not values:
                return default
            quantile = np.quantile(np.asarray(values, dtype=np.int64), 0.9, method="higher")
            return max(int(quantile), 1)

        user_counts = df.groupby("user_id").size().astype(int).tolist() if len(df) > 0 else []
        item_counts = df.groupby("item_id").size().astype(int).tolist() if len(df) > 0 else []
        review_lengths: list[int] = []
        for row in df.itertuples(index=False):
            review_text = cls._normalize_text(getattr(row, "reviewText", ""), preprocess_mode)
            review_lengths.append(len(cls._tokenize(review_text)))

        user_review_count = percentile_90(user_counts, 1)
        item_review_count = percentile_90(item_counts, 1)
        review_length = max(percentile_90(review_lengths, min_review_length), min_review_length)
        return user_review_count, item_review_count, review_length, review_length

    @classmethod
    def _build_original_text_corpora(
        cls,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame | None,
    ) -> tuple[dict[int, list[list[str]]], dict[int, list[list[str]]]]:
        user_text: dict[int, list[list[str]]] = {}
        item_text: dict[int, list[list[str]]] = {}

        for row in train_df.itertuples(index=False):
            user_id = int(getattr(row, "user_id"))
            item_id = int(getattr(row, "item_id"))
            tokens = cls._tokenize(cls._normalize_text(getattr(row, "reviewText", ""), "original"))
            user_text.setdefault(user_id, []).append(tokens)
            item_text.setdefault(item_id, []).append(tokens)

        if valid_df is not None:
            for row in valid_df.itertuples(index=False):
                user_id = int(getattr(row, "user_id"))
                item_id = int(getattr(row, "item_id"))
                if user_id not in user_text:
                    user_text[user_id] = [["<PAD/>"]]
                if item_id not in item_text:
                    item_text[item_id] = [["<PAD/>"]]

        return user_text, item_text

    @classmethod
    def _compute_original_review_shape_from_corpora(
        cls,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame | None,
        min_review_length: int,
    ) -> tuple[int, int, int, int]:
        def original_rank_percentile(values: list[int], default: int) -> int:
            if not values:
                return default
            sorted_values = np.sort(np.asarray(values, dtype=np.int64))
            index = max(int(0.9 * len(sorted_values)) - 1, 0)
            return max(int(sorted_values[index]), 1)

        user_text, item_text = cls._build_original_text_corpora(train_df, valid_df)
        user_counts = [len(reviews) for reviews in user_text.values()]
        item_counts = [len(reviews) for reviews in item_text.values()]
        user_lengths = [len(review) for reviews in user_text.values() for review in reviews]
        item_lengths = [len(review) for reviews in item_text.values() for review in reviews]

        user_review_count = original_rank_percentile(user_counts, 1)
        item_review_count = original_rank_percentile(item_counts, 1)
        user_review_length = max(original_rank_percentile(user_lengths, min_review_length), min_review_length)
        item_review_length = max(original_rank_percentile(item_lengths, min_review_length), min_review_length)
        return user_review_count, item_review_count, user_review_length, item_review_length

    @classmethod
    def _collect_original_side_vocab_tokens(
        cls,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame | None,
        user_review_count: int,
        item_review_count: int,
        user_review_length: int,
        item_review_length: int,
    ) -> tuple[set[str], set[str]]:
        user_text, item_text = cls._build_original_text_corpora(train_df, valid_df)

        def padded_tokens(corpus: dict[int, list[list[str]]], review_count: int, review_length: int) -> set[str]:
            tokens: set[str] = set()
            for reviews in corpus.values():
                adjusted = reviews[:review_count]
                if len(adjusted) < review_count:
                    adjusted = adjusted + [["<PAD/>"] * review_length for _ in range(review_count - len(adjusted))]
                for review in adjusted:
                    padded = review[:review_length] + ["<PAD/>"] * max(0, review_length - len(review))
                    tokens.update(padded)
            return tokens

        return (
            padded_tokens(user_text, user_review_count, user_review_length),
            padded_tokens(item_text, item_review_count, item_review_length),
        )

    @classmethod
    def _get_max_filter_size(cls, configs: Mapping[str, object]) -> int:
        filter_sizes_raw = configs.get("filter_sizes")
        if filter_sizes_raw is None:
            return cls._coerce_int(configs.get("kernel_size", 3), 3)
        if isinstance(filter_sizes_raw, (list, tuple)):
            return max(int(value) for value in filter_sizes_raw)
        return max(int(value.strip()) for value in str(filter_sizes_raw).split(",") if value.strip())

    def _sync_review_shape_to_configs(self, configs: Mapping[str, object]) -> None:
        if not isinstance(configs, MutableMapping):
            return
        configs["user_review_count"] = self.user_review_count
        configs["item_review_count"] = self.item_review_count
        configs["user_review_length"] = self.user_review_length
        configs["item_review_length"] = self.item_review_length
        configs["review_count"] = self.review_count
        configs["review_length"] = self.review_length

    @staticmethod
    def _build_word_to_idx(vocab_tokens: set[str]) -> dict[str, int]:
        # Maps to data_pro.py::build_vocab.
        # The original builds vocab via Counter.most_common() then sorted().
        # most_common() with no argument returns all items (frequency order),
        # and the subsequent sorted() call re-sorts alphabetically, so the
        # final result is purely alphabetical. We skip the no-op most_common()
        # step and sort directly for the same outcome.
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
    def _normalize_text(text: object, preprocess_mode: str = "original") -> str:
        # Maps to data_pro.py::clean_str when preprocess_mode == "original".
        if text is None:
            return ""
        if isinstance(text, (float, np.floating)) and np.isnan(text):
            return ""
        if preprocess_mode != "original":
            return str(text).strip().lower()
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

    def _adjust_review_tokens(self, reviews: list[list[int]], review_count: int, review_length: int) -> list[list[int]]:
        # Maps to data_pro.py::pad_sentences.
        adjusted = reviews[:review_count]
        if len(adjusted) < review_count:
            adjusted = adjusted + [[self.pad_idx] * review_length for _ in range(review_count - len(adjusted))]
        return [review[:review_length] + [self.pad_idx] * max(0, review_length - len(review)) for review in adjusted]

    def _adjust_side_ids(self, side_ids: list[int], padding_id: int, review_count: int) -> list[int]:
        # Maps to data_pro.py::pad_reviewid.
        adjusted = side_ids[:review_count]
        if len(adjusted) < review_count:
            adjusted = adjusted + [padding_id] * (review_count - len(adjusted))
        return adjusted

    def _select_review_entries(
        self,
        lookup: ReviewLookup,
        query_id: int,
        exclude_id: int,
    ) -> list[ReviewEntry]:
        # Maps to data_pro.py::load_data_and_labels cold-start handling
        # (u_text / i_text initialisation with [['<PAD/>']] and [int(0)]).
        entries = lookup.get(int(query_id), [])
        if not entries and self.original_cold_start_dummy:
            return [(0, [self.pad_idx])]
        if self.retain_rui:
            return list(entries)
        return [entry for entry in entries if entry[0] != int(exclude_id)]

    def _load_reviews(self, lookup: ReviewLookup, query_id: int, exclude_id: int, review_count: int, review_length: int) -> torch.Tensor:
        selected_entries = self._select_review_entries(lookup, query_id, exclude_id)
        selected_reviews = [tokens for _, tokens in selected_entries]
        return torch.tensor(self._adjust_review_tokens(selected_reviews, review_count, review_length), dtype=torch.long)

    def _load_review_ids(
        self, lookup: ReviewLookup, query_id: int, exclude_id: int, padding_id: int, review_count: int
    ) -> torch.Tensor:
        selected_entries = self._select_review_entries(lookup, query_id, exclude_id)
        selected_ids = [other_id for other_id, _ in selected_entries]
        return torch.tensor(self._adjust_side_ids(selected_ids, padding_id, review_count), dtype=torch.long)

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
            user_reviews.append(
                self._load_reviews(
                    review_lookup_by_user,
                    query_id=user_id,
                    exclude_id=item_id,
                    review_count=self.user_review_count,
                    review_length=self.user_review_length,
                )
            )
            item_reviews.append(
                self._load_reviews(
                    review_lookup_by_item,
                    query_id=item_id,
                    exclude_id=user_id,
                    review_count=self.item_review_count,
                    review_length=self.item_review_length,
                )
            )
            user_review_item_ids.append(
                self._load_review_ids(
                    review_lookup_by_user,
                    query_id=user_id,
                    exclude_id=item_id,
                    padding_id=self.item_review_padding_id,
                    review_count=self.user_review_count,
                )
            )
            item_review_user_ids.append(
                self._load_review_ids(
                    review_lookup_by_item,
                    query_id=item_id,
                    exclude_id=user_id,
                    padding_id=self.user_review_padding_id,
                    review_count=self.item_review_count,
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

        cold_start_rows = self._count_cold_start_rows(train_dataset)

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
        self._print_preprocess_summary(self.split, cold_start_rows=cold_start_rows)

    def _count_cold_start_rows(self, train_dataset: NARREDataset) -> int:
        train_users = set(train_dataset.review_lookup_by_user.keys())
        train_items = set(train_dataset.review_lookup_by_item.keys())
        count = 0
        for user_id, item_id, _, _ in self.interactions:
            if user_id not in train_users or item_id not in train_items:
                count += 1
        return count

    def _padding_review_ratio(self) -> float:
        if self.user_review_tensors is None or self.item_review_tensors is None:
            return 0.0
        user_padding_reviews = self.user_review_tensors.eq(self.pad_idx).all(dim=2).sum().item()
        item_padding_reviews = self.item_review_tensors.eq(self.pad_idx).all(dim=2).sum().item()
        total_reviews = self.user_review_tensors.size(0) * self.user_review_tensors.size(1)
        total_reviews += self.item_review_tensors.size(0) * self.item_review_tensors.size(1)
        if total_reviews == 0:
            return 0.0
        return float((user_padding_reviews + item_padding_reviews) / total_reviews)

    def _print_preprocess_summary(self, split: str, cold_start_rows: int) -> None:
        print(
            "[NARRE preprocess] split={}, mode={}, review_num_u={}, review_num_i={}, "
            "review_len_u={}, review_len_i={}, user_vocab_size={}, item_vocab_size={}, "
            "cold_start_rows={}, padding_review_ratio={:.6f}".format(
                split,
                self.preprocess_mode,
                self.user_review_count,
                self.item_review_count,
                self.user_review_length,
                self.item_review_length,
                len(self.user_word_to_idx),
                len(self.item_word_to_idx),
                cold_start_rows,
                self._padding_review_ratio(),
            )
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
