from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from .abstract_dataset import RecDataset


class DAMLDataset(RecDataset):
    """DAML document-level review dataset.

    This dataset reproduces the preprocessing pipeline of the original
    DAML implementation (Neu-Review-Rec, KDD'19):

    1. ``data_pro.py::clean_str``          -> ``_normalize_text``
    2. ``data_pro.py::build_doc``          -> ``_build_documents``
    3. ``data_pro.py::padding_doc``        -> ``_pad_doc``
    4. ``data_pro.py::build_vocab``        -> ``_build_word_to_idx``
    5. ``data_pro.py::word2vec loading``   -> ``_load_glove``

    Unlike NARRE which uses review-level features, DAML concatenates all
    reviews into a single document per user/item with a ``<sep>`` token.
    """

    _glove_cache: Dict[str, Tuple[Dict[str, int], torch.Tensor, int]] = {}

    def __init__(
        self,
        df: pd.DataFrame,
        configs: Mapping[str, object],
        split: str = "train",
        train_dataset: DAMLDataset | None = None,
    ) -> None:
        super().__init__(df, configs, split)
        self.doc_len = int(configs.get("doc_len", 500))
        self.word_dim = int(configs.get("word_dim", 300))
        self.glove_path = str(
            configs.get(
                "glove_path",
                "/home/infolab/mnt/mingyu/review_rec/cached/others/GoogleNews-vectors-negative300.txt",
            )
        )
        self.retain_rui = bool(configs.get("retain_rui", False))

        self.user_ids = torch.as_tensor(df["user_id"].to_numpy(dtype=np.int64), dtype=torch.long)
        self.item_ids = torch.as_tensor(df["item_id"].to_numpy(dtype=np.int64), dtype=torch.long)
        self.ratings = torch.as_tensor(df["rating"].to_numpy(dtype=np.float32), dtype=torch.float32)

        if train_dataset is not None:
            self.pad_idx = train_dataset.pad_idx
            self.word_to_idx = train_dataset.word_to_idx
            self.embedding_matrix = train_dataset.embedding_matrix.clone()
        else:
            vocab, embedding_matrix, self.pad_idx = self._load_glove(self.glove_path, self.word_dim)
            self.word_to_idx = vocab
            self.embedding_matrix = embedding_matrix

        # Build interactions: (user_id, item_id, review_text)
        self.interactions: List[Tuple[int, int, str]] = []
        for row in df.itertuples(index=False):
            review_text = self._normalize_text(getattr(row, "reviewText", ""))
            self.interactions.append(
                (
                    int(getattr(row, "user_id")),
                    int(getattr(row, "item_id")),
                    review_text,
                )
            )

        # Build review lookups
        self.review_lookup_by_user = self._build_review_lookups(self.interactions, use_user_key=True)
        self.review_lookup_by_item = self._build_review_lookups(self.interactions, use_user_key=False)

        self.user_doc_tensors: Optional[torch.Tensor] = None
        self.item_doc_tensors: Optional[torch.Tensor] = None

        if split == "train":
            self.user_doc_tensors, self.item_doc_tensors = self._build_doc_tensors(
                self.interactions,
                self.review_lookup_by_user,
                self.review_lookup_by_item,
            )
            self._print_preprocess_summary(split)
        else:
            self._setup_evaluation_from_train(train_dataset)

    @classmethod
    def _load_glove(cls, glove_path: str, word_dim: int) -> Tuple[Dict[str, int], torch.Tensor, int]:
        cache_key = f"{glove_path}:{word_dim}"
        if cache_key in cls._glove_cache:
            return cls._glove_cache[cache_key]

        # DAML original initializes word embeddings uniformly in [-1, 1]
        # then overwrites with pre-trained vectors (same as NARRE)
        word_to_idx: Dict[str, int] = {"<pad>": 0, "<sep>": 1}
        vectors: List[List[float]] = [[0.0] * word_dim, [0.0] * word_dim]

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
        # Initialize <sep> with small random values (not in GloVe)
        embedding_matrix[1] = torch.empty(word_dim).uniform_(-0.1, 0.1)

        cached_value = (word_to_idx, embedding_matrix, 0)
        cls._glove_cache[cache_key] = cached_value
        return cached_value

    @staticmethod
    def _normalize_text(text: object) -> str:
        # Maps to data_pro.py::clean_str (DAML version allows numbers)
        if text is None:
            return ""
        if isinstance(text, (float, np.floating)) and np.isnan(text):
            return ""
        string = re.sub(r"[^A-Za-z0-9]", " ", str(text))
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        if not text:
            return []
        return text.split(" ")

    def _tokens_to_ids(self, text: str) -> List[int]:
        return [self.word_to_idx.get(token, self.pad_idx) for token in self._tokenize(text)]

    @staticmethod
    def _build_review_lookups(
        interactions: List[Tuple[int, int, str]],
        use_user_key: bool,
    ) -> Dict[int, List[Tuple[int, str]]]:
        review_lookup: Dict[int, List[Tuple[int, str]]] = {}
        for user_id, item_id, review_text in interactions:
            query_id = user_id if use_user_key else item_id
            other_id = item_id if use_user_key else user_id
            review_lookup.setdefault(query_id, []).append((other_id, review_text))
        return review_lookup

    def _build_document(self, reviews: List[str]) -> List[int]:
        """Concatenate reviews with <sep> token and convert to ids.

        Maps to data_pro.py::build_doc (document concatenation logic).
        """
        tokens: List[str] = []
        for i, review in enumerate(reviews):
            if i > 0:
                tokens.append("<sep>")
            tokens.extend(self._tokenize(review))
        token_ids = [self.word_to_idx.get(t, self.pad_idx) for t in tokens]
        # Pad or truncate to doc_len
        if len(token_ids) < self.doc_len:
            token_ids = token_ids + [self.pad_idx] * (self.doc_len - len(token_ids))
        else:
            token_ids = token_ids[: self.doc_len]
        return token_ids

    def _load_doc(self, lookup: Dict[int, List[Tuple[int, str]]], query_id: int, exclude_id: int) -> torch.Tensor:
        entries = lookup.get(int(query_id), [])
        if self.retain_rui:
            selected_reviews = [text for _, text in entries]
        else:
            selected_reviews = [text for other_id, text in entries if other_id != int(exclude_id)]
        if not selected_reviews:
            selected_reviews = [""]
        doc_ids = self._build_document(selected_reviews)
        return torch.tensor(doc_ids, dtype=torch.long)

    def _build_doc_tensors(
        self,
        target_interactions: List[Tuple[int, int, str]],
        review_lookup_by_user: Dict[int, List[Tuple[int, str]]],
        review_lookup_by_item: Dict[int, List[Tuple[int, str]]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        user_docs: List[torch.Tensor] = []
        item_docs: List[torch.Tensor] = []

        for user_id, item_id, _ in target_interactions:
            user_docs.append(self._load_doc(review_lookup_by_user, user_id, item_id))
            item_docs.append(self._load_doc(review_lookup_by_item, item_id, user_id))

        return torch.stack(user_docs), torch.stack(item_docs)

    def _setup_evaluation_from_train(self, train_dataset: DAMLDataset | None) -> None:
        if self.split not in {"valid", "test"} or train_dataset is None:
            return

        (
            self.user_doc_tensors,
            self.item_doc_tensors,
        ) = self._build_doc_tensors(
            self.interactions,
            train_dataset.review_lookup_by_user,
            train_dataset.review_lookup_by_item,
        )
        self._print_preprocess_summary(self.split)

    def _print_preprocess_summary(self, split: str) -> None:
        vocab_size = len(self.word_to_idx)
        print(
            f"[DAML preprocess] split={split}, doc_len={self.doc_len}, "
            f"vocab_size={vocab_size}, word_dim={self.word_dim}"
        )

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.user_doc_tensors is None or self.item_doc_tensors is None:
            raise RuntimeError("Document tensors must be initialized before accessing DAML samples.")

        return {
            "user_id": self.user_ids[idx].clone().detach(),
            "item_id": self.item_ids[idx].clone().detach(),
            "rating": self.ratings[idx].clone().detach(),
            "user_doc": self.user_doc_tensors[idx].clone().detach(),
            "item_doc": self.item_doc_tensors[idx].clone().detach(),
        }
