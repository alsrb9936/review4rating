from __future__ import annotations

import hashlib
import os
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

    Word embeddings are cached to disk under
    ``{embedding_path}/{dataset}/daml_word_embeddings_{word_dim}.pt``
    so that subsequent training runs do not re-read the multi-gigabyte
    GloVe file.
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
        self.embedding_path = str(configs.get("embedding_path", "/home/infolab/mnt/mingyu/review_rec/cached/embedding"))
        self.dataset_name = str(configs.get("dataset", "unknown"))
        self.retain_rui = bool(configs.get("retain_rui", False))

        self.user_ids = torch.as_tensor(df["user_id"].to_numpy(dtype=np.int64), dtype=torch.long)
        self.item_ids = torch.as_tensor(df["item_id"].to_numpy(dtype=np.int64), dtype=torch.long)
        self.ratings = torch.as_tensor(df["rating"].to_numpy(dtype=np.float32), dtype=torch.float32)

        # Build interactions first so we can derive vocab from review text
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

        if train_dataset is not None:
            self.pad_idx = train_dataset.pad_idx
            self.word_to_idx = train_dataset.word_to_idx
            self.embedding_matrix = train_dataset.embedding_matrix.clone()
        else:
            vocab_tokens = self._build_vocab_from_reviews(df)
            cache_path = self._word_embedding_cache_path()
            if cache_path and os.path.exists(cache_path):
                print(f"[DAML] Loading cached word embeddings from {cache_path}")
                cached = torch.load(cache_path, map_location="cpu", weights_only=False)
                self.word_to_idx = cached["word_to_idx"]
                self.embedding_matrix = cached["embedding_matrix"]
                self.pad_idx = int(cached.get("pad_idx", 0))
            else:
                vocab, embedding_matrix, self.pad_idx = self._load_glove_for_vocab(
                    self.glove_path, self.word_dim, vocab_tokens
                )
                self.word_to_idx = vocab
                self.embedding_matrix = embedding_matrix
                if cache_path:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    torch.save(
                        {
                            "word_to_idx": self.word_to_idx,
                            "embedding_matrix": self.embedding_matrix,
                            "pad_idx": self.pad_idx,
                        },
                        cache_path,
                    )
                    print(f"[DAML] Saved word embeddings to {cache_path}")

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
        elif train_dataset is not None:
            # valid/test with train_dataset passed: build tensors using train lookups
            self.user_doc_tensors, self.item_doc_tensors = self._build_doc_tensors(
                self.interactions,
                train_dataset.review_lookup_by_user,
                train_dataset.review_lookup_by_item,
            )
            self._print_preprocess_summary(split)

    def _word_embedding_cache_path(self) -> str | None:
        glove_name = os.path.basename(self.glove_path).replace(".txt", "").replace(".bin", "")
        return (
            f"{self.embedding_path}/{self.dataset_name}/"
            f"daml_word_embeddings_{glove_name}_{self.word_dim}.pt"
        )

    @classmethod
    def _build_vocab_from_reviews(cls, df: pd.DataFrame) -> set[str]:
        """Collect unique tokens from all review texts in the dataset."""
        vocab_tokens: set[str] = set()
        for row in df.itertuples(index=False):
            text = cls._normalize_text(getattr(row, "reviewText", ""))
            vocab_tokens.update(cls._tokenize(text))
        vocab_tokens.discard("")
        return vocab_tokens

    @classmethod
    def _load_glove_for_vocab(
        cls,
        glove_path: str,
        word_dim: int,
        vocab_tokens: set[str],
    ) -> Tuple[Dict[str, int], torch.Tensor, int]:
        """Load GloVe embeddings only for tokens present in the dataset vocab.

        This prevents embedding tables from growing to millions of parameters
        when the pretrained file contains 3M+ words.
        """
        cache_key = (glove_path, word_dim, frozenset(vocab_tokens))
        if cache_key in cls._glove_cache:
            return cls._glove_cache[cache_key]

        word_to_idx: Dict[str, int] = {"<pad>": 0, "<sep>": 1}
        # Start with zero vectors; will overwrite with GloVe or random init
        vectors: List[List[float]] = [[0.0] * word_dim, [0.0] * word_dim]
        found_in_glove: set[str] = set()

        with open(glove_path, "r", encoding="utf-8") as glove_file:
            for line in glove_file:
                parts = line.rstrip().split()
                if len(parts) != word_dim + 1:
                    continue
                token = parts[0]
                if token not in vocab_tokens or token in word_to_idx:
                    continue
                values = [float(value) for value in parts[1:]]
                word_to_idx[token] = len(vectors)
                vectors.append(values)
                found_in_glove.add(token)

        # OOV tokens: initialize with uniform random in [-1, 1]
        oov_tokens = vocab_tokens - found_in_glove
        for token in sorted(oov_tokens):
            if token not in word_to_idx:
                word_to_idx[token] = len(vectors)
                vectors.append(np.random.uniform(-1.0, 1.0, word_dim).tolist())

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

    def _setup_evaluation(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        if self.split not in {"valid", "test"}:
            return

        # Rebuild train interactions from train_df to construct review lookups
        history_interactions: List[Tuple[int, int, str]] = []
        for row in train_df.itertuples(index=False):
            review_text = self._normalize_text(getattr(row, "reviewText", ""))
            history_interactions.append(
                (
                    int(getattr(row, "user_id")),
                    int(getattr(row, "item_id")),
                    review_text,
                )
            )

        review_lookup_by_user = self._build_review_lookups(history_interactions, use_user_key=True)
        review_lookup_by_item = self._build_review_lookups(history_interactions, use_user_key=False)

        (
            self.user_doc_tensors,
            self.item_doc_tensors,
        ) = self._build_doc_tensors(
            self.interactions,
            review_lookup_by_user,
            review_lookup_by_item,
        )
        self._print_preprocess_summary(self.split)

    def _print_preprocess_summary(self, split: str) -> None:
        vocab_size = len(self.word_to_idx)
        embed_params = vocab_size * self.word_dim
        print(
            f"[DAML preprocess] split={split}, doc_len={self.doc_len}, "
            f"vocab_size={vocab_size}, embedding_params={embed_params}, word_dim={self.word_dim}"
        )

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.user_doc_tensors is None or self.item_doc_tensors is None:
            raise RuntimeError("Document tensors must be initialized before accessing DAML samples.")

        sample: dict[str, torch.Tensor] = {
            "user_id": self.user_ids[idx].clone().detach(),
            "item_id": self.item_ids[idx].clone().detach(),
            "rating": self.ratings[idx].clone().detach(),
            "user_doc": self.user_doc_tensors[idx].clone().detach(),
            "item_doc": self.item_doc_tensors[idx].clone().detach(),
        }
        return sample
