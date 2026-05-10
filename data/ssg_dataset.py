from __future__ import annotations

# pyright: reportImplicitOverride=false, reportMissingTypeStubs=false, reportExplicitAny=false, reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false

import re
from collections.abc import Mapping

import numpy as np
import pandas as pd
import torch

from .abstract_dataset import RecDataset


Interaction = tuple[int, int, float, float, list[int], str]
ReviewLookup = dict[int, list[Interaction]]


class SSGDataset(RecDataset):
    _glove_cache: dict[tuple[str, int, frozenset[str]], torch.Tensor] = {}
    review_length: int
    review_count: int
    seq_count: int
    word_dim: int
    glove_path: str
    retain_rui: bool
    use_graph_view: bool
    max_rel_bucket: int
    time_percentile: float
    user_review_padding_id: int
    item_review_padding_id: int
    graph_edge_count: int
    user_ids: torch.Tensor
    item_ids: torch.Tensor
    ratings: torch.Tensor
    timestamps: torch.Tensor
    pad_idx: int
    word_to_idx: dict[str, int]
    embedding_matrix: torch.Tensor
    interactions: list[Interaction]
    review_lookup_by_user: ReviewLookup
    review_lookup_by_item: ReviewLookup
    user_review_tensors: torch.Tensor | None
    item_review_tensors: torch.Tensor | None
    user_review_item_ids: torch.Tensor | None
    item_review_user_ids: torch.Tensor | None
    user_seq_reviews: torch.Tensor | None
    item_seq_reviews: torch.Tensor | None
    user_seq_len: torch.Tensor | None
    item_seq_len: torch.Tensor | None
    user_pos_ind: torch.Tensor | None
    item_pos_ind: torch.Tensor | None
    user_rel_dt: torch.Tensor | None
    item_rel_dt: torch.Tensor | None
    user_abs_dt: torch.Tensor | None
    item_abs_dt: torch.Tensor | None
    graph_adj: torch.Tensor | None
    graph_reviews: torch.Tensor | None
    graph_ratings: torch.Tensor | None
    time_scale: float

    def __init__(self, df: pd.DataFrame, configs: Mapping[str, object], split: str = "train") -> None:
        if "timestamp" not in df.columns:
            raise ValueError("SSG requires 'timestamp' column in dataset. The .inter file must have 'timestamp:float' column.")

        super().__init__(df, configs, split)
        self.review_length = self._coerce_int(configs.get("review_length", 40), 40)
        self.review_count = self._coerce_int(configs.get("review_count", 10), 10)
        self.seq_count = self._coerce_int(configs.get("seq_count", self.review_count), self.review_count)
        self.word_dim = self._coerce_int(configs.get("word_dim", 300), 300)
        self.glove_path = self._coerce_str(
            configs.get(
                "glove_path",
                "/home/infolab/mnt/mingyu/review_rec/cached/others/GoogleNews-vectors-negative300.txt",
            ),
            "/home/infolab/mnt/mingyu/review_rec/cached/others/GoogleNews-vectors-negative300.txt",
        )
        self.retain_rui = self._coerce_bool(configs.get("retain_rui", False), False)
        self.use_graph_view = self._coerce_bool(configs.get("use_graph_view", False), False)
        self.max_rel_bucket = 149
        self.time_percentile = 90.0

        self.user_review_padding_id = self.num_users
        self.item_review_padding_id = self.num_items
        self.graph_edge_count = max(1, self.review_count * 2)

        self.user_ids = torch.tensor(df["user_id"].tolist(), dtype=torch.long)
        self.item_ids = torch.tensor(df["item_id"].tolist(), dtype=torch.long)
        self.ratings = torch.tensor(df["rating"].tolist(), dtype=torch.float32)
        self.timestamps = torch.tensor(df["timestamp"].tolist(), dtype=torch.float32)

        vocab_tokens = self._collect_vocab_tokens(df)
        self.pad_idx = 0
        self.word_to_idx = self._build_word_to_idx(vocab_tokens)
        self.embedding_matrix = self._load_glove(
            self.glove_path,
            self.word_dim,
            vocab_tokens,
            self.word_to_idx,
        )

        self.interactions = self._build_interactions(df)
        self.review_lookup_by_user = self._build_review_lookups(self.interactions, use_user_key=True)
        self.review_lookup_by_item = self._build_review_lookups(self.interactions, use_user_key=False)

        self.user_review_tensors = None
        self.item_review_tensors = None
        self.user_review_item_ids = None
        self.item_review_user_ids = None
        self.user_seq_reviews = None
        self.item_seq_reviews = None
        self.user_seq_len = None
        self.item_seq_len = None
        self.user_pos_ind = None
        self.item_pos_ind = None
        self.user_rel_dt = None
        self.item_rel_dt = None
        self.user_abs_dt = None
        self.item_abs_dt = None
        self.graph_adj = None
        self.graph_reviews = None
        self.graph_ratings = None
        self.time_scale = 1.0

        if split == "train":
            self._initialize_context(self.interactions, self.review_lookup_by_user, self.review_lookup_by_item)

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

        embedding_matrix = torch.randn(len(word_to_idx), word_dim, dtype=torch.float32) * 0.01
        embedding_matrix[0] = 0.0

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
            vocab_tokens.update(cls._tokenize(cls._normalize_text(getattr(row, "reviewText", ""))))
        return vocab_tokens

    @staticmethod
    def _build_word_to_idx(vocab_tokens: set[str]) -> dict[str, int]:
        word_to_idx = {"<pad>": 0}
        for idx, token in enumerate(sorted(vocab_tokens), start=1):
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
        return str(text).strip().lower()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9']+", text)

    def _tokens_to_ids(self, text: str) -> list[int]:
        return [self.word_to_idx.get(token, self.pad_idx) for token in self._tokenize(text)]

    def _build_interactions(self, frame: pd.DataFrame) -> list[Interaction]:
        interactions: list[Interaction] = []
        for row in frame.itertuples(index=False):
            review_text = self._normalize_text(getattr(row, "reviewText", ""))
            interactions.append(
                (
                    int(getattr(row, "user_id")),
                    int(getattr(row, "item_id")),
                    float(getattr(row, "rating")),
                    float(getattr(row, "timestamp")),
                    self._tokens_to_ids(review_text),
                    review_text,
                )
            )
        return interactions

    @staticmethod
    def _build_review_lookups(interactions: list[Interaction], use_user_key: bool) -> ReviewLookup:
        review_lookup: ReviewLookup = {}
        for interaction in interactions:
            key = interaction[0] if use_user_key else interaction[1]
            review_lookup.setdefault(key, []).append(interaction)
        return review_lookup

    @staticmethod
    def _is_target_interaction(interaction: Interaction, user_id: int, item_id: int) -> bool:
        return interaction[0] == int(user_id) and interaction[1] == int(item_id)

    def _initialize_context(
        self,
        history_interactions: list[Interaction],
        review_lookup_by_user: ReviewLookup,
        review_lookup_by_item: ReviewLookup,
    ) -> None:
        del history_interactions
        (
            self.user_review_tensors,
            self.item_review_tensors,
            self.user_review_item_ids,
            self.item_review_user_ids,
        ) = self._build_set_context_tensors(self.interactions, review_lookup_by_user, review_lookup_by_item)

        self.time_scale = self._estimate_time_scale(self.interactions, review_lookup_by_user, review_lookup_by_item)
        (
            self.user_seq_reviews,
            self.item_seq_reviews,
            self.user_seq_len,
            self.item_seq_len,
            self.user_pos_ind,
            self.item_pos_ind,
            self.user_rel_dt,
            self.item_rel_dt,
            self.user_abs_dt,
            self.item_abs_dt,
        ) = self._build_sequence_context_tensors(self.interactions, review_lookup_by_user, review_lookup_by_item, self.time_scale)

        if self.use_graph_view:
            self.graph_adj, self.graph_reviews, self.graph_ratings = self._build_graph_tensors(
                self.interactions,
                review_lookup_by_user,
                review_lookup_by_item,
            )
        else:
            self.graph_adj = None
            self.graph_reviews = None
            self.graph_ratings = None

    def _adjust_review_tokens(self, reviews: list[list[int]], limit: int) -> list[list[int]]:
        adjusted = reviews[:limit]
        if len(adjusted) < limit:
            adjusted = adjusted + [[self.pad_idx] * self.review_length for _ in range(limit - len(adjusted))]
        return [review[: self.review_length] + [self.pad_idx] * max(0, self.review_length - len(review)) for review in adjusted]

    def _adjust_side_ids(self, side_ids: list[int], limit: int, padding_id: int) -> list[int]:
        adjusted = side_ids[:limit]
        if len(adjusted) < limit:
            adjusted = adjusted + [padding_id] * (limit - len(adjusted))
        return adjusted

    def _select_set_entries(self, lookup: ReviewLookup, query_id: int, user_id: int, item_id: int) -> list[Interaction]:
        entries = lookup.get(int(query_id), [])
        if self.retain_rui:
            return list(entries)
        return [entry for entry in entries if not self._is_target_interaction(entry, user_id, item_id)]

    def _select_sequence_entries(
        self,
        lookup: ReviewLookup,
        query_id: int,
        user_id: int,
        item_id: int,
        target_ts: float,
    ) -> list[Interaction]:
        filtered = [entry for entry in lookup.get(int(query_id), []) if entry[3] <= float(target_ts)]
        if not self.retain_rui:
            filtered = [entry for entry in filtered if not self._is_target_interaction(entry, user_id, item_id)]
        filtered.sort(key=lambda entry: entry[3], reverse=True)
        return filtered

    def _build_set_context_tensors(
        self,
        target_interactions: list[Interaction],
        review_lookup_by_user: ReviewLookup,
        review_lookup_by_item: ReviewLookup,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        user_reviews: list[torch.Tensor] = []
        item_reviews: list[torch.Tensor] = []
        user_review_item_ids: list[torch.Tensor] = []
        item_review_user_ids: list[torch.Tensor] = []

        for user_id, item_id, _, _, _, _ in target_interactions:
            user_entries = self._select_set_entries(review_lookup_by_user, user_id, user_id, item_id)
            item_entries = self._select_set_entries(review_lookup_by_item, item_id, user_id, item_id)

            user_reviews.append(
                torch.tensor(self._adjust_review_tokens([entry[4] for entry in user_entries], self.review_count), dtype=torch.long)
            )
            item_reviews.append(
                torch.tensor(self._adjust_review_tokens([entry[4] for entry in item_entries], self.review_count), dtype=torch.long)
            )
            user_review_item_ids.append(
                torch.tensor(
                    self._adjust_side_ids([entry[1] for entry in user_entries], self.review_count, self.item_review_padding_id),
                    dtype=torch.long,
                )
            )
            item_review_user_ids.append(
                torch.tensor(
                    self._adjust_side_ids([entry[0] for entry in item_entries], self.review_count, self.user_review_padding_id),
                    dtype=torch.long,
                )
            )

        return (
            torch.stack(user_reviews),
            torch.stack(item_reviews),
            torch.stack(user_review_item_ids),
            torch.stack(item_review_user_ids),
        )

    def _build_sequence_features(
        self,
        entries: list[Interaction],
        target_ts: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        limited_entries = entries[: self.seq_count]
        seq_len = torch.tensor(min(len(entries), self.seq_count), dtype=torch.long)

        reviews = self._adjust_review_tokens([entry[4] for entry in limited_entries], self.seq_count)
        pos_ind = [position for position in range(1, len(limited_entries) + 1)]
        rel_dt = [min(int(max(float(target_ts) - entry[3], 0.0) / self.time_scale), self.max_rel_bucket) for entry in limited_entries]
        abs_dt = [entry[3] for entry in limited_entries]

        if len(pos_ind) < self.seq_count:
            pad_size = self.seq_count - len(pos_ind)
            pos_ind.extend([0] * pad_size)
            rel_dt.extend([0] * pad_size)
            abs_dt.extend([0.0] * pad_size)

        return (
            torch.tensor(reviews, dtype=torch.long),
            seq_len,
            torch.tensor(pos_ind, dtype=torch.long),
            torch.tensor(rel_dt, dtype=torch.long),
            torch.tensor(abs_dt, dtype=torch.float32),
        )

    def _build_sequence_context_tensors(
        self,
        target_interactions: list[Interaction],
        review_lookup_by_user: ReviewLookup,
        review_lookup_by_item: ReviewLookup,
        time_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del time_scale
        user_seq_reviews: list[torch.Tensor] = []
        item_seq_reviews: list[torch.Tensor] = []
        user_seq_len: list[torch.Tensor] = []
        item_seq_len: list[torch.Tensor] = []
        user_pos_ind: list[torch.Tensor] = []
        item_pos_ind: list[torch.Tensor] = []
        user_rel_dt: list[torch.Tensor] = []
        item_rel_dt: list[torch.Tensor] = []
        user_abs_dt: list[torch.Tensor] = []
        item_abs_dt: list[torch.Tensor] = []

        for user_id, item_id, _, timestamp, _, _ in target_interactions:
            user_entries = self._select_sequence_entries(review_lookup_by_user, user_id, user_id, item_id, timestamp)
            item_entries = self._select_sequence_entries(review_lookup_by_item, item_id, user_id, item_id, timestamp)

            user_reviews, user_len, user_pos, user_rel, user_abs = self._build_sequence_features(user_entries, timestamp)
            item_reviews, item_len, item_pos, item_rel, item_abs = self._build_sequence_features(item_entries, timestamp)

            user_seq_reviews.append(user_reviews)
            item_seq_reviews.append(item_reviews)
            user_seq_len.append(user_len)
            item_seq_len.append(item_len)
            user_pos_ind.append(user_pos)
            item_pos_ind.append(item_pos)
            user_rel_dt.append(user_rel)
            item_rel_dt.append(item_rel)
            user_abs_dt.append(user_abs)
            item_abs_dt.append(item_abs)

        return (
            torch.stack(user_seq_reviews),
            torch.stack(item_seq_reviews),
            torch.stack(user_seq_len),
            torch.stack(item_seq_len),
            torch.stack(user_pos_ind),
            torch.stack(item_pos_ind),
            torch.stack(user_rel_dt),
            torch.stack(item_rel_dt),
            torch.stack(user_abs_dt),
            torch.stack(item_abs_dt),
        )

    def _estimate_time_scale(
        self,
        target_interactions: list[Interaction],
        review_lookup_by_user: ReviewLookup,
        review_lookup_by_item: ReviewLookup,
    ) -> float:
        non_zero_rel_dt: list[float] = []

        for user_id, item_id, _, timestamp, _, _ in target_interactions:
            for entries, query_id in ((review_lookup_by_user, user_id), (review_lookup_by_item, item_id)):
                for entry in self._select_sequence_entries(entries, query_id, user_id, item_id, timestamp):
                    rel_dt = float(timestamp) - entry[3]
                    if rel_dt > 0.0:
                        non_zero_rel_dt.append(rel_dt)

        if not non_zero_rel_dt:
            return 1.0
        return max(float(np.percentile(np.asarray(non_zero_rel_dt, dtype=np.float32), self.time_percentile)), 1.0)

    def _build_graph_tensors(
        self,
        target_interactions: list[Interaction],
        review_lookup_by_user: ReviewLookup,
        review_lookup_by_item: ReviewLookup,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        graph_adj: list[torch.Tensor] = []
        graph_reviews: list[torch.Tensor] = []
        graph_ratings: list[torch.Tensor] = []

        for user_id, item_id, _, _, _, _ in target_interactions:
            selected_user_entries = self._select_set_entries(review_lookup_by_user, user_id, user_id, item_id)
            selected_item_entries = self._select_set_entries(review_lookup_by_item, item_id, user_id, item_id)

            unique_entries: list[Interaction] = []
            seen_edges: set[tuple[int, int, float]] = set()
            for entry in selected_user_entries + selected_item_entries:
                edge_key = (entry[0], entry[1], entry[3])
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                unique_entries.append(entry)
                if len(unique_entries) >= self.graph_edge_count:
                    break

            edge_index = torch.full((2, self.graph_edge_count), -1, dtype=torch.long)
            edge_reviews = torch.full((self.graph_edge_count, self.review_length), self.pad_idx, dtype=torch.long)
            edge_ratings = torch.zeros(self.graph_edge_count, dtype=torch.float32)

            for edge_idx, entry in enumerate(unique_entries):
                edge_index[0, edge_idx] = entry[0]
                edge_index[1, edge_idx] = self.num_users + entry[1]
                edge_reviews[edge_idx] = torch.tensor(
                    self._adjust_review_tokens([entry[4]], 1)[0],
                    dtype=torch.long,
                )
                edge_ratings[edge_idx] = float(entry[2])

            graph_adj.append(edge_index)
            graph_reviews.append(edge_reviews)
            graph_ratings.append(edge_ratings)

        return torch.stack(graph_adj), torch.stack(graph_reviews), torch.stack(graph_ratings)

    def _setup_evaluation(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        del valid_df, test_df
        if self.split not in {"valid", "test"}:
            return

        if "timestamp" not in train_df.columns:
            raise ValueError("SSG requires 'timestamp' column in dataset. The .inter file must have 'timestamp:float' column.")

        history_interactions = self._build_interactions(train_df)
        review_lookup_by_user = self._build_review_lookups(history_interactions, use_user_key=True)
        review_lookup_by_item = self._build_review_lookups(history_interactions, use_user_key=False)
        self._initialize_context(history_interactions, review_lookup_by_user, review_lookup_by_item)

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        required_tensors = (
            self.user_review_tensors,
            self.item_review_tensors,
            self.user_review_item_ids,
            self.item_review_user_ids,
            self.user_seq_reviews,
            self.item_seq_reviews,
            self.user_seq_len,
            self.item_seq_len,
            self.user_pos_ind,
            self.item_pos_ind,
            self.user_rel_dt,
            self.item_rel_dt,
            self.user_abs_dt,
            self.item_abs_dt,
        )
        if any(tensor is None for tensor in required_tensors):
            raise RuntimeError("SSG context tensors must be initialized before accessing samples.")

        user_review_tensors = self.user_review_tensors
        item_review_tensors = self.item_review_tensors
        user_review_item_ids = self.user_review_item_ids
        item_review_user_ids = self.item_review_user_ids
        user_seq_reviews = self.user_seq_reviews
        item_seq_reviews = self.item_seq_reviews
        user_seq_len = self.user_seq_len
        item_seq_len = self.item_seq_len
        user_pos_ind = self.user_pos_ind
        item_pos_ind = self.item_pos_ind
        user_rel_dt = self.user_rel_dt
        item_rel_dt = self.item_rel_dt
        user_abs_dt = self.user_abs_dt
        item_abs_dt = self.item_abs_dt

        assert user_review_tensors is not None
        assert item_review_tensors is not None
        assert user_review_item_ids is not None
        assert item_review_user_ids is not None
        assert user_seq_reviews is not None
        assert item_seq_reviews is not None
        assert user_seq_len is not None
        assert item_seq_len is not None
        assert user_pos_ind is not None
        assert item_pos_ind is not None
        assert user_rel_dt is not None
        assert item_rel_dt is not None
        assert user_abs_dt is not None
        assert item_abs_dt is not None

        sample: dict[str, torch.Tensor] = {
            "user_id": self.user_ids[idx].clone().detach(),
            "item_id": self.item_ids[idx].clone().detach(),
            "rating": self.ratings[idx].clone().detach(),
            "user_review": user_review_tensors[idx].clone().detach(),
            "item_review": item_review_tensors[idx].clone().detach(),
            "user_review_item_ids": user_review_item_ids[idx].clone().detach(),
            "item_review_user_ids": item_review_user_ids[idx].clone().detach(),
            "user_seq_reviews": user_seq_reviews[idx].clone().detach(),
            "item_seq_reviews": item_seq_reviews[idx].clone().detach(),
            "user_seq_len": user_seq_len[idx].clone().detach(),
            "item_seq_len": item_seq_len[idx].clone().detach(),
            "user_pos_ind": user_pos_ind[idx].clone().detach(),
            "item_pos_ind": item_pos_ind[idx].clone().detach(),
            "user_rel_dt": user_rel_dt[idx].clone().detach(),
            "item_rel_dt": item_rel_dt[idx].clone().detach(),
            "user_abs_dt": user_abs_dt[idx].clone().detach(),
            "item_abs_dt": item_abs_dt[idx].clone().detach(),
        }

        if self.use_graph_view:
            if self.graph_adj is None or self.graph_reviews is None or self.graph_ratings is None:
                raise RuntimeError("Graph tensors must be initialized when use_graph_view=True.")
            sample["graph_adj"] = self.graph_adj[idx].clone().detach()
            sample["graph_reviews"] = self.graph_reviews[idx].clone().detach()
            sample["graph_ratings"] = self.graph_ratings[idx].clone().detach()

        return sample
