from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false, reportImplicitOverride=false, reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false

import copy
import os
import pickle
import random
import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .abstract_dataset import RecDataset
from .ssg_time import ssg_time_handler


@dataclass(frozen=True)
class SSGRecord:
    user_id: int
    item_id: int
    review_text: str
    rating: float
    timestamp: float


def clean_str(string: object) -> str:
    text = "" if string is None or (isinstance(string, float) and np.isnan(string)) else str(string)
    text = re.sub(r"[^A-Za-z]", " ", text)
    text = re.sub(r"\'s", " \'s", text)
    text = re.sub(r"\'ve", " \'ve", text)
    text = re.sub(r"n\'t", " n\'t", text)
    text = re.sub(r"\'re", " \'re", text)
    text = re.sub(r"\'d", " \'d", text)
    text = re.sub(r"\'ll", " \'ll", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\(", " ( ", text)
    text = re.sub(r"\)", " ) ", text)
    text = re.sub(r"\?", " ? ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip().lower()


def getSubGraph(target_uid: tuple[int, ...], target_iid: tuple[int, ...], neigh_list: Mapping[int, list[tuple[int, int]]], user_num: int, n_hops: int = 2, sample_num: int = 10) -> tuple[dict[int, list[tuple[int, int]]], set[int]]:
    target_ids: set[int] = set(int(uid) for uid in target_uid)
    for iid in target_iid:
        target_ids.add(int(iid) + user_num)

    target_edges: set[int] = set()
    for _ in range(n_hops):
        cur_list = list(target_ids)
        for target_id in cur_list:
            n_list = copy.deepcopy(neigh_list.get(target_id, []))
            random.shuffle(n_list)
            if len(n_list) > sample_num:
                n_list = n_list[:sample_num]
            for next_id, next_edge in n_list:
                target_ids.add(int(next_id))
                target_edges.add(int(next_edge))

    sub_neigh_list: dict[int, list[tuple[int, int]]] = {}
    for target_id in target_ids:
        sub_neigh_list[target_id] = []
        for next_id, edge_idx in neigh_list.get(target_id, []):
            if next_id in target_ids and edge_idx in target_edges:
                sub_neigh_list[target_id].append((int(next_id), int(edge_idx)))
    return sub_neigh_list, target_ids


class SSGDataset(RecDataset):
    _artifact_cache: dict[str, dict[str, object]] = {}

    def __init__(self, df: pd.DataFrame, configs: Mapping[str, object], split: str = "train", train_dataset: SSGDataset | None = None) -> None:
        if "timestamp" not in df.columns:
            raise ValueError("SSG requires a timestamp column.")
        if "reviewText" not in df.columns:
            raise ValueError("SSG requires reviewText; set use_review_text: true.")
        super().__init__(df, configs, split)

        self.percentile = float(configs.get("time_percentile", configs.get("percentile", 10)))
        self.max_rel = int(configs.get("max_rel_bucket", configs.get("max_rel", 100)))
        self.n_hops = int(configs.get("n_hops", 2))
        self.sample_num = int(configs.get("sample_num", 10))
        self.max_rating = int(configs.get("max_rating", 5))
        self.cache_enabled = self._coerce_bool(configs.get("ssg_cache", True), True)

        if train_dataset is None:
            artifacts = self._build_or_load_artifacts(df)
        else:
            artifacts = train_dataset.artifacts

        self.artifacts = artifacts
        self.para = artifacts["para"]
        self.graph_info = artifacts["graph_info"]

        self.user_num = int(self.para["user_num"])
        self.item_num = int(self.para["item_num"])
        self.node_num = self.user_num + self.item_num
        self.num_users = self.user_num
        self.num_items = self.item_num

        self.review_num_u = int(self.para["review_num_u"])
        self.review_num_i = int(self.para["review_num_i"])
        self.review_len_u = int(self.para["review_len_u"])
        self.review_len_i = int(self.para["review_len_i"])
        self.review_len_g = self.review_len_u
        self.review_count = self.review_num_u
        self.review_length = self.review_len_u

        self.user_vocab = self.para["user_vocab"]
        self.item_vocab = self.para["item_vocab"]
        self.vocabulary_user = self.user_vocab
        self.vocabulary_item = self.item_vocab
        self.vocabulary = self.graph_info["vocabulary"]
        self.pad_idx = int(self.user_vocab.get("<PAD/>", 0))
        self.embedding_matrix = torch.empty((1, int(configs.get("word_dim", 300))), dtype=torch.float32)

        self.u_text = self.para["u_text"]
        self.i_text = self.para["i_text"]
        self.u_time = self.para["u_time"]
        self.i_time = self.para["i_time"]
        self.neigh_list = self.graph_info["neigh_list"]
        self.edge_id1 = self.graph_info["edge_id1"]
        self.edge_id2 = self.graph_info["edge_id2"]
        self.edge_ratings = self.graph_info["edge_ratings"]
        self.edge_reviews = self.graph_info["edge_reviews"]

        self.samples = self._build_samples(df)
        self.user_ids = torch.tensor([sample[0] for sample in self.samples], dtype=torch.long)
        self.item_ids = torch.tensor([sample[1] for sample in self.samples], dtype=torch.long)
        self.ratings = torch.tensor([sample[4] for sample in self.samples], dtype=torch.float32)

    @staticmethod
    def _coerce_bool(value: object, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def _cache_path(self, df: pd.DataFrame) -> str:
        dataset = str(self.configs.get("dataset", "unknown"))
        seed = int(self.configs.get("seed", 42))
        signature = f"n{len(df)}_u{int(df['user_id'].sum())}_i{int(df['item_id'].sum())}_r{int(float(df['rating'].sum()) * 1000)}"
        cache_dir = os.path.join(str(self.configs.get("embedding_path", "cached/embedding")), dataset, "ssg_original")
        return os.path.join(cache_dir, f"para_graph_seed{seed}_{signature}.pkl")

    def _build_or_load_artifacts(self, df: pd.DataFrame) -> dict[str, object]:
        cache_path = self._cache_path(df)
        if self.cache_enabled and cache_path in self._artifact_cache:
            return self._artifact_cache[cache_path]
        if self.cache_enabled and os.path.exists(cache_path):
            with open(cache_path, "rb") as handle:
                loaded = pickle.load(handle)
            self._artifact_cache[cache_path] = loaded
            print(f"Loaded SSG para/graph_info cache from {cache_path}")
            return loaded

        records = self._records_from_df(df)
        para = self._build_para(records)
        graph_info = self._build_graph_info(records, para)
        artifacts: dict[str, object] = {"para": para, "graph_info": graph_info}
        if self.cache_enabled:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as handle:
                pickle.dump(artifacts, handle, protocol=2)
            self._artifact_cache[cache_path] = artifacts
            print(f"Saved SSG para/graph_info cache to {cache_path}")
        return artifacts

    def _records_from_df(self, df: pd.DataFrame) -> list[SSGRecord]:
        records: list[SSGRecord] = []
        for row in df.itertuples(index=False):
            records.append(
                SSGRecord(
                    user_id=int(getattr(row, "user_id")),
                    item_id=int(getattr(row, "item_id")),
                    review_text=str(getattr(row, "reviewText", "")),
                    rating=float(getattr(row, "rating")),
                    timestamp=float(getattr(row, "timestamp")),
                )
            )
        return records

    @staticmethod
    def _percentile_len(lengths: list[int]) -> int:
        if not lengths:
            return 1
        sorted_lengths = np.sort(np.asarray(lengths, dtype=np.int64))
        idx = max(int(0.9 * len(sorted_lengths)) - 1, 0)
        return max(int(sorted_lengths[idx]), 1)

    def _build_para(self, records: list[SSGRecord]) -> dict[str, object]:
        user_meta: dict[int, list[tuple[int, str, float]]] = defaultdict(list)
        item_meta: dict[int, list[tuple[int, str, float]]] = defaultdict(list)
        for record in records:
            user_meta[record.user_id].append((record.item_id, record.review_text, record.timestamp))
            item_meta[record.item_id].append((record.user_id, record.review_text, record.timestamp))

        user_tokens: dict[int, list[tuple[list[str], float]]] = {}
        item_tokens: dict[int, list[tuple[list[str], float]]] = {}
        user_rids: dict[int, list[int]] = {}
        item_rids: dict[int, list[int]] = {}
        future_time = self._future_padding_time(records)

        for uid in range(self.num_users):
            entries = sorted(user_meta.get(uid, []), key=lambda entry: entry[2])
            if not entries:
                user_tokens[uid] = [(["<PAD/>"], future_time)]
                user_rids[uid] = [self.num_items + 1]
                continue
            user_tokens[uid] = [(clean_str(text).split(" ") if clean_str(text) else ["<PAD/>"], timestamp) for _, text, timestamp in entries]
            user_rids[uid] = [int(item_id) for item_id, _, _ in entries]

        for iid in range(self.num_items):
            entries = sorted(item_meta.get(iid, []), key=lambda entry: entry[2])
            if not entries:
                item_tokens[iid] = [(["<PAD/>"], future_time)]
                item_rids[iid] = [self.num_users + 1]
                continue
            item_tokens[iid] = [(clean_str(text).split(" ") if clean_str(text) else ["<PAD/>"], timestamp) for _, text, timestamp in entries]
            item_rids[iid] = [int(user_id) for user_id, _, _ in entries]

        u_len = self._percentile_len([len(values) for values in user_tokens.values()])
        i_len = self._percentile_len([len(values) for values in item_tokens.values()])
        min_review_len = max(self._config_filter_sizes())
        u2_len = max(self._percentile_len([len(tokens) for values in user_tokens.values() for tokens, _ in values]), min_review_len)
        i2_len = max(self._percentile_len([len(tokens) for values in item_tokens.values() for tokens, _ in values]), min_review_len)

        padded_user = self._pad_sentences(user_tokens, u_len, u2_len, future_time)
        padded_item = self._pad_sentences(item_tokens, i_len, i2_len, future_time)
        user_vocab, item_vocab = self._build_dual_vocab(padded_user, padded_item)
        u_text, u_time = self._build_input_data(padded_user, user_vocab)
        i_text, i_time = self._build_input_data(padded_item, item_vocab)

        return {
            "user_num": self.num_users,
            "item_num": self.num_items,
            "review_num_u": int(next(iter(u_text.values())).shape[0]),
            "review_num_i": int(next(iter(i_text.values())).shape[0]),
            "review_len_u": int(next(iter(u_text.values())).shape[1]),
            "review_len_i": int(next(iter(i_text.values())).shape[1]),
            "user_vocab": user_vocab,
            "item_vocab": item_vocab,
            "u_text": u_text,
            "i_text": i_text,
            "u_time": u_time,
            "i_time": i_time,
            "user_rid": user_rids,
            "item_rid": item_rids,
        }

    def _config_filter_sizes(self) -> list[int]:
        value = self.configs.get("filter_sizes", [3])
        if isinstance(value, str):
            parsed = [int(part.strip()) for part in value.strip().strip("[]").split(",") if part.strip()]
            return parsed or [3]
        if isinstance(value, (list, tuple)):
            return [int(part) for part in value] or [3]
        return [3]

    @staticmethod
    def _future_padding_time(records: list[SSGRecord]) -> float:
        if not records:
            return 1.0
        return max(record.timestamp for record in records) + 1.0

    @staticmethod
    def _pad_sentences(text: dict[int, list[tuple[list[str], float]]], review_num: int, review_len: int, future_time: float) -> dict[int, list[tuple[list[str], float]]]:
        padded: dict[int, list[tuple[list[str], float]]] = {}
        for key, reviews in text.items():
            rows: list[tuple[list[str], float]] = []
            for ridx in range(review_num):
                if ridx < len(reviews):
                    tokens, timestamp = reviews[ridx]
                    new_tokens = tokens[:review_len] + ["<PAD/>"] * max(0, review_len - len(tokens))
                    rows.append((new_tokens, float(timestamp)))
                else:
                    rows.append((["<PAD/>"] * review_len, future_time))
            rows.insert(0, (["<PAD/>"] * review_len, 0.0))
            padded[key] = rows
        return padded

    @staticmethod
    def _build_dual_vocab(user_text: dict[int, list[tuple[list[str], float]]], item_text: dict[int, list[tuple[list[str], float]]]) -> tuple[dict[str, int], dict[str, int]]:
        user_words = [tokens for reviews in user_text.values() for tokens, _ in reviews]
        item_words = [tokens for reviews in item_text.values() for tokens, _ in reviews]

        def build_vocab(sentences: list[list[str]]) -> dict[str, int]:
            counts = Counter(word for sentence in sentences for word in sentence)
            vocab_inv = sorted([word for word, _ in counts.most_common()])
            return {word: idx for idx, word in enumerate(vocab_inv)}

        return build_vocab(user_words), build_vocab(item_words)

    @staticmethod
    def _build_input_data(text: dict[int, list[tuple[list[str], float]]], vocab: Mapping[str, int]) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        text_ids: dict[int, np.ndarray] = {}
        time_ids: dict[int, np.ndarray] = {}
        pad_id = int(vocab.get("<PAD/>", 0))
        for key, reviews in text.items():
            text_ids[key] = np.asarray([[int(vocab.get(word, pad_id)) for word in tokens] for tokens, _ in reviews], dtype=np.int64)
            time_ids[key] = np.asarray([[float(timestamp)] for _, timestamp in reviews], dtype=np.float32)
        return text_ids, time_ids

    def _build_graph_info(self, records: list[SSGRecord], para: Mapping[str, object]) -> dict[str, object]:
        review_len = int(para["review_len_u"])
        edge_reviews_words: list[list[str]] = []
        edge_ratings: list[int] = []
        edge_id1: list[int] = []
        edge_id2: list[int] = []
        neigh_list: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)
        for edge_idx, record in enumerate(records):
            review = clean_str(record.review_text).split(" ") if clean_str(record.review_text) else ["<PAD/>"]
            review = review[:review_len] + ["<PAD/>"] * max(0, review_len - len(review))
            rating = int(float(record.rating))
            edge_id1.append(record.user_id)
            edge_id2.append(record.item_id)
            edge_ratings.append(max(1, min(self.max_rating, rating)))
            edge_reviews_words.append(review)
            neigh_list[record.user_id].append((record.item_id + self.num_users, edge_idx))
            neigh_list[record.item_id + self.num_users].append((record.user_id, edge_idx))
        vocabulary = self._build_single_vocab(edge_reviews_words)
        pad_id = int(vocabulary.get("<PAD/>", 0))
        edge_reviews = [[int(vocabulary.get(word, pad_id)) for word in review] for review in edge_reviews_words]
        for node in range(self.num_users + self.num_items):
            neigh_list[node] = neigh_list[node]
        return {
            "neigh_list": dict(neigh_list),
            "edge_id1": edge_id1,
            "edge_id2": edge_id2,
            "edge_ratings": edge_ratings,
            "edge_reviews": edge_reviews,
            "vocabulary": vocabulary,
        }

    @staticmethod
    def _build_single_vocab(sentences: list[list[str]]) -> dict[str, int]:
        counts = Counter(word for sentence in sentences for word in sentence)
        vocab_inv = sorted([word for word, _ in counts.most_common()])
        return {word: idx for idx, word in enumerate(vocab_inv)}

    def _build_samples(self, df: pd.DataFrame) -> list[tuple[int, int, list[int], list[int], float, float]]:
        user_rid = self.para["user_rid"]
        item_rid = self.para["item_rid"]
        samples: list[tuple[int, int, list[int], list[int], float, float]] = []
        for row in df.itertuples(index=False):
            uid = int(getattr(row, "user_id"))
            iid = int(getattr(row, "item_id"))
            reuid = self._pad_review_ids(list(user_rid.get(uid, [self.num_items + 1])), self.review_num_u, self.num_items + 1)
            reiid = self._pad_review_ids(list(item_rid.get(iid, [self.num_users + 1])), self.review_num_i, self.num_users + 1)
            samples.append((uid, iid, reuid, reiid, float(getattr(row, "rating")), float(getattr(row, "timestamp"))))
        return samples

    @staticmethod
    def _pad_review_ids(ids: list[int], target_len_with_sentinel: int, pad_value: int) -> list[int]:
        limit = max(target_len_with_sentinel - 1, 0)
        values = ids[:limit] + [pad_value] * max(0, limit - len(ids))
        values.insert(0, pad_value)
        return values

    def _setup_evaluation(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        del train_df, valid_df, test_df

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        uid, iid, reuid, reiid, y, timestamp = self.samples[idx]
        u_s_renum, u_pos_ind, u_rel_dt, u_abs_dt = ssg_time_handler(self.u_time[uid], timestamp, self.percentile, self.max_rel)
        i_s_renum, i_pos_ind, i_rel_dt, i_abs_dt = ssg_time_handler(self.i_time[iid], timestamp, self.percentile, self.max_rel)
        return {
            "_dataset": self,
            "user_id": uid,
            "item_id": iid,
            "rating": y,
            "timestamp": timestamp,
            "input_u": self.u_text[uid],
            "input_i": self.i_text[iid],
            "reuid": np.asarray(reuid, dtype=np.int64),
            "reiid": np.asarray(reiid, dtype=np.int64),
            "u_s_renum": u_s_renum,
            "i_s_renum": i_s_renum,
            "u_pos_ind": u_pos_ind,
            "i_pos_ind": i_pos_ind,
            "u_rel_dt": u_rel_dt,
            "i_rel_dt": i_rel_dt,
            "u_abs_dt": u_abs_dt,
            "i_abs_dt": i_abs_dt,
        }

    def collate_fn(self, data_in: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        uids_tuple = tuple(int(sample["user_id"]) for sample in data_in)
        iids_tuple = tuple(int(sample["item_id"]) for sample in data_in)
        sub_neigh_list, target_ids = getSubGraph(uids_tuple, iids_tuple, self.neigh_list, self.user_num, self.n_hops, self.sample_num)
        target_ids_list = list(target_ids)
        nodes = torch.tensor(target_ids_list, dtype=torch.long)
        adj = torch.zeros((len(target_ids_list), len(target_ids_list)), dtype=torch.float32)
        nodes_map = {node: idx for idx, node in enumerate(target_ids_list)}

        tmp_reviews: list[list[int]] = []
        tmp_rating_ids: list[int] = []
        for node in target_ids_list:
            neighs = []
            for neigh, edge in sub_neigh_list.get(node, []):
                neighs.append((nodes_map[node], nodes_map[neigh], edge))
            neighs = sorted(neighs, key=lambda value: value[1])
            for row_idx, col_idx, edge in neighs:
                adj[row_idx, col_idx] = 1.0
                tmp_reviews.append(self.edge_reviews[edge])
                tmp_rating_ids.append(int(self.edge_ratings[edge]) - 1)

        if tmp_reviews:
            reviews = torch.tensor(tmp_reviews, dtype=torch.long)
            rating_onehot = np.zeros((len(tmp_rating_ids), self.max_rating), dtype=np.float32)
            rating_onehot[np.arange(len(tmp_rating_ids)), tmp_rating_ids] = 1.0
            ratings = torch.tensor(rating_onehot, dtype=torch.float32)
        else:
            reviews = torch.zeros((0, self.review_len_g), dtype=torch.long)
            ratings = torch.zeros((0, self.max_rating), dtype=torch.float32)

        pairs = torch.tensor([(nodes_map[uid], nodes_map[iid + self.user_num]) for uid, iid in zip(uids_tuple, iids_tuple)], dtype=torch.long)
        uids = torch.tensor(uids_tuple, dtype=torch.long)
        iids = torch.tensor([iid + self.user_num for iid in iids_tuple], dtype=torch.long)

        def stack_long(key: str) -> torch.Tensor:
            return torch.tensor(np.asarray([sample[key] for sample in data_in]), dtype=torch.long)

        def stack_float(key: str) -> torch.Tensor:
            return torch.tensor(np.asarray([sample[key] for sample in data_in]), dtype=torch.float32)

        batch = {
            "nodes": nodes,
            "reviews": reviews,
            "ratings": ratings,
            "adj": adj,
            "pairs": pairs,
            "user_id": uids,
            "item_id": iids,
            "raw_item_id": torch.tensor(iids_tuple, dtype=torch.long),
            "input_u": stack_long("input_u"),
            "input_i": stack_long("input_i"),
            "reuid": stack_long("reuid"),
            "reiid": stack_long("reiid"),
            "u_s_renum": torch.tensor([int(sample["u_s_renum"]) for sample in data_in], dtype=torch.long),
            "i_s_renum": torch.tensor([int(sample["i_s_renum"]) for sample in data_in], dtype=torch.long),
            "u_pos_ind": stack_long("u_pos_ind"),
            "i_pos_ind": stack_long("i_pos_ind"),
            "u_rel_dt": stack_long("u_rel_dt"),
            "i_rel_dt": stack_long("i_rel_dt"),
            "u_abs_dt": stack_float("u_abs_dt"),
            "i_abs_dt": stack_float("i_abs_dt"),
            "rating": torch.tensor([float(sample["rating"]) for sample in data_in], dtype=torch.float32),
        }

        # Backward-compatible aliases used by older project code and debug checks.
        batch["item_review"] = batch["input_i"]
        batch["user_review"] = batch["input_u"]
        batch["user_review_item_ids"] = batch["reuid"]
        batch["item_review_user_ids"] = batch["reiid"]
        batch["user_seq_reviews"] = batch["input_u"]
        batch["item_seq_reviews"] = batch["input_i"]
        batch["user_seq_len"] = batch["u_s_renum"]
        batch["item_seq_len"] = batch["i_s_renum"]
        batch["user_pos_ind"] = batch["u_pos_ind"]
        batch["item_pos_ind"] = batch["i_pos_ind"]
        batch["user_rel_dt"] = batch["u_rel_dt"]
        batch["item_rel_dt"] = batch["i_rel_dt"]
        batch["user_abs_dt"] = batch["u_abs_dt"]
        batch["item_abs_dt"] = batch["i_abs_dt"]
        batch["graph_adj"] = batch["adj"]
        batch["graph_reviews"] = batch["reviews"]
        batch["graph_ratings"] = batch["ratings"]

        debug_shapes = self._coerce_bool(self.configs.get("debug_shapes", False), False)
        if debug_shapes and not getattr(self, "_shape_logged", False):
            print("SSG dataloader shapes:", {key: tuple(value.shape) for key, value in batch.items() if key in {"nodes", "reviews", "ratings", "adj", "pairs", "input_u", "input_i", "reuid", "reiid", "u_pos_ind", "u_rel_dt", "u_abs_dt"}})
            self._shape_logged = True
        return batch


def ssg_collate_fn(batch: list[dict[str, object]]) -> dict[str, torch.Tensor]:
    if not batch:
        return {}
    dataset = getattr(batch[0], "_dataset", None)
    if dataset is None and isinstance(batch[0], dict):
        dataset = batch[0].get("_dataset")
    if dataset is not None:
        return dataset.collate_fn(batch)
    raise RuntimeError("SSG collate requires dataset-bound records; use SSGDataset.collate_fn via monkey-patched __getitem__.")
