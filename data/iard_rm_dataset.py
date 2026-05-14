import os
import numpy as np
import torch

from .abstract_dataset import RecDataset


class IARDRMDataset(RecDataset):
    def __init__(self, df, configs, split="train"):
        super().__init__(df, configs, split)
        self.configs = configs
        self.review_context_mode = str(configs.get("review_context_mode", "history")).lower()
        if self.review_context_mode not in {"target", "history"}:
            raise ValueError("review_context_mode must be either 'target' or 'history'.")
        if split in {"valid", "test"} and self.review_context_mode == "target":
            raise ValueError("IARD-RM valid/test cannot use target review embeddings; use review_context_mode='history'.")
        self.history_temporal = bool(configs.get("history_temporal", False))
        self.history_aggregation = str(configs.get("history_aggregation", "mean")).lower()
        if self.history_aggregation != "mean":
            raise ValueError("IARD-RM history mode currently supports only mean aggregation.")
        self.history_encoder = str(configs.get("history_encoder", "mean")).lower()
        if self.history_encoder not in {"mean", "attention"}:
            raise ValueError("history_encoder must be either 'mean' or 'attention'.")
        self.history_top_k = int(configs.get("history_top_k", 10))
        if self.history_top_k <= 0:
            raise ValueError("history_top_k must be positive.")
        self.has_timestamp = "timestamp" in df.columns
        self.user_ids = torch.as_tensor(df["user_id"].to_numpy(dtype=np.int64), dtype=torch.long)
        self.item_ids = torch.as_tensor(df["item_id"].to_numpy(dtype=np.int64), dtype=torch.long)
        self.ratings = torch.as_tensor(df["rating"].to_numpy(dtype=np.float32), dtype=torch.float32)
        skip_target_review_load = self.review_context_mode == "history" and split in {"valid", "test"}
        raw_review_emb = None if skip_target_review_load else self._load_review_embeddings(df, configs)
        if skip_target_review_load:
            if configs.get("review_emb_dim") is None or configs.get("d_text") is None:
                raise ValueError(
                    "IARD-RM history-mode valid/test datasets require train dataset initialization first "
                    "so review_emb_dim and d_text are known."
                )
            self._configure_review_dimensions(int(configs["review_emb_dim"]), configs)
        self.interactions = self._build_interactions(df, raw_review_emb)
        if self.review_context_mode == "history":
            if split == "train":
                (
                    self.review_emb,
                    self.empty_history_mask,
                    self.user_history_emb,
                    self.user_history_mask,
                    self.item_history_emb,
                    self.item_history_mask,
                ) = self._build_history_review_embeddings(
                    target_interactions=self.interactions,
                    history_interactions=self.interactions,
                    exclude_target=True,
                )
                self.history_context_ready = True
            else:
                self.review_emb = torch.zeros((len(self.interactions), int(configs["d_text"])), dtype=torch.float32)
                self.empty_history_mask = torch.ones(len(self.interactions), dtype=torch.bool)
                self.user_history_emb, self.user_history_mask = self._empty_history_sequences(len(self.interactions))
                self.item_history_emb, self.item_history_mask = self._empty_history_sequences(len(self.interactions))
                self.history_context_ready = False
            self.review_source_used = torch.ones(len(self.interactions), dtype=torch.long)
        else:
            if raw_review_emb is None:
                raise RuntimeError("Target review embeddings must be loaded in target review context mode.")
            self.review_emb = raw_review_emb
            self.empty_history_mask = torch.zeros(len(self.interactions), dtype=torch.bool)
            self.user_history_emb, self.user_history_mask = self._empty_history_sequences(len(self.interactions))
            self.item_history_emb, self.item_history_mask = self._empty_history_sequences(len(self.interactions))
            self.review_source_used = torch.zeros(len(self.interactions), dtype=torch.long)
            self.history_context_ready = True
        self.edge_index = self._build_edge_index(df)
        self.edge_weight = None

    def _configure_review_dimensions(self, raw_dim, configs):
        configured_raw_dim = configs.get("review_emb_dim")
        if configured_raw_dim is not None and int(configured_raw_dim) != int(raw_dim):
            raise ValueError(
                f"Configured review_emb_dim={configured_raw_dim} does not match review embedding dim {raw_dim}."
            )
        configs["review_emb_dim"] = int(raw_dim)

        model_dim = int(raw_dim) * 4 if self.review_context_mode == "history" else int(raw_dim)
        configured_model_dim = configs.get("d_text")
        if configured_model_dim is None:
            configs["d_text"] = model_dim
        elif int(configured_model_dim) == model_dim:
            pass
        elif self.review_context_mode == "history" and int(configured_model_dim) == int(raw_dim):
            configs["d_text"] = model_dim
        else:
            raise ValueError(
                f"Configured d_text={configured_model_dim} does not match IARD-RM {self.review_context_mode} "
                f"review input dim {model_dim} from raw review embedding dim {raw_dim}."
            )

    def _load_review_embeddings(self, frame, configs):
        if "review_embedding" in frame.columns:
            embedding_values = frame["review_embedding"].tolist()
            configured_dim = configs.get("review_emb_dim")
            if configured_dim is None and self.review_context_mode == "target":
                configured_dim = configs.get("d_text")
            target_dim = int(configured_dim) if configured_dim is not None else None

            if target_dim is None:
                for review_embedding in embedding_values:
                    if isinstance(review_embedding, torch.Tensor):
                        target_dim = int(review_embedding.numel())
                        break
                    if review_embedding is None or (isinstance(review_embedding, float) and np.isnan(review_embedding)):
                        continue
                    target_dim = int(torch.tensor(review_embedding, dtype=torch.float32).numel())
                    break

            if target_dim is None:
                raise ValueError(
                    "IARD-RM could not infer review embedding dimension because all review embeddings are missing. "
                    "Set review_emb_dim in history mode, set d_text in target mode, or provide at least one valid review embedding."
                )

            review_vectors = []
            for review_embedding in embedding_values:
                if isinstance(review_embedding, torch.Tensor):
                    review_vector = review_embedding.float().view(-1)
                elif review_embedding is None or (isinstance(review_embedding, float) and np.isnan(review_embedding)):
                    review_vector = torch.zeros(target_dim, dtype=torch.float32)
                else:
                    review_vector = torch.tensor(review_embedding, dtype=torch.float32).view(-1)

                if review_vector.numel() != target_dim:
                    raise ValueError(
                        f"Review embedding dim {int(review_vector.numel())} does not match expected dim {target_dim}."
                    )
                review_vectors.append(review_vector)

            stacked = torch.stack(review_vectors, dim=0)
            self._configure_review_dimensions(target_dim, configs)
            return stacked

        review_emb_path = configs.get("review_emb_path")
        if not review_emb_path:
            raise ValueError("IARD-RM requires review_embedding column or review_emb_path.")
        if not os.path.exists(review_emb_path):
            raise FileNotFoundError(f"Review embedding file not found: {review_emb_path}")

        suffix = os.path.splitext(review_emb_path)[1].lower()
        if suffix == ".pt":
            try:
                loaded = torch.load(review_emb_path, map_location="cpu", weights_only=True)
            except TypeError:
                loaded = torch.load(review_emb_path, map_location="cpu")
        elif suffix == ".npy":
            loaded = np.load(review_emb_path, allow_pickle=False)
        elif suffix == ".pkl":
            raise ValueError(
                "Unsafe review embedding format '.pkl' is not supported. "
                "Use a trusted .pt tensor file or numeric .npy array instead."
            )
        else:
            raise ValueError(f"Unsupported review embedding format: {review_emb_path}")

        if len(loaded) != len(frame):
            raise ValueError(f"Review embedding length {len(loaded)} != dataframe length {len(frame)}")

        tensor = torch.as_tensor(np.asarray(loaded), dtype=torch.float32)
        if tensor.ndim != 2:
            raise ValueError(f"Review embeddings must be 2D, got shape {tuple(tensor.shape)}")
        inferred_dim = int(tensor.size(1))
        self._configure_review_dimensions(inferred_dim, configs)
        return tensor

    @staticmethod
    def _timestamp_value(row):
        if not hasattr(row, "timestamp"):
            return None
        value = getattr(row, "timestamp")
        if value is None:
            return None
        if isinstance(value, float) and np.isnan(value):
            return None
        return value

    def _build_interactions(self, frame, review_embeddings):
        interactions = []
        for position, row in enumerate(frame.itertuples(index=False)):
            review_emb = None
            if review_embeddings is not None:
                review_emb = review_embeddings[position].float().view(-1)
            interactions.append(
                {
                    "user_id": int(getattr(row, "user_id")),
                    "item_id": int(getattr(row, "item_id")),
                    "timestamp": self._timestamp_value(row),
                    "review_emb": review_emb,
                    "position": int(position),
                }
            )
        return interactions

    def _empty_history_stats(self):
        raw_dim = int(self.configs["review_emb_dim"])
        return {
            "sum": torch.zeros(raw_dim, dtype=torch.float32),
            "count": 0,
            "entries": [],
            "timestamps": [],
            "prefix_sums": torch.empty((0, raw_dim), dtype=torch.float32),
        }

    def _build_history_stats(self, interactions):
        by_user = {}
        by_item = {}
        for interaction in interactions:
            review_emb = interaction.get("review_emb")
            if review_emb is None:
                continue
            user_stats = by_user.setdefault(interaction["user_id"], self._empty_history_stats())
            item_stats = by_item.setdefault(interaction["item_id"], self._empty_history_stats())
            for stats in (user_stats, item_stats):
                stats["sum"] = stats["sum"] + review_emb
                stats["count"] += 1
                stats["entries"].append(interaction)

        if self.history_temporal:
            for stats_lookup in (by_user, by_item):
                for stats in stats_lookup.values():
                    sorted_entries = sorted(
                        [entry for entry in stats["entries"] if entry["timestamp"] is not None],
                        key=lambda entry: entry["timestamp"],
                    )
                    stats["entries"] = sorted_entries
                    stats["timestamps"] = [entry["timestamp"] for entry in sorted_entries]
                    if sorted_entries:
                        cumulative = torch.stack([entry["review_emb"] for entry in sorted_entries], dim=0).cumsum(dim=0)
                        stats["prefix_sums"] = cumulative
        return by_user, by_item

    def _empty_history_sequences(self, num_rows):
        raw_dim = int(self.configs["review_emb_dim"])
        return (
            torch.zeros((num_rows, self.history_top_k, raw_dim), dtype=torch.float32),
            torch.zeros((num_rows, self.history_top_k), dtype=torch.bool),
        )

    def _select_history_entries(self, stats, target, exclude_target):
        if stats is None:
            return []

        entries = list(stats["entries"])
        if exclude_target:
            entries = [entry for entry in entries if entry is not target]

        if self.history_temporal and self.has_timestamp:
            target_timestamp = target["timestamp"]
            if target_timestamp is None:
                return []
            entries = [entry for entry in entries if entry["timestamp"] is not None and entry["timestamp"] < target_timestamp]

        entries = [entry for entry in entries if entry.get("review_emb") is not None]
        if self.history_temporal and self.has_timestamp:
            entries = sorted(entries, key=lambda entry: entry["timestamp"])
        return entries[-self.history_top_k:]

    def _build_history_sequence(self, entries):
        raw_dim = int(self.configs["review_emb_dim"])
        history_emb = torch.zeros((self.history_top_k, raw_dim), dtype=torch.float32)
        history_mask = torch.zeros(self.history_top_k, dtype=torch.bool)
        if not entries:
            return history_emb, history_mask

        start = self.history_top_k - len(entries)
        for offset, entry in enumerate(entries):
            history_emb[start + offset] = entry["review_emb"]
            history_mask[start + offset] = True
        return history_emb, history_mask

    def _aggregate_history(self, stats, target, exclude_target):
        if stats is None:
            raw_dim = int(self.configs["review_emb_dim"])
            return torch.zeros(raw_dim, dtype=torch.float32), True

        if self.history_temporal and self.has_timestamp:
            target_timestamp = target["timestamp"]
            if target_timestamp is None:
                raw_dim = int(self.configs["review_emb_dim"])
                return torch.zeros(raw_dim, dtype=torch.float32), True
            count = 0
            for timestamp in stats["timestamps"]:
                if timestamp >= target_timestamp:
                    break
                count += 1
            if count == 0:
                raw_dim = int(self.configs["review_emb_dim"])
                return torch.zeros(raw_dim, dtype=torch.float32), True
            return stats["prefix_sums"][count - 1] / float(count), False

        history_sum = stats["sum"].clone()
        history_count = int(stats["count"])
        if exclude_target and target.get("review_emb") is not None:
            history_sum = history_sum - target["review_emb"]
            history_count -= 1
        if history_count <= 0:
            raw_dim = int(self.configs["review_emb_dim"])
            return torch.zeros(raw_dim, dtype=torch.float32), True
        return history_sum / float(history_count), False

    @staticmethod
    def _fuse_history_embeddings(user_hist_emb, item_hist_emb):
        return torch.cat(
            [
                user_hist_emb,
                item_hist_emb,
                user_hist_emb * item_hist_emb,
                torch.abs(user_hist_emb - item_hist_emb),
            ],
            dim=0,
        )

    def _build_history_review_embeddings(self, target_interactions, history_interactions, exclude_target):
        by_user, by_item = self._build_history_stats(history_interactions)
        fused_embeddings = []
        empty_masks = []
        user_history_embeddings = []
        user_history_masks = []
        item_history_embeddings = []
        item_history_masks = []
        for target in target_interactions:
            user_hist_emb, user_empty = self._aggregate_history(
                by_user.get(target["user_id"]),
                target=target,
                exclude_target=exclude_target,
            )
            item_hist_emb, item_empty = self._aggregate_history(
                by_item.get(target["item_id"]),
                target=target,
                exclude_target=exclude_target,
            )
            fused_embeddings.append(self._fuse_history_embeddings(user_hist_emb, item_hist_emb))
            empty_masks.append(bool(user_empty or item_empty))
            user_entries = self._select_history_entries(
                by_user.get(target["user_id"]),
                target=target,
                exclude_target=exclude_target,
            )
            item_entries = self._select_history_entries(
                by_item.get(target["item_id"]),
                target=target,
                exclude_target=exclude_target,
            )
            user_history_emb, user_history_mask = self._build_history_sequence(user_entries)
            item_history_emb, item_history_mask = self._build_history_sequence(item_entries)
            user_history_embeddings.append(user_history_emb)
            user_history_masks.append(user_history_mask)
            item_history_embeddings.append(item_history_emb)
            item_history_masks.append(item_history_mask)
        if not fused_embeddings:
            raw_dim = int(self.configs["review_emb_dim"])
            return (
                torch.empty((0, raw_dim * 4), dtype=torch.float32),
                torch.empty((0,), dtype=torch.bool),
                torch.empty((0, self.history_top_k, raw_dim), dtype=torch.float32),
                torch.empty((0, self.history_top_k), dtype=torch.bool),
                torch.empty((0, self.history_top_k, raw_dim), dtype=torch.float32),
                torch.empty((0, self.history_top_k), dtype=torch.bool),
            )
        return (
            torch.stack(fused_embeddings, dim=0),
            torch.tensor(empty_masks, dtype=torch.bool),
            torch.stack(user_history_embeddings, dim=0),
            torch.stack(user_history_masks, dim=0),
            torch.stack(item_history_embeddings, dim=0),
            torch.stack(item_history_masks, dim=0),
        )

    def _build_edge_index(self, frame):
        user_array = frame["user_id"].to_numpy(dtype=np.int64)
        item_array = frame["item_id"].to_numpy(dtype=np.int64)
        offset_items = item_array + self.num_users

        src = np.empty(user_array.size * 2, dtype=np.int64)
        dst = np.empty(user_array.size * 2, dtype=np.int64)
        src[0::2] = user_array
        dst[0::2] = offset_items
        src[1::2] = offset_items
        dst[1::2] = user_array
        return torch.as_tensor(np.stack([src, dst], axis=0), dtype=torch.long)

    def _setup_evaluation(self, train_df, valid_df, test_df):
        del valid_df, test_df
        self.edge_index = self._build_edge_index(train_df)
        self.edge_weight = None
        if self.review_context_mode == "history":
            train_review_emb = self._load_review_embeddings(train_df, self.configs)
            train_interactions = self._build_interactions(train_df, train_review_emb)
            (
                self.review_emb,
                self.empty_history_mask,
                self.user_history_emb,
                self.user_history_mask,
                self.item_history_emb,
                self.item_history_mask,
            ) = self._build_history_review_embeddings(
                target_interactions=self.interactions,
                history_interactions=train_interactions,
                exclude_target=False,
            )
            self.review_source_used = torch.ones(len(self.interactions), dtype=torch.long)
            self.history_context_ready = True
            print(
                f"IARD-RM {self.split} history context: source=train_only, "
                f"encoder={self.history_encoder}, top_k={self.history_top_k}, "
                f"target_review_used=False, empty_ratio={float(self.empty_history_mask.float().mean().item()) if len(self.empty_history_mask) else 0.0:.4f}"
            )
        else:
            raise RuntimeError("IARD-RM evaluation setup requires history mode to avoid target review leakage.")

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        if not self.history_context_ready:
            raise RuntimeError("IARD-RM history context for valid/test must be initialized via _setup_evaluation().")
        return {
            "user_ids": self.user_ids[idx].clone().detach(),
            "item_ids": self.item_ids[idx].clone().detach(),
            "ratings": self.ratings[idx].clone().detach(),
            "review_emb": self.review_emb[idx].clone().detach(),
            "user_history_emb": self.user_history_emb[idx].clone().detach(),
            "user_history_mask": self.user_history_mask[idx].clone().detach(),
            "item_history_emb": self.item_history_emb[idx].clone().detach(),
            "item_history_mask": self.item_history_mask[idx].clone().detach(),
            "review_source_used": self.review_source_used[idx].clone().detach(),
            "empty_history_mask": self.empty_history_mask[idx].clone().detach(),
        }
