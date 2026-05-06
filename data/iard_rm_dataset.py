import os
import numpy as np
import torch

from .abstract_dataset import RecDataset


class IARDRMDataset(RecDataset):
    def __init__(self, df, configs, split="train"):
        super().__init__(df, configs, split)
        self.user_ids = torch.as_tensor(df["user_id"].to_numpy(dtype=np.int64), dtype=torch.long)
        self.item_ids = torch.as_tensor(df["item_id"].to_numpy(dtype=np.int64), dtype=torch.long)
        self.ratings = torch.as_tensor(df["rating"].to_numpy(dtype=np.float32), dtype=torch.float32)
        self.review_emb = self._load_review_embeddings(df, configs)
        self.edge_index = self._build_edge_index(df)
        self.edge_weight = None

    def _load_review_embeddings(self, frame, configs):
        if "review_embedding" in frame.columns:
            embedding_values = frame["review_embedding"].tolist()
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
                    "Set d_text in config or provide at least one valid review embedding."
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
            if configs.get("d_text") is None:
                configs["d_text"] = target_dim
            elif int(configs.get("d_text")) != target_dim:
                raise ValueError(
                    f"Configured d_text={configs.get('d_text')} does not match review embedding dim {target_dim}."
                )
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
        if configs.get("d_text") is None:
            configs["d_text"] = inferred_dim
        elif int(configs.get("d_text")) != inferred_dim:
            raise ValueError(
                f"Configured d_text={configs.get('d_text')} does not match review embedding dim {inferred_dim}."
            )
        return tensor

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

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return {
            "user_ids": self.user_ids[idx].clone().detach(),
            "item_ids": self.item_ids[idx].clone().detach(),
            "ratings": self.ratings[idx].clone().detach(),
            "review_emb": self.review_emb[idx].clone().detach(),
        }
