import numpy as np
import torch

from .abstract_dataset import RecDataset


class RGCLDataset(RecDataset):
    """Full-graph RGCL dataset adapted to this repo's split-based pipeline."""

    def __init__(self, df, configs, split="train"):
        super().__init__(df, configs, split)
        self.rating_vals = [1, 2, 3, 4, 5]
        self.review_dim = int(configs.get("review_dim", 384))
        self.use_review_feat = bool(configs.get("use_review", True))
        self.symm = bool(configs.get("symm", True))

        self.encoder_data = {}
        self.norm_factors = {}
        self.decoder_user_ids = None
        self.decoder_item_ids = None
        self.decoder_review_feat = None
        self.labels = None
        self.ratings = None

        self._build_decoder_tensors(self.df)
        self._build_encoder_graph(self.df)

    def _extract_review_tensor(self, frame) -> torch.Tensor:
        if not self.use_review_feat or "review_embedding" not in frame.columns:
            return torch.zeros((len(frame), self.review_dim), dtype=torch.float32)

        review_vectors = []
        for review_embedding in frame["review_embedding"].tolist():
            if isinstance(review_embedding, torch.Tensor):
                review_vectors.append(review_embedding.float().view(-1))
            elif review_embedding is None or (isinstance(review_embedding, float) and np.isnan(review_embedding)):
                review_vectors.append(torch.zeros(self.review_dim, dtype=torch.float32))
            else:
                review_vectors.append(torch.tensor(review_embedding, dtype=torch.float32).view(-1))
        return torch.stack(review_vectors, dim=0)

    def _build_decoder_tensors(self, frame) -> None:
        ratings = frame["rating"].to_numpy(dtype=np.float32)
        self.decoder_user_ids = torch.tensor(frame["user_id"].to_numpy(dtype=np.int64), dtype=torch.long)
        self.decoder_item_ids = torch.tensor(frame["item_id"].to_numpy(dtype=np.int64), dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)
        self.labels = torch.tensor(np.searchsorted(np.array(self.rating_vals, dtype=np.float32), ratings), dtype=torch.long)
        self.decoder_review_feat = self._extract_review_tensor(frame)

    def _build_encoder_graph(self, frame) -> None:
        user_ids = frame["user_id"].to_numpy(dtype=np.int64)
        item_ids = frame["item_id"].to_numpy(dtype=np.int64)
        ratings = frame["rating"].to_numpy(dtype=np.float32)
        review_feat = self._extract_review_tensor(frame)

        self.encoder_data = {}
        for rating in self.rating_vals:
            mask = ratings == float(rating)
            if not np.any(mask):
                continue
            rating_key = str(rating)
            self.encoder_data[rating_key] = {
                "user_ids": torch.tensor(user_ids[mask], dtype=torch.long),
                "item_ids": torch.tensor(item_ids[mask], dtype=torch.long),
                "review_feat": review_feat[torch.from_numpy(mask)],
            }

        self.norm_factors = self._compute_norm_factors(self.encoder_data)

    def _compute_norm_factors(self, encoder_data):
        user_in = torch.zeros(self.num_users, dtype=torch.float32)
        user_out = torch.zeros(self.num_users, dtype=torch.float32)
        item_in = torch.zeros(self.num_items, dtype=torch.float32)
        item_out = torch.zeros(self.num_items, dtype=torch.float32)

        for edge_data in encoder_data.values():
            edge_users = edge_data["user_ids"]
            edge_items = edge_data["item_ids"]
            user_in.index_add_(0, edge_users, torch.ones_like(edge_users, dtype=torch.float32))
            item_in.index_add_(0, edge_items, torch.ones_like(edge_items, dtype=torch.float32))
            if self.symm:
                user_out.index_add_(0, edge_users, torch.ones_like(edge_users, dtype=torch.float32))
                item_out.index_add_(0, edge_items, torch.ones_like(edge_items, dtype=torch.float32))

        def normalize(degrees: torch.Tensor) -> torch.Tensor:
            degrees = degrees.clone()
            degrees[degrees == 0] = float("inf")
            return (1.0 / torch.sqrt(degrees)).unsqueeze(1)

        return {
            "user": {
                "ci": normalize(user_in),
                "cj": normalize(user_out) if self.symm else torch.ones(self.num_users, 1, dtype=torch.float32),
            },
            "item": {
                "ci": normalize(item_in),
                "cj": normalize(item_out) if self.symm else torch.ones(self.num_items, 1, dtype=torch.float32),
            },
        }

    def _setup_evaluation(self, train_df, valid_df, test_df):
        if self.split == "valid":
            self._build_encoder_graph(train_df)
        elif self.split == "test":
            self._build_encoder_graph(train_df)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {
            "encoder_data": self.encoder_data,
            "norm_factors": self.norm_factors,
            "decoder_user_ids": self.decoder_user_ids,
            "decoder_item_ids": self.decoder_item_ids,
            "decoder_review_feat": self.decoder_review_feat,
            "labels": self.labels,
            "ratings": self.ratings,
        }


def rgcl_collate_fn(batch):
    return batch[0]
