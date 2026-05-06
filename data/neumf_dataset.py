import torch
import numpy as np
from .abstract_dataset import RecDataset


class NeuMFDataset(RecDataset):
    def __init__(self, df, configs, split="train"):
        super().__init__(df, configs, split)
        self.user_ids = torch.as_tensor(df["user_id"].to_numpy(dtype=np.int64), dtype=torch.long)
        self.item_ids = torch.as_tensor(df["item_id"].to_numpy(dtype=np.int64), dtype=torch.long)
        self.ratings = torch.as_tensor(df["rating"].to_numpy(dtype=np.float32), dtype=torch.float32)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return {
            "user_id": self.user_ids[idx].clone().detach(),
            "item_id": self.item_ids[idx].clone().detach(),
            "rating": self.ratings[idx].clone().detach(),
        }
