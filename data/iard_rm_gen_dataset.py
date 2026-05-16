import numpy as np
import torch

from .abstract_dataset import RecDataset
from .iard_rm_dataset import IARDRMDataset


class IARDRMGenDataset(IARDRMDataset):
    """
    Interaction-level dataset for IARD-RM-Gen.

    Training rows expose the target interaction review embedding so the model can
    learn a decoder from the interaction/rating representation to review space.
    Validation and test rows expose only zero placeholders with the same shape;
    the model must rely on its generated review embedding for prediction.
    """

    def __init__(self, df, configs, split="train"):
        RecDataset.__init__(self, df, configs, split)
        self.configs = configs
        self.review_context_mode = "target"
        self.history_temporal = bool(configs.get("history_temporal", False))
        self.history_aggregation = str(configs.get("history_aggregation", "mean")).lower()
        self.history_encoder = "mean"
        self.history_top_k = int(configs.get("history_top_k", 10))
        if self.history_top_k <= 0:
            raise ValueError("history_top_k must be positive.")
        self.retain_rui = False
        self.has_timestamp = "timestamp" in df.columns
        self.user_ids = torch.as_tensor(df["user_id"].to_numpy(dtype=np.int64), dtype=torch.long)
        self.item_ids = torch.as_tensor(df["item_id"].to_numpy(dtype=np.int64), dtype=torch.long)
        self.ratings = torch.as_tensor(df["rating"].to_numpy(dtype=np.float32), dtype=torch.float32)

        if split == "train":
            raw_review_emb = self._load_review_embeddings(df, configs)
            self.review_emb = raw_review_emb
            self.review_source_used = torch.zeros(len(df), dtype=torch.long)
            print(
                "IARD-RM-Gen train context: source=target_interaction_review_for_decoder_supervision, "
                "target_review_used_for_prediction=False"
            )
        else:
            if configs.get("review_emb_dim") is None or configs.get("d_text") is None:
                raise ValueError(
                    "IARD-RM-Gen valid/test datasets require train dataset initialization first "
                    "so review_emb_dim and d_text are known."
                )
            self._configure_review_dimensions(int(configs["review_emb_dim"]), configs)
            self.review_emb = torch.zeros((len(df), int(configs["d_text"])), dtype=torch.float32)
            self.review_source_used = torch.ones(len(df), dtype=torch.long)

        self.interactions = self._build_interactions(df, self.review_emb if split == "train" else None)
        self.empty_history_mask = torch.ones(len(df), dtype=torch.bool)
        self.user_history_emb, self.user_history_mask = self._empty_history_sequences(len(df))
        self.item_history_emb, self.item_history_mask = self._empty_history_sequences(len(df))
        self.history_context_ready = True
        self.edge_index = self._build_edge_index(df)
        self.edge_weight = None

    def _setup_evaluation(self, train_df, valid_df, test_df):
        del valid_df, test_df
        self.edge_index = self._build_edge_index(train_df)
        self.edge_weight = None
        self.history_context_ready = True
        print(
            f"IARD-RM-Gen {self.split} context: source=generated_review_only, "
            "target_review_used=False"
        )
