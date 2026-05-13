from collections.abc import Mapping
from typing import Dict, Tuple, Union, cast, final

import torch
import torch.nn as nn
import torch.nn.functional as F

from .abstract import AbstractRec


@final
class DAML(AbstractRec):
    """PyTorch reproduction of DAML (Dual Attention Mutual Learning).

    Paper: Liu et al., "DAML: Dual Attention Mutual Learning between Ratings
    and Reviews for Item Recommendation", KDD 2019.

    Architecture mapping to the original Neu-Review-Rec implementation:
    - ``local_attention_cnn``       -> word-level attention (Eq.1-7)
    - ``dual attention matrix``     -> mutual attention via Euclidean distance
    - ``local_pooling_cnn``         -> abstract-level CNN + attention pooling (Eq.11-13)
    - ``FusionLayer`` (external)    -> reshape + concat (integrated below)
    - ``PredictionLayer(LFM)``      -> FC + user/item bias (integrated below)

    Unlike the external framework where the model returns feature tensors and
    a separate framework handles fusion + prediction, this implementation
    returns rating predictions directly to match the current project's
    ``AbstractRec`` interface.
    """

    def __init__(self, configs: Mapping[str, object], train_dataset: object) -> None:
        super().__init__()
        self.configs = configs
        self.train_dataset = train_dataset

        self.doc_len = self._get_int_config("doc_len", 500)
        self.word_dim = self._get_int_config("word_dim", 300)
        self.filters_num = self._get_int_config("filters_num", 100)
        self.kernel_size = self._get_int_config("kernel_size", 3)
        self.id_emb_size = self._get_int_config("id_emb_size", 32)
        self.dropout_prob = self._get_float_config("dropout_prob", 0.5)
        self.l2_reg_lambda = self._get_float_config("l2_reg_lambda", 0.0)
        self.freeze_word_embedding = self._get_bool_config("freeze_word_embedding", False)

        self.num_users = int(getattr(train_dataset, "num_users"))
        self.num_items = int(getattr(train_dataset, "num_items"))
        self.pad_idx = int(getattr(train_dataset, "pad_idx", 0))

        embedding_weight = torch.as_tensor(
            getattr(train_dataset, "embedding_matrix"),
            dtype=torch.float32,
        )
        if embedding_weight.size(1) != self.word_dim:
            raise ValueError(
                f"Configured word_dim={self.word_dim} does not match embedding dim={embedding_weight.size(1)}"
            )

        # Word embeddings (shared vocab, separate tables like original)
        self.user_word_embs = nn.Embedding.from_pretrained(
            embedding_weight,
            freeze=self.freeze_word_embedding,
            padding_idx=self.pad_idx,
        )
        self.item_word_embs = nn.Embedding.from_pretrained(
            embedding_weight,
            freeze=self.freeze_word_embedding,
            padding_idx=self.pad_idx,
        )

        # Local attention word CNN (shared, 1x1 conv over word_dim)
        # Original: kernel=(5, word_dim), padding=(2, 0)
        self.word_cnn = nn.Conv2d(1, 1, (5, self.word_dim), padding=(2, 0))

        # Document-level CNN (extract local features from weighted embeddings)
        # Original: Conv2d(1, filters_num, (kernel_size, word_dim), padding=(1, 0))
        self.user_doc_cnn = nn.Conv2d(1, self.filters_num, (self.kernel_size, self.word_dim), padding=(1, 0))
        self.item_doc_cnn = nn.Conv2d(1, self.filters_num, (self.kernel_size, self.word_dim), padding=(1, 0))

        # Abstract-level CNN (for local pooling)
        # Original: Conv2d(1, filters_num, (kernel_size, filters_num))
        self.user_abs_cnn = nn.Conv2d(1, self.filters_num, (self.kernel_size, self.filters_num))
        self.item_abs_cnn = nn.Conv2d(1, self.filters_num, (self.kernel_size, self.filters_num))

        # Unfold for local pooling (window=3 over filters_num)
        self.unfold = nn.Unfold((3, self.filters_num), padding=(1, 0))

        # FC layers to project doc features to id_emb_size
        self.user_fc = nn.Linear(self.filters_num, self.id_emb_size)
        self.item_fc = nn.Linear(self.filters_num, self.id_emb_size)

        # ID embeddings
        self.uid_embedding = nn.Embedding(self.num_users + 2, self.id_emb_size)
        self.iid_embedding = nn.Embedding(self.num_items + 2, self.id_emb_size)

        # --- Integrated Fusion + Prediction layers ---
        # Original framework: FusionLayer(r_id_merge='cat', ui_merge='cat')
        #   user_feature (B, 2, id_emb_size) -> reshape -> (B, 2*id_emb_size)
        #   item_feature (B, 2, id_emb_size) -> reshape -> (B, 2*id_emb_size)
        #   concat -> (B, 4*id_emb_size)
        # Original framework: PredictionLayer('lfm')
        #   fc: Linear(4*id_emb_size, 1) + user_bias + item_bias
        self.fusion_fc = nn.Linear(self.id_emb_size * 4, 1)
        self.user_bias = nn.Parameter(torch.randn(self.num_users, 1) * 0.01)
        self.item_bias = nn.Parameter(torch.randn(self.num_items, 1) * 0.01)

        self.dropout = nn.Dropout(self.dropout_prob)

        self.reset_para()

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"DAML parameters: total={total_params}, trainable={trainable_params}"
        )

    def _get_int_config(self, key: str, default: int) -> int:
        return int(cast(Union[int, float, str], self.configs.get(key, default)))

    def _get_float_config(self, key: str, default: float) -> float:
        return float(cast(Union[int, float, str], self.configs.get(key, default)))

    def _get_bool_config(self, key: str, default: bool) -> bool:
        value = self.configs.get(key, default)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def reset_para(self) -> None:
        cnns = [
            self.word_cnn,
            self.user_doc_cnn,
            self.item_doc_cnn,
            self.user_abs_cnn,
            self.item_abs_cnn,
        ]
        for cnn in cnns:
            nn.init.xavier_normal_(cnn.weight)
            nn.init.uniform_(cnn.bias, -0.1, 0.1)

        fcs = [self.user_fc, self.item_fc, self.fusion_fc]
        for fc in fcs:
            nn.init.uniform_(fc.weight, -0.1, 0.1)
            nn.init.constant_(fc.bias, 0.1)

        nn.init.uniform_(self.uid_embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.iid_embedding.weight, -0.1, 0.1)

    def local_attention_cnn(self, word_embs: torch.Tensor, doc_cnn: nn.Conv2d) -> torch.Tensor:
        """Local attention CNN (Eq.1-7 in paper).

        Args:
            word_embs: (B, DOC_LEN, word_dim)
            doc_cnn: document-level CNN module

        Returns:
            Local features: (B, filters_num, DOC_LEN, 1)
        """
        # word_cnn input: (B, 1, DOC_LEN, word_dim)
        local_att_words = self.word_cnn(word_embs.unsqueeze(1))
        # (B, 1, DOC_LEN, 1)
        local_word_weight = torch.sigmoid(local_att_words.squeeze(1))
        # (B, DOC_LEN, 1) -- broadcast over word_dim
        word_embs = word_embs * local_word_weight
        # doc_cnn input: (B, 1, DOC_LEN, word_dim)
        d_fea = doc_cnn(word_embs.unsqueeze(1))
        # (B, filters_num, DOC_LEN, 1)
        return d_fea

    def local_pooling_cnn(
        self,
        feature: torch.Tensor,
        attention: torch.Tensor,
        cnn: nn.Conv2d,
        fc: nn.Linear,
    ) -> torch.Tensor:
        """Local pooling CNN with attention (Eq.11-13 in paper).

        Args:
            feature: (B, filters_num, DOC_LEN, 1)
            attention: (B, DOC_LEN)
            cnn: abstract-level CNN module
            fc: final FC projection

        Returns:
            Doc feature: (B, id_emb_size)
        """
        bs, n_filters, doc_len, _ = feature.shape
        # (B, 1, DOC_LEN, filters_num)
        feature = feature.permute(0, 3, 2, 1)
        # (B, 1, DOC_LEN, 1)
        attention = attention.reshape(bs, 1, doc_len, 1)
        pools = feature * attention
        # unfold: (B, 3*filters_num, DOC_LEN)
        pools = self.unfold(pools)
        # (B, 3, filters_num, DOC_LEN)
        pools = pools.reshape(bs, 3, n_filters, doc_len)
        # (B, 1, filters_num, DOC_LEN)
        pools = pools.sum(dim=1, keepdim=True)
        # (B, 1, DOC_LEN, filters_num)
        pools = pools.transpose(2, 3)

        # abs_cnn: (B, filters_num, DOC_LEN-2)
        abs_fea = cnn(pools).squeeze(3)
        # (B, filters_num)
        abs_fea = F.avg_pool1d(abs_fea, abs_fea.size(2))
        # (B, id_emb_size)
        abs_fea = F.relu(fc(abs_fea.squeeze(2)))
        return abs_fea

    def forward(
        self,
        user_id: torch.Tensor,
        item_id: torch.Tensor,
        user_doc: torch.Tensor,
        item_doc: torch.Tensor,
    ) -> torch.Tensor:
        # Review encoder
        user_word_embs = self.user_word_embs(user_doc)
        item_word_embs = self.item_word_embs(item_doc)

        user_local_fea = self.local_attention_cnn(user_word_embs, self.user_doc_cnn)
        item_local_fea = self.local_attention_cnn(item_word_embs, self.item_doc_cnn)

        # Dual attention via Euclidean distance
        # user_local_fea: (B, filters_num, DOC_LEN, 1)
        # item_local_fea.permute(0, 1, 3, 2): (B, filters_num, 1, DOC_LEN)
        euclidean = (user_local_fea - item_local_fea.permute(0, 1, 3, 2)).pow(2).sum(1).sqrt()
        # (B, DOC_LEN, DOC_LEN)
        attention_matrix = 1.0 / (1.0 + euclidean)
        user_attention = attention_matrix.sum(2)  # (B, DOC_LEN)
        item_attention = attention_matrix.sum(1)  # (B, DOC_LEN)

        user_doc_fea = self.local_pooling_cnn(user_local_fea, user_attention, self.user_abs_cnn, self.user_fc)
        item_doc_fea = self.local_pooling_cnn(item_local_fea, item_attention, self.item_abs_cnn, self.item_fc)

        # ID embeddings
        uid_emb = self.uid_embedding(user_id)
        iid_emb = self.iid_embedding(item_id)

        # Feature stacking: (B, 2, id_emb_size)
        user_feature = torch.stack([user_doc_fea, uid_emb], 1)
        item_feature = torch.stack([item_doc_fea, iid_emb], 1)

        # Fusion (equivalent to FusionLayer with r_id_merge='cat', ui_merge='cat')
        user_feature = user_feature.reshape(user_feature.size(0), -1)  # (B, 2*id_emb_size)
        item_feature = item_feature.reshape(item_feature.size(0), -1)  # (B, 2*id_emb_size)
        ui_feature = torch.cat([user_feature, item_feature], dim=1)    # (B, 4*id_emb_size)
        ui_feature = self.dropout(ui_feature)

        # Prediction (equivalent to LFM prediction layer)
        rating = self.fusion_fc(ui_feature)
        rating = rating + self.user_bias[user_id] + self.item_bias[item_id]
        return rating

    def cal_loss(
        self,
        batch_data: Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        user_id, item_id, user_doc, item_doc, ratings = batch_data

        predictions = self.forward(user_id, item_id, user_doc, item_doc)
        residual = predictions - ratings.view(-1, 1).float()
        rating_loss = 0.5 * torch.sum(residual ** 2)

        l2_loss = torch.tensor(0.0, device=rating_loss.device)
        if self.l2_reg_lambda > 0:
            l2_loss = (
                0.5 * torch.sum(self.fusion_fc.weight ** 2)
                + 0.5 * torch.sum(self.user_fc.weight ** 2)
                + 0.5 * torch.sum(self.item_fc.weight ** 2)
            )

        total_loss = rating_loss + self.l2_reg_lambda * l2_loss

        loss_value = float(total_loss.detach().item())
        rating_loss_value = float(rating_loss.detach().item())
        l2_value = float(l2_loss.detach().item())
        return total_loss, {
            "rating_l2_loss": rating_loss_value,
            "attention_l2_loss": l2_value,
            "total_loss": loss_value,
        }

    def predict_scores(
        self,
        user_id: torch.Tensor,
        item_id: torch.Tensor,
        user_doc: torch.Tensor,
        item_doc: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(user_id, item_id, user_doc, item_doc)
