from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .abstract import AbstractRec


class ContrastLoss(nn.Module):
    """ReviewGraph-style bilinear BCE contrastive loss."""

    def __init__(self, feat_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(feat_size, feat_size))
        nn.init.xavier_uniform_(self.weight)
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        y_neg: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pos_scores = (x @ self.weight * y).sum(dim=1)
        pos_loss = self.loss_fn(pos_scores, torch.ones_like(pos_scores))

        if y_neg is None:
            neg_perm = torch.randperm(y.size(0), device=y.device)
            y_neg = y[neg_perm]

        neg_scores = (x @ self.weight * y_neg).sum(dim=1)
        neg_loss = self.loss_fn(neg_scores, torch.zeros_like(neg_scores))
        return pos_loss + neg_loss


class GCMCGraphConv(nn.Module):
    """Rating-specific review-aware message passing block."""

    def __init__(self, in_feats: int, out_feats: int, review_dim: int, dropout_rate: float):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_feats, out_feats))
        self.dropout = nn.Dropout(dropout_rate)
        self.prob_score = nn.Linear(review_dim, 1, bias=False)
        self.review_score = nn.Linear(review_dim, 1, bias=False)
        self.review_proj = nn.Linear(review_dim, out_feats, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.prob_score.weight)
        nn.init.xavier_uniform_(self.review_score.weight)
        nn.init.xavier_uniform_(self.review_proj.weight)

    def forward(
        self,
        src_ids: torch.Tensor,
        dst_ids: torch.Tensor,
        review_feat: torch.Tensor,
        src_cj: torch.Tensor,
        dst_ci: torch.Tensor,
        num_dst: int,
    ) -> torch.Tensor:
        if src_ids.numel() == 0:
            return self.weight.new_zeros((num_dst, self.weight.size(1)))

        src_feat = self.weight[src_ids]
        pa = torch.sigmoid(self.prob_score(review_feat))
        ra = torch.sigmoid(self.review_score(review_feat))
        rf = self.review_proj(review_feat)
        msg = (src_feat * pa + rf * ra) * self.dropout(src_cj[src_ids])

        out = self.weight.new_zeros((num_dst, self.weight.size(1)))
        out.index_add_(0, dst_ids, msg)
        return out * dst_ci


class GCMCLayer(nn.Module):
    def __init__(self, rating_vals: list[int], user_in_units: int, item_in_units: int, review_dim: int, out_units: int, dropout_rate: float):
        super().__init__()
        self.rating_vals = rating_vals
        self.user_convs = nn.ModuleDict()
        self.item_convs = nn.ModuleDict()
        for rating in rating_vals:
            key = str(rating)
            self.user_convs[key] = GCMCGraphConv(item_in_units, out_units, review_dim, dropout_rate)
            self.item_convs[key] = GCMCGraphConv(user_in_units, out_units, review_dim, dropout_rate)

        self.ufc = nn.Linear(out_units, out_units)
        self.ifc = nn.Linear(out_units, out_units)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.ufc.weight)
        nn.init.xavier_uniform_(self.ifc.weight)
        if self.ufc.bias is not None:
            nn.init.zeros_(self.ufc.bias)
        if self.ifc.bias is not None:
            nn.init.zeros_(self.ifc.bias)

    def forward(self, encoder_data: dict[str, dict[str, torch.Tensor]], norm_factors: dict[str, dict[str, torch.Tensor]]):
        num_users = norm_factors["user"]["ci"].size(0)
        num_items = norm_factors["item"]["ci"].size(0)
        user_out = self.ufc.weight.new_zeros((num_users, self.ufc.in_features))
        item_out = self.ifc.weight.new_zeros((num_items, self.ifc.in_features))

        for rating in self.rating_vals:
            key = str(rating)
            edge_data = encoder_data.get(key)
            if edge_data is None:
                continue

            edge_users = edge_data["user_ids"]
            edge_items = edge_data["item_ids"]
            review_feat = edge_data["review_feat"]

            user_out = user_out + self.user_convs[key](
                src_ids=edge_items,
                dst_ids=edge_users,
                review_feat=review_feat,
                src_cj=norm_factors["item"]["cj"],
                dst_ci=norm_factors["user"]["ci"],
                num_dst=num_users,
            )
            item_out = item_out + self.item_convs[key](
                src_ids=edge_users,
                dst_ids=edge_items,
                review_feat=review_feat,
                src_cj=norm_factors["user"]["cj"],
                dst_ci=norm_factors["item"]["ci"],
                num_dst=num_items,
            )

        user_out = self.ufc(self.dropout(self.activation(user_out)))
        item_out = self.ifc(self.dropout(self.activation(item_out)))
        return user_out, item_out


class MLPPredictorMI(nn.Module):
    def __init__(self, hidden_dim: int, review_dim: int, classification: bool, num_classes: int, dropout_rate: float):
        super().__init__()
        self.classification = classification
        self.review_proj = nn.Linear(review_dim, hidden_dim, bias=False)
        self.contrast_loss = ContrastLoss(hidden_dim)
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.predictor = nn.Linear(hidden_dim, num_classes if classification else 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def forward(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        review_feat: Optional[torch.Tensor] = None,
        cal_edge_mi: bool = True,
    ):
        interaction = self.linear(torch.cat([user_emb[user_ids], item_emb[item_ids]], dim=1))
        interaction = self.dropout(interaction)
        scores = self.predictor(interaction)

        if cal_edge_mi and review_feat is not None:
            projected_review = self.review_proj(review_feat)
            edge_mi = self.contrast_loss(interaction, projected_review)
            return scores, edge_mi

        return scores


class RGCL(AbstractRec):
    """ReviewGraph-style RGCL adapted to this repo's cached review embeddings."""

    def __init__(self, configs, train_dataset):
        super().__init__()
        self.configs = configs
        self.num_users = train_dataset.num_users
        self.num_items = train_dataset.num_items
        self.embedding_size = int(configs.get("embedding_size", 64))
        self.review_dim = int(configs.get("review_dim", 384))
        self.num_layers = int(configs.get("num_layers", 1))
        self.dropout = float(configs.get("dropout", 0.3))
        self.use_contrastive = bool(configs.get("use_contrastive", True))
        self.nd_weight = float(configs.get("nd_weight", 0.3))
        self.ed_weight = float(configs.get("ed_weight", 1.0))
        self.classification = bool(configs.get("classification", True))
        self.rating_vals = [int(v) for v in configs.get("rating_values", [1, 2, 3, 4, 5])]

        self.encoder = nn.ModuleList(
            GCMCLayer(
                rating_vals=self.rating_vals,
                user_in_units=self.num_users,
                item_in_units=self.num_items,
                review_dim=self.review_dim,
                out_units=self.embedding_size,
                dropout_rate=self.dropout,
            )
            for _ in range(self.num_layers)
        )
        self.decoder = MLPPredictorMI(
            hidden_dim=self.embedding_size,
            review_dim=self.review_dim,
            classification=self.classification,
            num_classes=len(self.rating_vals),
            dropout_rate=self.dropout,
        )
        self.nd_contrast = ContrastLoss(self.embedding_size)
        self.rating_loss = nn.CrossEntropyLoss() if self.classification else nn.MSELoss()

    def _to_device(self, value, device):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {key: self._to_device(sub_value, device) for key, sub_value in value.items()}
        return value

    def _prepare_batch(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        device = next(self.parameters()).device
        prepared = self._to_device(batch_data, device)
        if not isinstance(prepared, dict):
            raise TypeError("RGCL expects batch data as a dictionary.")
        return prepared

    def encode(self, encoder_data: dict[str, dict[str, torch.Tensor]], norm_factors: dict[str, dict[str, torch.Tensor]]):
        user_out = None
        item_out = None
        for layer in self.encoder:
            user_out, item_out = layer(encoder_data, norm_factors)
        if user_out is None or item_out is None:
            raise RuntimeError("RGCL encoder produced no outputs.")
        return user_out, item_out

    def _forward_once(self, batch: Dict[str, Any], cal_edge_mi: bool):
        user_out, item_out = self.encode(batch["encoder_data"], batch["norm_factors"])
        decode_result = self.decoder(
            user_emb=user_out,
            item_emb=item_out,
            user_ids=batch["decoder_user_ids"],
            item_ids=batch["decoder_item_ids"],
            review_feat=batch.get("decoder_review_feat"),
            cal_edge_mi=cal_edge_mi,
        )
        if cal_edge_mi:
            pred_ratings, ed_loss = decode_result
            return pred_ratings, ed_loss, user_out, item_out
        else:
            pred_ratings = decode_result
            ed_loss = pred_ratings.new_tensor(0.0)
            return pred_ratings, ed_loss, user_out, item_out

    def _rating_loss(self, pred_ratings: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        if self.classification:
            return self.rating_loss(pred_ratings, batch["labels"])
        return self.rating_loss(pred_ratings.squeeze(-1), batch["ratings"])

    def cal_loss(self, *args, **kwargs):
        batch_data = args[0] if args else kwargs["batch_data"]
        batch = self._prepare_batch(batch_data)

        if self.use_contrastive:
            pred_ratings1, ed_loss1, user1, item1 = self._forward_once(batch, cal_edge_mi=True)
            pred_ratings2, ed_loss2, user2, item2 = self._forward_once(batch, cal_edge_mi=True)
            rating_loss = (self._rating_loss(pred_ratings1, batch) + self._rating_loss(pred_ratings2, batch)) / 2.0
            nd_loss = (self.nd_contrast(user1, user2).mean() + self.nd_contrast(item1, item2).mean()) / 2.0
            ed_loss = (ed_loss1.mean() + ed_loss2.mean()) / 2.0
        else:
            pred_ratings, ed_loss, user_out, item_out = self._forward_once(batch, cal_edge_mi=False)
            rating_loss = self._rating_loss(pred_ratings, batch)
            nd_loss = rating_loss.new_tensor(0.0)
            ed_loss = rating_loss.new_tensor(0.0)

        total_loss = rating_loss + self.nd_weight * nd_loss + self.ed_weight * ed_loss
        loss_dict = {
            "total_loss": float(total_loss.detach().item()),
            "rating_loss": float(rating_loss.detach().item()),
            "nd_loss": float(nd_loss.detach().item()),
            "ed_loss": float(ed_loss.detach().item()),
        }
        return total_loss, loss_dict

    def predict_ratings(self, batch_data):
        batch = self._prepare_batch(batch_data)
        pred_ratings, ed_loss, user_out, item_out = self._forward_once(batch, cal_edge_mi=False)
        if self.classification:
            probs = F.softmax(pred_ratings, dim=1)
            rating_vals = pred_ratings.new_tensor(self.rating_vals, dtype=torch.float32)
            return (probs * rating_vals.view(1, -1)).sum(dim=1)
        return pred_ratings.squeeze(-1)

    def predict_scores(self, *args, **kwargs):
        user_ids = args[0] if args else kwargs["user_ids"]
        return torch.zeros((user_ids.size(0), self.num_items), device=user_ids.device)
