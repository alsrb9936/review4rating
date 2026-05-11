from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .abstract import AbstractRec


class FactorDistributionEncoder(nn.Module):
    """Compute factor probabilities from semantic (review) + structural (user/item) signals.

    For each edge, produces a K-way distribution over latent factors.
    """

    def __init__(self, review_dim: int, embedding_size: int, num_factors: int, dropout: float):
        super().__init__()
        self.num_factors = num_factors

        self.semantic_mlp = nn.Sequential(
            nn.Linear(review_dim, embedding_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size, num_factors),
        )
        self.structural_mlp = nn.Sequential(
            nn.Linear(embedding_size * 2, embedding_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size, num_factors),
        )
        self.gate = nn.Sequential(
            nn.Linear(review_dim + embedding_size * 2, embedding_size),
            nn.GELU(),
            nn.Linear(embedding_size, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        review_feat: torch.Tensor,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return (num_edges, num_factors) softmax distribution."""
        u_feat = user_emb[user_ids]
        i_feat = item_emb[item_ids]

        semantic_logits = self.semantic_mlp(review_feat)
        structural_logits = self.structural_mlp(torch.cat([u_feat, i_feat], dim=1))

        gate_input = torch.cat([review_feat, u_feat, i_feat], dim=1)
        alpha = self.gate(gate_input)

        combined = alpha * semantic_logits + (1 - alpha) * structural_logits
        return F.softmax(combined, dim=1)


class FactorizedMessagePassing(nn.Module):
    """Per-factor message passing with K latent factors."""

    def __init__(
        self,
        num_factors: int,
        in_feats: int,
        out_feats: int,
        review_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.num_factors = num_factors
        self.in_feats = in_feats
        self.out_feats = out_feats

        self.user_transforms = nn.Parameter(torch.empty(num_factors, in_feats, out_feats))
        self.item_transforms = nn.Parameter(torch.empty(num_factors, in_feats, out_feats))
        self.review_proj = nn.Linear(review_dim, num_factors, bias=False)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        for k in range(self.num_factors):
            nn.init.xavier_uniform_(self.user_transforms[k])
            nn.init.xavier_uniform_(self.item_transforms[k])
        nn.init.xavier_uniform_(self.review_proj.weight)

    def forward(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        encoder_data: dict,
        norm_factors: dict,
        factor_dist: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (num_users, out_feats), (num_items, out_feats), factor_reps."""
        num_users = user_emb.size(0)
        num_items = item_emb.size(0)

        user_out = user_emb.new_zeros((num_users, self.out_feats))
        item_out = item_emb.new_zeros((num_items, self.out_feats))
        user_factor_reps = []
        item_factor_reps = []

        for k in range(self.num_factors):
            u_transform = self.user_transforms[k]
            i_transform = self.item_transforms[k]

            u_factor = user_emb.new_zeros((num_users, self.out_feats))
            i_factor = item_emb.new_zeros((num_items, self.out_feats))

            for rating_key, edge_data in encoder_data.items():
                edge_users = edge_data["user_ids"]
                edge_items = edge_data["item_ids"]
                review_feat = edge_data["review_feat"]

                if factor_dist is not None and rating_key in factor_dist:
                    edge_factor_weight = factor_dist[rating_key][:, k].unsqueeze(1)
                else:
                    review_factor_logits = self.review_proj(review_feat)
                    edge_factor_weight = torch.softmax(review_factor_logits[:, k], dim=0).unsqueeze(1)

                src_feat_u = item_emb[edge_items] @ i_transform
                src_feat_i = user_emb[edge_users] @ u_transform

                msg_u = (src_feat_u * edge_factor_weight) * norm_factors["item"]["cj"][edge_items]
                msg_i = (src_feat_i * edge_factor_weight) * norm_factors["user"]["cj"][edge_items]

                u_factor.index_add_(0, edge_users, msg_u)
                i_factor.index_add_(0, edge_items, msg_i)

            u_factor = u_factor * norm_factors["user"]["ci"]
            i_factor = i_factor * norm_factors["item"]["ci"]

            user_factor_reps.append(u_factor)
            item_factor_reps.append(i_factor)

            user_out = user_out + u_factor
            item_out = item_out + i_factor

        user_factor_reps = torch.stack(user_factor_reps, dim=0)
        item_factor_reps = torch.stack(item_factor_reps, dim=0)

        return user_out, item_out, user_factor_reps, item_factor_reps


class SGDNContrastiveLoss(nn.Module):
    """Intent-aware contrastive loss across factors (InfoNCE)."""

    def __init__(self, embedding_size: int, temperature: float = 0.2):
        super().__init__()
        self.temperature = temperature
        self.proj = nn.Linear(embedding_size, embedding_size)

    def forward(
        self,
        user_factor_reps: torch.Tensor,
        item_factor_reps: torch.Tensor,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> torch.Tensor:
        """user_factor_reps: (num_factors, num_users, d), item_factor_reps: (num_factors, num_items, d)."""
        num_factors = user_factor_reps.size(0)
        batch_size = user_ids.size(0)

        u_proj = self.proj(user_factor_reps)
        i_proj = self.proj(item_factor_reps)

        u_batch = u_proj[:, user_ids]
        i_batch = i_proj[:, item_ids]

        total_loss = u_batch.new_tensor(0.0)
        count = 0

        for k in range(num_factors):
            u_k = u_batch[k]
            i_k = i_batch[k]

            sim_matrix = u_k @ i_k.T / self.temperature

            labels = torch.arange(batch_size, device=u_k.device)
            loss = F.cross_entropy(sim_matrix, labels)
            total_loss = total_loss + loss
            count += 1

        return total_loss / max(count, 1)


class DisentangleRegularization(nn.Module):
    """Orthogonality regularization across factor representations."""

    def forward(
        self,
        user_factor_reps: torch.Tensor,
        item_factor_reps: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize cosine similarity between different factors.

        user_factor_reps: (num_factors, num_users, d)
        """
        num_factors = user_factor_reps.size(0)

        u_norm = user_factor_reps / (user_factor_reps.norm(dim=-1, keepdim=True) + 1e-8)
        i_norm = item_factor_reps / (item_factor_reps.norm(dim=-1, keepdim=True) + 1e-8)

        u_sim = torch.bmm(u_norm.transpose(0, 1), u_norm.permute(1, 2, 0))
        i_sim = torch.bmm(i_norm.transpose(0, 1), i_norm.permute(1, 2, 0))

        mask = 1 - torch.eye(num_factors, device=u_sim.device)

        u_loss = (u_sim * mask.unsqueeze(0)).abs().mean()
        i_loss = (i_sim * mask.unsqueeze(0)).abs().mean()

        return (u_loss + i_loss) / 2


class SGDN(AbstractRec):
    """Self-supervised Graph Disentangled Networks for Review-based Recommendation."""

    def __init__(self, configs, train_dataset):
        super().__init__()
        self.configs = configs
        self.num_users = train_dataset.num_users
        self.num_items = train_dataset.num_items
        self.embedding_size = int(configs.get("embedding_size", 64))
        self.hidden_dim = int(configs.get("hidden_dim", 64))
        self.review_dim = int(configs.get("review_dim", 384))
        self.num_factors = int(configs.get("num_factors", 4))
        self.num_layers = int(configs.get("num_layers", 1))
        self.dropout = float(configs.get("dropout", 0.3))
        self.use_contrastive = bool(configs.get("use_contrastive", True))
        self.cl_weight = float(configs.get("cl_weight", 0.1))
        self.disentangle_weight = float(configs.get("disentangle_weight", 0.01))
        self.temperature = float(configs.get("temperature", 0.2))
        self.classification = bool(configs.get("classification", True))
        self.rating_vals = [int(v) for v in configs.get("rating_values", [1, 2, 3, 4, 5])]

        self.user_embedding = nn.Embedding(self.num_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_size)

        self.factor_dist_encoder = FactorDistributionEncoder(
            review_dim=self.review_dim,
            embedding_size=self.embedding_size,
            num_factors=self.num_factors,
            dropout=self.dropout,
        )

        self.mp_layers = nn.ModuleList(
            FactorizedMessagePassing(
                num_factors=self.num_factors,
                in_feats=self.embedding_size,
                out_feats=self.hidden_dim,
                review_dim=self.review_dim,
                dropout=self.dropout,
            )
            for _ in range(self.num_layers)
        )

        self.rating_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, len(self.rating_vals) if self.classification else 1),
        )

        self.contrastive_loss = SGDNContrastiveLoss(
            embedding_size=self.hidden_dim,
            temperature=self.temperature,
        )
        self.disentangle_reg = DisentangleRegularization()

        self.rating_loss_fn = nn.CrossEntropyLoss() if self.classification else nn.MSELoss()

        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

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
            raise TypeError("SGDN expects batch data as a dictionary.")
        return prepared

    def encode(self, encoder_data, norm_factors, decoder_user_ids, decoder_item_ids, decoder_review_feat):
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight

        # Compute factor distribution per encoder edge group
        encoder_factor_dist = {}
        for rating_key, edge_data in encoder_data.items():
            fd = self.factor_dist_encoder(
                review_feat=edge_data["review_feat"],
                user_emb=user_emb,
                item_emb=item_emb,
                user_ids=edge_data["user_ids"],
                item_ids=edge_data["item_ids"],
            )
            encoder_factor_dist[rating_key] = fd

        user_out = user_emb
        item_out = item_emb
        all_user_factor_reps = []
        all_item_factor_reps = []

        for layer in self.mp_layers:
            user_out, item_out, u_factor_reps, i_factor_reps = layer(
                user_emb=user_out,
                item_emb=item_out,
                encoder_data=encoder_data,
                norm_factors=norm_factors,
                factor_dist=encoder_factor_dist,
            )
            all_user_factor_reps.append(u_factor_reps)
            all_item_factor_reps.append(i_factor_reps)

        user_factor_reps = torch.stack(all_user_factor_reps, dim=0).mean(dim=0)
        item_factor_reps = torch.stack(all_item_factor_reps, dim=0).mean(dim=0)

        return user_out, item_out, user_factor_reps, item_factor_reps

    def _forward_once(self, batch: Dict[str, Any]):
        user_out, item_out, user_factor_reps, item_factor_reps = self.encode(
            encoder_data=batch["encoder_data"],
            norm_factors=batch["norm_factors"],
            decoder_user_ids=batch["decoder_user_ids"],
            decoder_item_ids=batch["decoder_item_ids"],
            decoder_review_feat=batch.get("decoder_review_feat"),
        )

        u_decoded = user_out[batch["decoder_user_ids"]]
        i_decoded = item_out[batch["decoder_item_ids"]]
        interaction = torch.cat([u_decoded, i_decoded], dim=1)
        pred_ratings = self.rating_predictor(interaction)

        return pred_ratings, user_factor_reps, item_factor_reps

    def _rating_loss(self, pred_ratings: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        if self.classification:
            return self.rating_loss_fn(pred_ratings, batch["labels"])
        return self.rating_loss_fn(pred_ratings.squeeze(-1), batch["ratings"])

    def cal_loss(self, *args, **kwargs):
        batch_data = args[0] if args else kwargs["batch_data"]
        batch = self._prepare_batch(batch_data)

        pred_ratings, user_factor_reps, item_factor_reps = self._forward_once(batch)
        rating_loss = self._rating_loss(pred_ratings, batch)

        cl_loss = rating_loss.new_tensor(0.0)
        disentangle_loss = rating_loss.new_tensor(0.0)

        if self.use_contrastive:
            cl_loss = self.contrastive_loss(
                user_factor_reps=user_factor_reps,
                item_factor_reps=item_factor_reps,
                user_ids=batch["decoder_user_ids"],
                item_ids=batch["decoder_item_ids"],
            )
            disentangle_loss = self.disentangle_reg(user_factor_reps, item_factor_reps)

        total_loss = (
            rating_loss
            + self.cl_weight * cl_loss
            + self.disentangle_weight * disentangle_loss
        )

        loss_dict = {
            "total_loss": float(total_loss.detach().item()),
            "rating_loss": float(rating_loss.detach().item()),
            "cl_loss": float(cl_loss.detach().item()),
            "disentangle_loss": float(disentangle_loss.detach().item()),
        }
        return total_loss, loss_dict

    def predict_ratings(self, batch_data):
        batch = self._prepare_batch(batch_data)
        pred_ratings, _, _ = self._forward_once(batch)
        if self.classification:
            probs = F.softmax(pred_ratings, dim=1)
            rating_vals = pred_ratings.new_tensor(self.rating_vals, dtype=torch.float32)
            return (probs * rating_vals.view(1, -1)).sum(dim=1)
        return pred_ratings.squeeze(-1)

    def predict_scores(self, *args, **kwargs):
        user_ids = args[0] if args else kwargs["user_ids"]
        return torch.zeros((user_ids.size(0), self.num_items), device=user_ids.device)
