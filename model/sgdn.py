from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .abstract import AbstractRec


class FactorDistributionEncoder(nn.Module):
    """Compute factor probabilities from semantic (review) + structural (user/item) signals.

    For each edge, produces a K-way distribution over latent factors.
    """

    def __init__(
        self,
        review_dim: int,
        embedding_size: int,
        num_factors: int,
        factor_dropout: float,
        use_review_prototypes: bool,
    ):
        super().__init__()
        self.num_factors = num_factors
        self.use_review_prototypes = use_review_prototypes

        self.semantic_mlp = nn.Sequential(
            nn.Linear(review_dim, embedding_size),
            nn.GELU(),
            nn.Dropout(factor_dropout),
            nn.Linear(embedding_size, num_factors),
        )
        self.structural_mlp = nn.Sequential(
            nn.Linear(embedding_size * 2, embedding_size),
            nn.GELU(),
            nn.Dropout(factor_dropout),
            nn.Linear(embedding_size, num_factors),
        )
        self.gate = nn.Sequential(
            nn.Linear(review_dim + embedding_size * 2, embedding_size),
            nn.GELU(),
            nn.Linear(embedding_size, 1),
            nn.Sigmoid(),
        )
        if use_review_prototypes:
            self.prototype_gate = nn.Sequential(
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
        review_prototypes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return (num_edges, num_factors) softmax distribution."""
        u_feat = user_emb[user_ids]
        i_feat = item_emb[item_ids]

        semantic_logits = self.semantic_mlp(review_feat)
        structural_logits = self.structural_mlp(torch.cat([u_feat, i_feat], dim=1))

        gate_input = torch.cat([review_feat, u_feat, i_feat], dim=1)
        alpha = self.gate(gate_input)

        combined_logits = alpha * semantic_logits + (1 - alpha) * structural_logits
        combined = F.softmax(combined_logits, dim=1)

        if self.use_review_prototypes and review_prototypes is not None:
            review_norm = F.normalize(review_feat, dim=1)
            proto_norm = F.normalize(review_prototypes, dim=1)
            prototype_dist = F.softmax(review_norm @ proto_norm.T / 0.5, dim=1)
            proto_alpha = self.prototype_gate(gate_input)
            combined = proto_alpha * prototype_dist + (1 - proto_alpha) * combined
            combined = combined / combined.sum(dim=1, keepdim=True).clamp_min(1e-8)

        return combined


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
        encoder_data: Dict[str, Dict[str, torch.Tensor]],
        norm_factors: Dict[str, Dict[str, torch.Tensor]],
        factor_dist: Optional[Dict[str, torch.Tensor]] = None,
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

                src_feat_u = self.dropout(item_emb[edge_items] @ i_transform)
                src_feat_i = self.dropout(user_emb[edge_users] @ u_transform)

                msg_u = (src_feat_u * edge_factor_weight) * norm_factors["item"]["cj"][edge_items]
                msg_i = (src_feat_i * edge_factor_weight) * norm_factors["user"]["cj"][edge_users]

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
    """Intent-aware contrastive loss using factor/interaction distributions.

    This follows the original SGDN `cal_c_loss` idea: two stochastic forward
    views produce interaction representations, and positives are selected by
    top-k similarity in the interaction distribution (`int_dist`) rather than
    by identity alone.
    """

    def __init__(self, embedding_size: int, temperature: float = 0.2, num_pos: int = 1, num_neg: int = 64):
        super().__init__()
        self.temperature = temperature
        self.num_pos = num_pos
        self.num_neg = num_neg
        self.proj = nn.Linear(embedding_size, embedding_size)

    def forward(
        self,
        interaction_view1: torch.Tensor,
        interaction_view2: torch.Tensor,
        int_dist: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        z1 = F.normalize(self.proj(interaction_view1), dim=1)
        z2 = F.normalize(self.proj(interaction_view2), dim=1)
        int_dist = F.normalize(int_dist, dim=1)

        total_loss = z1.new_tensor(0.0)
        count = 0

        for label in labels.unique(sorted=True):
            group_idx = torch.where(labels == label)[0]
            group_size = int(group_idx.numel())
            if group_size < 2:
                continue

            group_z1 = z1[group_idx]
            group_z2 = z2[group_idx]
            group_dist = int_dist[group_idx]
            k_pos = min(self.num_pos, group_size)
            k_neg = min(self.num_neg, group_size)

            intent_sim = group_dist @ group_dist.T
            _, pos_idx = torch.topk(intent_sim, k=k_pos, dim=1)
            pos_score = torch.exp((group_z1.unsqueeze(1) * group_z2[pos_idx]).sum(dim=2) / self.temperature).sum(dim=1)

            neg_idx = torch.randperm(group_size, device=z1.device)[:k_neg]
            ttl_score = torch.exp((group_z1 @ group_z2[neg_idx].T) / self.temperature).sum(dim=1)
            loss = -torch.log(pos_score / ttl_score.clamp_min(1e-8)).mean()
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
        self.gcn_dropout = float(configs.get("gcn_dropout", self.dropout))
        self.factor_dropout = float(configs.get("factor_dropout", self.dropout))
        self.pred_dropout = float(configs.get("pred_dropout", self.dropout))
        self.init_pred_bias_with_rating_mean = bool(configs.get("init_pred_bias_with_rating_mean", False))
        self.use_contrastive = bool(configs.get("use_contrastive", True))
        self.cl_weight = float(configs.get("cl_weight", 0.1))
        self.disentangle_weight = float(configs.get("disentangle_weight", 0.01))
        self.temperature = float(configs.get("temperature", 0.2))
        self.num_pos = int(configs.get("num_pos", 1))
        self.num_neg = int(configs.get("num_neg", 64))
        self.use_review_prototypes = bool(configs.get("use_review_prototypes", True))
        self.init_review_prototypes = bool(configs.get("init_review_prototypes", True))
        self.debug_shapes = bool(configs.get("debug_shapes", False))
        self.classification = bool(configs.get("classification", True))
        self.rating_vals = [int(v) for v in configs.get("rating_values", [1, 2, 3, 4, 5])]
        self._shape_logged = False

        self.user_embedding = nn.Embedding(self.num_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_size)
        self.review_prototypes = nn.Parameter(torch.empty(self.num_factors, self.review_dim))

        self.factor_dist_encoder = FactorDistributionEncoder(
            review_dim=self.review_dim,
            embedding_size=self.embedding_size,
            num_factors=self.num_factors,
            factor_dropout=self.factor_dropout,
            use_review_prototypes=self.use_review_prototypes,
        )

        layer_dims = [self.embedding_size] + [self.hidden_dim] * self.num_layers
        self.mp_layers = nn.ModuleList(
            [
                FactorizedMessagePassing(
                    num_factors=self.num_factors,
                    in_feats=layer_dims[layer_idx],
                    out_feats=layer_dims[layer_idx + 1],
                    review_dim=self.review_dim,
                    dropout=self.gcn_dropout,
                )
                for layer_idx in range(self.num_layers)
            ]
        )

        self.rating_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.pred_dropout),
            nn.Linear(self.hidden_dim, len(self.rating_vals) if self.classification else 1),
        )

        self.contrastive_loss = SGDNContrastiveLoss(
            embedding_size=self.hidden_dim * 2,
            temperature=self.temperature,
            num_pos=self.num_pos,
            num_neg=self.num_neg,
        )
        self.disentangle_reg = DisentangleRegularization()

        self.rating_loss_fn = nn.CrossEntropyLoss() if self.classification else nn.MSELoss()

        self._init_embeddings()
        if self.use_review_prototypes and self.init_review_prototypes:
            self._init_review_prototypes(train_dataset)
        if self.init_pred_bias_with_rating_mean:
            self._init_pred_bias_with_train_mean(train_dataset)

    def _init_embeddings(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.review_prototypes)

    def _init_review_prototypes(self, train_dataset) -> None:
        review_feat = getattr(train_dataset, "decoder_review_feat", None)
        if not isinstance(review_feat, torch.Tensor) or review_feat.numel() == 0:
            return
        try:
            from sklearn.cluster import KMeans

            num_clusters = min(self.num_factors, review_feat.size(0))
            kmeans = KMeans(n_clusters=num_clusters, random_state=int(self.configs.get("seed", 42)))
            kmeans.fit(review_feat.detach().cpu().numpy())
            centroids = torch.as_tensor(kmeans.cluster_centers_, dtype=self.review_prototypes.dtype)
            if num_clusters < self.num_factors:
                pad = self.review_prototypes.detach().cpu()[num_clusters:]
                centroids = torch.cat([centroids, pad], dim=0)
            with torch.no_grad():
                self.review_prototypes.copy_(F.normalize(centroids, dim=1))
            print(f"SGDN initialized review prototypes with KMeans: {tuple(self.review_prototypes.shape)}")
        except Exception as exc:
            # Optional author-code alignment path; random prototypes remain valid.
            print(f"SGDN review prototype KMeans init skipped: {exc}")

    def _init_pred_bias_with_train_mean(self, train_dataset) -> None:
        if self.classification:
            return
        ratings = getattr(train_dataset, "ratings", None)
        if not isinstance(ratings, torch.Tensor) or ratings.numel() == 0:
            return
        final_layer = self.rating_predictor[-1]
        if not isinstance(final_layer, nn.Linear) or final_layer.bias is None:
            return
        rating_mean = float(ratings.float().mean().item())
        with torch.no_grad():
            final_layer.bias.fill_(rating_mean)
        print(f"SGDN initialized regression predictor bias with train rating mean: {rating_mean:.4f}")

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

    def _compute_factor_dist(self, review_feat, user_emb, item_emb, user_ids, item_ids):
        return self.factor_dist_encoder(
            review_feat=review_feat,
            user_emb=user_emb,
            item_emb=item_emb,
            user_ids=user_ids,
            item_ids=item_ids,
            review_prototypes=self.review_prototypes if self.use_review_prototypes else None,
        )

    def _assert_and_log_shapes(self, batch, encoder_factor_dist, int_dist, user_factor_reps, item_factor_reps):
        assert int_dist.dim() == 2 and int_dist.size(1) == self.num_factors, f"int_dist shape mismatch: {tuple(int_dist.shape)}"
        assert user_factor_reps.shape == (self.num_factors, self.num_users, self.hidden_dim), tuple(user_factor_reps.shape)
        assert item_factor_reps.shape == (self.num_factors, self.num_items, self.hidden_dim), tuple(item_factor_reps.shape)
        for node_type in ("user", "item"):
            for norm_name in ("ci", "cj"):
                assert batch["norm_factors"][node_type][norm_name].dim() == 2
        if self.debug_shapes and not self._shape_logged:
            edge_shapes = {rating: tuple(data["user_ids"].shape) for rating, data in batch["encoder_data"].items()}
            factor_shapes = {rating: tuple(dist.shape) for rating, dist in encoder_factor_dist.items()}
            norm_shapes = {
                node_type: {name: tuple(value.shape) for name, value in norms.items()}
                for node_type, norms in batch["norm_factors"].items()
            }
            print(
                "SGDN shape check:",
                {
                    "encoder_edges": edge_shapes,
                    "norm_factors": norm_shapes,
                    "factor_dist": factor_shapes,
                    "int_dist": tuple(int_dist.shape),
                },
            )
            self._shape_logged = True

    def encode(self, encoder_data, norm_factors, decoder_user_ids, decoder_item_ids, decoder_review_feat):
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight

        # Compute factor distribution per encoder edge group
        encoder_factor_dist = {}
        for rating_key, edge_data in encoder_data.items():
            fd = self._compute_factor_dist(edge_data["review_feat"], user_emb, item_emb, edge_data["user_ids"], edge_data["item_ids"])
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
        int_dist = self._compute_factor_dist(decoder_review_feat, user_emb, item_emb, decoder_user_ids, decoder_item_ids)

        return user_out, item_out, user_factor_reps, item_factor_reps, encoder_factor_dist, int_dist

    def _forward_once(self, batch: Dict[str, Any]):
        user_out, item_out, user_factor_reps, item_factor_reps, encoder_factor_dist, int_dist = self.encode(
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
        self._assert_and_log_shapes(batch, encoder_factor_dist, int_dist, user_factor_reps, item_factor_reps)

        return pred_ratings, user_factor_reps, item_factor_reps, interaction, int_dist

    def _rating_loss(self, pred_ratings: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        if self.classification:
            return self.rating_loss_fn(pred_ratings, batch["labels"])
        return self.rating_loss_fn(pred_ratings.squeeze(-1), batch["ratings"])

    def cal_loss(self, *args, **kwargs):
        batch_data = args[0] if args else kwargs["batch_data"]
        batch = self._prepare_batch(batch_data)

        pred_ratings1, user_factor_reps1, item_factor_reps1, interaction1, int_dist1 = self._forward_once(batch)
        if self.use_contrastive:
            pred_ratings2, _, _, interaction2, _ = self._forward_once(batch)
            rating_loss = (self._rating_loss(pred_ratings1, batch) + self._rating_loss(pred_ratings2, batch)) / 2
        else:
            pred_ratings2 = None
            interaction2 = None
            rating_loss = self._rating_loss(pred_ratings1, batch)

        cl_loss = rating_loss.new_tensor(0.0)
        disentangle_loss = rating_loss.new_tensor(0.0)

        if self.use_contrastive:
            cl_loss = self.contrastive_loss(
                interaction_view1=interaction1,
                interaction_view2=interaction2,
                int_dist=int_dist1,
                labels=batch["labels"],
            )
            disentangle_loss = self.disentangle_reg(user_factor_reps1, item_factor_reps1)

        total_loss = (
            rating_loss
            + self.cl_weight * cl_loss
            + self.disentangle_weight * disentangle_loss
        )

        if self.classification:
            probs = F.softmax(pred_ratings1, dim=1)
            rating_vals = pred_ratings1.new_tensor(self.rating_vals, dtype=torch.float32)
            pred_for_diag = (probs * rating_vals.view(1, -1)).sum(dim=1)
        else:
            pred_for_diag = pred_ratings1.squeeze(-1)

        loss_dict = {
            "total_loss": float(total_loss.detach().item()),
            "rating_loss": float(rating_loss.detach().item()),
            "cl_loss": float(cl_loss.detach().item()),
            "disentangle_loss": float(disentangle_loss.detach().item()),
            "pred_min": float(pred_for_diag.detach().min().item()),
            "pred_max": float(pred_for_diag.detach().max().item()),
            "pred_mean": float(pred_for_diag.detach().mean().item()),
            "pred_std": float(pred_for_diag.detach().std(unbiased=False).item()),
        }
        return total_loss, loss_dict

    def predict_ratings(self, batch_data):
        batch = self._prepare_batch(batch_data)
        pred_ratings, _, _, _, _ = self._forward_once(batch)
        if self.classification:
            probs = F.softmax(pred_ratings, dim=1)
            rating_vals = pred_ratings.new_tensor(self.rating_vals, dtype=torch.float32)
            return (probs * rating_vals.view(1, -1)).sum(dim=1)
        return pred_ratings.squeeze(-1)

    def predict_scores(self, *args, **kwargs):
        user_ids = args[0] if args else kwargs["user_ids"]
        return torch.zeros((user_ids.size(0), self.num_items), device=user_ids.device)
