from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .abstract import AbstractRec
from .rgcl import ContrastLoss


class MAEdgeGate(nn.Module):
    """Learnable edge gate g_ui from user, item, review, rating, sentiment, and mismatch."""

    def __init__(self, hidden_dim: int, review_dim: int, gate_hidden_dim: int):
        super().__init__()
        self.review_proj = nn.Linear(review_dim, hidden_dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 3, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        review_feat: torch.Tensor,
        normalized_rating: torch.Tensor,
        sentiment_score: torch.Tensor,
        misalign_score: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # user_emb/item_emb/review_shared: (num_edges, hidden_dim); scalar features: (num_edges, 1).
        review_shared = self.review_proj(review_feat)
        gate_input = torch.cat(
            [
                user_emb,
                item_emb,
                review_shared,
                normalized_rating.view(-1, 1),
                sentiment_score.view(-1, 1),
                misalign_score.view(-1, 1),
            ],
            dim=1,
        )
        gate = torch.sigmoid(self.mlp(gate_input)).squeeze(1)
        return gate, review_shared


class MAGCMCGraphConv(nn.Module):
    """Rating-specific message passing with soft shared/residual propagation."""

    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        review_dim: int,
        dropout_rate: float,
        gate_hidden_dim: int,
        epsilon: float,
        lambda_residual: float,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_feats, out_feats))
        self.dropout = nn.Dropout(dropout_rate)
        self.prob_score = nn.Linear(review_dim, 1, bias=False)
        self.shared_score = nn.Linear(review_dim, 1, bias=False)
        self.residual_score = nn.Linear(review_dim, 1, bias=False)
        self.residual_proj = nn.Linear(review_dim, out_feats, bias=False)
        self.edge_gate = MAEdgeGate(out_feats, review_dim, gate_hidden_dim)
        self.epsilon = float(epsilon)
        self.lambda_residual = float(lambda_residual)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.prob_score.weight)
        nn.init.xavier_uniform_(self.shared_score.weight)
        nn.init.xavier_uniform_(self.residual_score.weight)
        nn.init.xavier_uniform_(self.residual_proj.weight)

    def forward(
        self,
        src_ids: torch.Tensor,
        dst_ids: torch.Tensor,
        review_feat: torch.Tensor,
        src_cj: torch.Tensor,
        dst_ci: torch.Tensor,
        num_dst: int,
        dst_weight: torch.Tensor,
        normalized_rating: torch.Tensor,
        sentiment_score: torch.Tensor,
        misalign_score: torch.Tensor,
        user_is_dst: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if src_ids.numel() == 0:
            empty = self.weight.new_zeros((0,))
            return self.weight.new_zeros((num_dst, self.weight.size(1))), empty, empty, empty, empty, empty

        src_feat = self.weight[src_ids]
        dst_feat = dst_weight[dst_ids]
        if user_is_dst:
            user_feat, item_feat = dst_feat, src_feat
        else:
            user_feat, item_feat = src_feat, dst_feat

        # Gate tensors: (num_edges,). No interaction is removed; gates only scale propagation strength.
        gate, review_shared = self.edge_gate(
            user_emb=user_feat,
            item_emb=item_feat,
            review_feat=review_feat,
            normalized_rating=normalized_rating,
            sentiment_score=sentiment_score,
            misalign_score=misalign_score,
        )
        gamma = self.epsilon + self.lambda_residual * (1.0 - gate)

        pa = torch.sigmoid(self.prob_score(review_feat))
        shared_ra = torch.sigmoid(self.shared_score(review_feat))
        residual_ra = torch.sigmoid(self.residual_score(review_feat))
        residual_review = self.residual_proj(review_feat)

        # shared/residual messages: (num_edges, hidden_dim).
        shared_message = src_feat * pa + review_shared * shared_ra
        residual_message = residual_review * residual_ra
        message = gate.view(-1, 1) * shared_message + gamma.view(-1, 1) * residual_message
        message = message * self.dropout(src_cj[src_ids])

        out = self.weight.new_zeros((num_dst, self.weight.size(1)))
        out.index_add_(0, dst_ids, message)
        residual_energy = residual_message.pow(2).mean(dim=1)
        align_loss = F.mse_loss(gate, (1.0 - misalign_score).clamp(0.0, 1.0))
        return out * dst_ci, gate, gamma, misalign_score, residual_energy, align_loss


class MAGCMCLayer(nn.Module):
    def __init__(
        self,
        rating_vals: list[int],
        user_in_units: int,
        item_in_units: int,
        review_dim: int,
        out_units: int,
        dropout_rate: float,
        gate_hidden_dim: int,
        epsilon: float,
        lambda_residual: float,
    ):
        super().__init__()
        self.rating_vals = rating_vals
        self.user_convs = nn.ModuleDict()
        self.item_convs = nn.ModuleDict()
        for rating in rating_vals:
            key = str(rating)
            self.user_convs[key] = MAGCMCGraphConv(
                item_in_units, out_units, review_dim, dropout_rate, gate_hidden_dim, epsilon, lambda_residual
            )
            self.item_convs[key] = MAGCMCGraphConv(
                user_in_units, out_units, review_dim, dropout_rate, gate_hidden_dim, epsilon, lambda_residual
            )

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
        gate_values = []
        gamma_values = []
        misalign_values = []
        residual_values = []
        align_values = []

        for rating in self.rating_vals:
            key = str(rating)
            edge_data = encoder_data.get(key)
            if edge_data is None:
                continue

            edge_users = edge_data["user_ids"]
            edge_items = edge_data["item_ids"]
            review_feat = edge_data["review_feat"]
            normalized_rating = edge_data.get("normalized_rating")
            sentiment_score = edge_data.get("sentiment_score")
            misalign_score = edge_data.get("edge_misalign_score")
            if normalized_rating is None:
                normalized_rating = review_feat.new_full((edge_users.size(0),), (float(rating) - 1.0) / 4.0)
            if sentiment_score is None:
                sentiment_score = normalized_rating
            if misalign_score is None:
                misalign_score = (normalized_rating - sentiment_score).abs()

            user_msg, user_gate, user_gamma, user_misalign, user_residual, user_align = self.user_convs[key](
                src_ids=edge_items,
                dst_ids=edge_users,
                review_feat=review_feat,
                src_cj=norm_factors["item"]["cj"],
                dst_ci=norm_factors["user"]["ci"],
                num_dst=num_users,
                dst_weight=self.item_convs[key].weight,
                normalized_rating=normalized_rating,
                sentiment_score=sentiment_score,
                misalign_score=misalign_score,
                user_is_dst=True,
            )
            item_msg, item_gate, item_gamma, item_misalign, item_residual, item_align = self.item_convs[key](
                src_ids=edge_users,
                dst_ids=edge_items,
                review_feat=review_feat,
                src_cj=norm_factors["user"]["cj"],
                dst_ci=norm_factors["item"]["ci"],
                num_dst=num_items,
                dst_weight=self.user_convs[key].weight,
                normalized_rating=normalized_rating,
                sentiment_score=sentiment_score,
                misalign_score=misalign_score,
                user_is_dst=False,
            )
            user_out = user_out + user_msg
            item_out = item_out + item_msg
            gate_values.extend([user_gate.detach(), item_gate.detach()])
            gamma_values.extend([user_gamma.detach(), item_gamma.detach()])
            misalign_values.extend([user_misalign.detach(), item_misalign.detach()])
            residual_values.extend([user_residual, item_residual])
            align_values.extend([user_align.view(1), item_align.view(1)])

        user_out = self.ufc(self.dropout(self.activation(user_out)))
        item_out = self.ifc(self.dropout(self.activation(item_out)))
        return user_out, item_out, gate_values, gamma_values, misalign_values, residual_values, align_values


class MAMLPPredictorMI(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        review_dim: int,
        classification: bool,
        num_classes: int,
        dropout_rate: float,
        gate_hidden_dim: int,
        decoder_review_mode: str,
    ):
        super().__init__()
        self.classification = classification
        self.decoder_review_mode = decoder_review_mode
        self.edge_gate = MAEdgeGate(hidden_dim, review_dim, gate_hidden_dim)
        self.contrast_loss = ContrastLoss(hidden_dim)
        self.review_generator = nn.Sequential(
            nn.Linear(hidden_dim * 2 + review_dim * 4, max(gate_hidden_dim, review_dim)),
            nn.GELU(),
            nn.Linear(max(gate_hidden_dim, review_dim), review_dim),
        )
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 1, hidden_dim, bias=False),
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
        user_history_feat: Optional[torch.Tensor] = None,
        item_history_feat: Optional[torch.Tensor] = None,
        normalized_rating: Optional[torch.Tensor] = None,
        sentiment_score: Optional[torch.Tensor] = None,
        misalign_score: Optional[torch.Tensor] = None,
        target_review_feat: Optional[torch.Tensor] = None,
        cal_edge_mi: bool = True,
    ):
        edge_user = user_emb[user_ids]
        edge_item = item_emb[item_ids]
        generation_loss = edge_user.new_tensor(0.0)
        if self.decoder_review_mode == "generated":
            if user_history_feat is None:
                user_history_feat = edge_user.new_zeros((edge_user.size(0), self.edge_gate.review_proj.in_features))
            if item_history_feat is None:
                item_history_feat = edge_user.new_zeros((edge_user.size(0), self.edge_gate.review_proj.in_features))
            generator_input = torch.cat(
                [
                    edge_user,
                    edge_item,
                    user_history_feat,
                    item_history_feat,
                    user_history_feat * item_history_feat,
                    torch.abs(user_history_feat - item_history_feat),
                ],
                dim=1,
            )
            review_feat = self.review_generator(generator_input)
            if target_review_feat is not None:
                generation_loss = F.mse_loss(review_feat, target_review_feat)
        elif review_feat is None:
            review_feat = edge_user.new_zeros((edge_user.size(0), self.edge_gate.review_proj.in_features))
        if normalized_rating is None:
            normalized_rating = edge_user.new_full((edge_user.size(0),), 0.5)
        if sentiment_score is None:
            sentiment_score = edge_user.new_full((edge_user.size(0),), 0.5)
        if misalign_score is None:
            misalign_score = (normalized_rating - sentiment_score).abs()

        gate, projected_review = self.edge_gate(edge_user, edge_item, review_feat, normalized_rating, sentiment_score, misalign_score)
        # Prediction input shape: (num_decoder_edges, hidden_dim * 3 + 1).
        interaction = self.linear(torch.cat([edge_user, edge_item, projected_review, gate.view(-1, 1)], dim=1))
        interaction = self.dropout(interaction)
        scores = self.predictor(interaction)

        if cal_edge_mi and review_feat is not None:
            edge_mi = self.contrast_loss(interaction, projected_review)
            return scores, edge_mi, gate, misalign_score, generation_loss
        return scores, gate, misalign_score, generation_loss


class MA_RGCL(AbstractRec):
    """Misalignment-aware RGCL with soft shared/residual graph propagation."""

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
        self.align_weight = float(configs.get("align_weight", 0.02))
        self.residual_weight = float(configs.get("residual_weight", 0.001))
        self.epsilon = float(configs.get("epsilon", 0.05))
        self.lambda_residual = float(configs.get("lambda_residual", 0.25))
        self.gate_hidden_dim = int(configs.get("gate_hidden_dim", self.embedding_size))
        self.decoder_review_mode = str(configs.get("decoder_review_mode", "generated")).lower()
        if self.decoder_review_mode not in {"target", "generated"}:
            raise ValueError("decoder_review_mode must be either 'target' or 'generated'.")
        self.review_generation_weight = float(configs.get("review_generation_weight", 0.05))
        self.classification = bool(configs.get("classification", True))
        self.log_gate_stats = self._get_bool_config("log_gate_stats", True)
        self.rating_vals = [int(v) for v in configs.get("rating_values", [1, 2, 3, 4, 5])]
        self.current_gate_stats = {}

        self.encoder = nn.ModuleList(
            MAGCMCLayer(
                rating_vals=self.rating_vals,
                user_in_units=self.num_users,
                item_in_units=self.num_items,
                review_dim=self.review_dim,
                out_units=self.embedding_size,
                dropout_rate=self.dropout,
                gate_hidden_dim=self.gate_hidden_dim,
                epsilon=self.epsilon,
                lambda_residual=self.lambda_residual,
            )
            for _ in range(self.num_layers)
        )
        self.decoder = MAMLPPredictorMI(
            hidden_dim=self.embedding_size,
            review_dim=self.review_dim,
            classification=self.classification,
            num_classes=len(self.rating_vals),
            dropout_rate=self.dropout,
            gate_hidden_dim=self.gate_hidden_dim,
            decoder_review_mode=self.decoder_review_mode,
        )
        self.nd_contrast = ContrastLoss(self.embedding_size)
        self.rating_loss = nn.CrossEntropyLoss() if self.classification else nn.MSELoss()

    def _get_bool_config(self, key: str, default: bool = False) -> bool:
        value = self.configs.get(key, default)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

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
            raise TypeError("MA_RGCL expects batch data as a dictionary.")
        return prepared

    def _summarize_gate_stats(
        self,
        gate_values: list[torch.Tensor],
        gamma_values: list[torch.Tensor],
        misalign_values: list[torch.Tensor],
    ) -> dict[str, float]:
        if not gate_values:
            return {
                "mean_gate": float("nan"),
                "std_gate": float("nan"),
                "min_gate_value": float("nan"),
                "max_gate_value": float("nan"),
                "mean_gamma": float("nan"),
                "mean_edge_misalign_score": float("nan"),
            }
        gates = torch.cat([gate.view(-1) for gate in gate_values])
        gammas = torch.cat([gamma.view(-1) for gamma in gamma_values])
        misalign = torch.cat([score.view(-1) for score in misalign_values])
        return {
            "mean_gate": float(gates.mean().item()),
            "std_gate": float(gates.std(unbiased=False).item()),
            "min_gate_value": float(gates.min().item()),
            "max_gate_value": float(gates.max().item()),
            "mean_gamma": float(gammas.mean().item()),
            "mean_edge_misalign_score": float(misalign.mean().item()),
        }

    def encode(self, encoder_data: dict[str, dict[str, torch.Tensor]], norm_factors: dict[str, dict[str, torch.Tensor]]):
        user_out = None
        item_out = None
        gate_values = []
        gamma_values = []
        misalign_values = []
        residual_values = []
        align_values = []
        for layer in self.encoder:
            user_out, item_out, gates, gammas, misalign, residual, align = layer(encoder_data, norm_factors)
            gate_values.extend(gates)
            gamma_values.extend(gammas)
            misalign_values.extend(misalign)
            residual_values.extend(residual)
            align_values.extend(align)
        if user_out is None or item_out is None:
            raise RuntimeError("MA_RGCL encoder produced no outputs.")
        self.current_gate_stats = self._summarize_gate_stats(gate_values, gamma_values, misalign_values)
        residual_loss = user_out.new_tensor(0.0)
        if residual_values:
            residual_loss = torch.cat([value.view(-1) for value in residual_values]).mean()
        align_loss = user_out.new_tensor(0.0)
        if align_values:
            align_loss = torch.cat([value.view(-1) for value in align_values]).mean()
        return user_out, item_out, residual_loss, align_loss

    def _forward_once(self, batch: Dict[str, Any], cal_edge_mi: bool):
        user_out, item_out, residual_loss, align_loss = self.encode(batch["encoder_data"], batch["norm_factors"])
        decode_result = self.decoder(
            user_emb=user_out,
            item_emb=item_out,
            user_ids=batch["decoder_user_ids"],
            item_ids=batch["decoder_item_ids"],
            review_feat=batch.get("decoder_review_feat") if self.decoder_review_mode == "target" else None,
            user_history_feat=batch.get("decoder_user_history_feat"),
            item_history_feat=batch.get("decoder_item_history_feat"),
            normalized_rating=None,
            sentiment_score=batch.get("decoder_sentiment_score") if self.decoder_review_mode == "target" else None,
            misalign_score=None,
            target_review_feat=batch.get("decoder_review_feat") if self.training else None,
            cal_edge_mi=cal_edge_mi,
        )
        if cal_edge_mi:
            pred_ratings, ed_loss, decoder_gate, decoder_misalign, generation_loss = decode_result
        else:
            pred_ratings, decoder_gate, decoder_misalign, generation_loss = decode_result
            ed_loss = pred_ratings.new_tensor(0.0)
        return pred_ratings, ed_loss, user_out, item_out, align_loss, residual_loss, generation_loss

    def _rating_loss(self, pred_ratings: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        if self.classification:
            return self.rating_loss(pred_ratings, batch["labels"])
        return self.rating_loss(pred_ratings.squeeze(-1), batch["ratings"])

    def cal_loss(self, *args, **kwargs):
        batch_data = args[0] if args else kwargs["batch_data"]
        batch = self._prepare_batch(batch_data)

        if self.use_contrastive:
            pred1, ed1, user1, item1, align1, res1, gen1 = self._forward_once(batch, cal_edge_mi=True)
            pred2, ed2, user2, item2, align2, res2, gen2 = self._forward_once(batch, cal_edge_mi=True)
            rating_loss = (self._rating_loss(pred1, batch) + self._rating_loss(pred2, batch)) / 2.0
            nd_loss = (self.nd_contrast(user1, user2).mean() + self.nd_contrast(item1, item2).mean()) / 2.0
            ed_loss = (ed1.mean() + ed2.mean()) / 2.0
            align_loss = (align1 + align2) / 2.0
            residual_loss = (res1 + res2) / 2.0
            generation_loss = (gen1 + gen2) / 2.0
        else:
            pred, ed_loss, user_out, item_out, align_loss, residual_loss, generation_loss = self._forward_once(batch, cal_edge_mi=False)
            rating_loss = self._rating_loss(pred, batch)
            nd_loss = rating_loss.new_tensor(0.0)
            ed_loss = rating_loss.new_tensor(0.0)

        total_loss = (
            rating_loss
            + self.nd_weight * nd_loss
            + self.ed_weight * ed_loss
            + self.align_weight * align_loss
            + self.residual_weight * residual_loss
            + self.review_generation_weight * generation_loss
        )
        loss_dict = {
            "total_loss": float(total_loss.detach().item()),
            "rating_loss": float(rating_loss.detach().item()),
            "nd_loss": float(nd_loss.detach().item()),
            "ed_loss": float(ed_loss.detach().item()),
            "align_loss": float(align_loss.detach().item()),
            "residual_loss": float(residual_loss.detach().item()),
            "generation_loss": float(generation_loss.detach().item()),
        }
        if self.log_gate_stats:
            loss_dict.update(self.current_gate_stats)
        return total_loss, loss_dict

    def predict_ratings(self, batch_data):
        batch = self._prepare_batch(batch_data)
        pred_ratings, _, _, _, _, _, _ = self._forward_once(batch, cal_edge_mi=False)
        if self.classification:
            probs = F.softmax(pred_ratings, dim=1)
            rating_vals = pred_ratings.new_tensor(self.rating_vals, dtype=torch.float32)
            return (probs * rating_vals.view(1, -1)).sum(dim=1)
        return pred_ratings.squeeze(-1)

    def predict_scores(self, *args, **kwargs):
        user_ids = args[0] if args else kwargs["user_ids"]
        return torch.zeros((user_ids.size(0), self.num_items), device=user_ids.device)
