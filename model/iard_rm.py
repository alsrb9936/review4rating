import torch
import torch.nn as nn
import torch.nn.functional as F

from .abstract import AbstractRec


class RatingGraphEncoder(nn.Module):
    def __init__(self, num_users, num_items, d_id=64, d_model=128, num_layers=2):
        super().__init__()
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.num_nodes = self.num_users + self.num_items
        self.d_id = int(d_id)
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.user_embedding = nn.Embedding(self.num_users, self.d_id)
        self.item_embedding = nn.Embedding(self.num_items, self.d_id)
        self.projection = nn.Linear(self.d_id * 3, self.d_model)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)

    def _propagate_once(self, node_embeddings, edge_index, edge_weight=None):
        src = edge_index[0]
        dst = edge_index[1]
        if edge_weight is None:
            base_weight = node_embeddings.new_ones(src.size(0))
        else:
            base_weight = edge_weight

        degree = node_embeddings.new_zeros(self.num_nodes)
        degree.index_add_(0, src, base_weight)
        norm = base_weight / torch.sqrt(degree[src].clamp_min(1e-8) * degree[dst].clamp_min(1e-8))

        aggregated = node_embeddings.new_zeros(node_embeddings.size())
        aggregated.index_add_(0, dst, node_embeddings[src] * norm.unsqueeze(-1))
        return aggregated

    def _compute_node_embeddings(self, edge_index, edge_weight=None):
        edge_index = edge_index.to(self.user_embedding.weight.device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(self.user_embedding.weight.device)

        initial_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        layer_outputs = [initial_embeddings]
        propagated = initial_embeddings
        for _ in range(self.num_layers):
            propagated = self._propagate_once(propagated, edge_index=edge_index, edge_weight=edge_weight)
            layer_outputs.append(propagated)
        stacked = torch.stack(layer_outputs, dim=0)
        return stacked.mean(dim=0)

    def forward(self, user_ids, item_ids, edge_index, edge_weight=None):
        all_embeddings = self._compute_node_embeddings(edge_index=edge_index, edge_weight=edge_weight)
        user_embeddings = all_embeddings[user_ids]
        item_embeddings = all_embeddings[self.num_users + item_ids]
        interaction_embeddings = torch.cat(
            [user_embeddings, item_embeddings, user_embeddings * item_embeddings],
            dim=-1,
        )
        return self.projection(interaction_embeddings)


class ReviewProjectionEncoder(nn.Module):
    def __init__(self, d_text=768, d_model=128, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_text, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self._init_weights()

    def _init_weights(self):
        for module in self.encoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, review_emb):
        return self.layer_norm(self.encoder(review_emb))


class HistoryAttentionEncoder(nn.Module):
    def __init__(self, raw_dim=768, d_model=128):
        super().__init__()
        self.raw_dim = int(raw_dim)
        self.d_model = int(d_model)
        self.key_projection = nn.Linear(self.raw_dim, self.d_model)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.key_projection.weight)
        if self.key_projection.bias is not None:
            nn.init.zeros_(self.key_projection.bias)

    def _attend(self, query, history_emb, history_mask):
        key = self.key_projection(history_emb)
        scores = torch.sum(query.unsqueeze(1) * key, dim=-1) / float(self.d_model) ** 0.5
        history_mask = history_mask.bool()
        scores = scores.masked_fill(~history_mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        has_history = history_mask.any(dim=-1, keepdim=True)
        attention = torch.where(has_history, attention, torch.zeros_like(attention))
        return torch.sum(attention.unsqueeze(-1) * history_emb, dim=1)

    @staticmethod
    def _fuse(user_context, item_context):
        return torch.cat(
            [
                user_context,
                item_context,
                user_context * item_context,
                torch.abs(user_context - item_context),
            ],
            dim=-1,
        )

    def forward(self, query, user_history_emb, user_history_mask, item_history_emb, item_history_mask):
        user_context = self._attend(query, user_history_emb, user_history_mask)
        item_context = self._attend(query, item_history_emb, item_history_mask)
        return self._fuse(user_context, item_context)


class PrototypeIntentExtractor(nn.Module):
    def __init__(self, d_model=128, num_intents=8, tau_p=0.2):
        super().__init__()
        self.d_model = int(d_model)
        self.num_intents = int(num_intents)
        self.tau_p = float(tau_p)
        self.prototypes = nn.Parameter(torch.empty(self.num_intents, self.d_model))
        self.layer_norm = nn.LayerNorm(self.d_model)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, hX):
        normalized_hX = F.normalize(hX, dim=-1, eps=1e-8)
        normalized_prototypes = F.normalize(self.prototypes, dim=-1, eps=1e-8)
        scores = normalized_hX @ normalized_prototypes.T
        scores = scores / self.tau_p
        intent_weights = F.softmax(scores, dim=-1)
        zX_proto = intent_weights @ self.prototypes
        zX = self.layer_norm(zX_proto + hX)
        return zX, intent_weights


class SharedResidualDisentangler(nn.Module):
    def __init__(self, d_model=128, dropout=0.1, disentangler_mode="conditioned"):
        super().__init__()
        self.disentangler_mode = str(disentangler_mode)
        if self.disentangler_mode == "conditioned":
            pair_dim = d_model * 4
        elif self.disentangler_mode == "independent":
            pair_dim = d_model
        else:
            raise ValueError(f"Unsupported disentangler_mode: {self.disentangler_mode}")
        self.shared_mlp = nn.Sequential(
            nn.Linear(pair_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.residual_mlp = nn.Sequential(
            nn.Linear(pair_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.recon_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.shared_norm = nn.LayerNorm(d_model)
        self.residual_norm = nn.LayerNorm(d_model)
        self.recon_norm = nn.LayerNorm(d_model)
        self._init_weights()

    def _init_weights(self):
        for module in list(self.shared_mlp) + list(self.residual_mlp) + list(self.recon_mlp):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, zX, zY):
        if self.disentangler_mode == "conditioned":
            pair_representation = torch.cat([zX, zY, zX * zY, torch.abs(zX - zY)], dim=-1)
        else:
            pair_representation = zX
        zS = self.shared_norm(self.shared_mlp(pair_representation))
        zR = self.residual_norm(self.residual_mlp(pair_representation))
        recon_zX = self.recon_norm(self.recon_mlp(torch.cat([zS, zR], dim=-1)))
        return zS, zR, recon_zX


class InconsistencyGate(nn.Module):
    def __init__(self, d_model=128, dropout=0.1, gate_alpha=5.0, fixed_gate_value=None):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(gate_alpha), dtype=torch.float32))
        self.fixed_gate_value = fixed_gate_value
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model * 5, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.gate_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, zY, zS, zR):
        alignment = F.cosine_similarity(zY, zS, dim=-1, eps=1e-8)
        if self.fixed_gate_value is not None:
            gate = torch.full_like(alignment, float(self.fixed_gate_value))
            p_inc = 1.0 - gate
            return alignment, p_inc, gate
        gate_input = torch.cat([zY, zS, zR, zY * zS, torch.abs(zY - zS)], dim=-1)
        raw_gate = torch.sigmoid(self.gate_mlp(gate_input).squeeze(-1))
        alignment_gate = torch.sigmoid(self.alpha * alignment)
        gate = 0.5 * raw_gate + 0.5 * alignment_gate
        p_inc = 1.0 - gate
        return alignment, p_inc, gate


class RatingPredictor(nn.Module):
    def __init__(self, d_model=128, dropout=0.1, eta=0.1, shared_scale=1.0, residual_scale=1.0):
        super().__init__()
        self.eta = float(eta)
        self.shared_scale = float(shared_scale)
        self.residual_scale = float(residual_scale)
        self.fY = self._build_head(d_model=d_model, dropout=dropout)
        self.fS = self._build_head(d_model=d_model, dropout=dropout)
        self.fR = self._build_head(d_model=d_model, dropout=dropout)
        self._init_weights()

    @staticmethod
    def _build_head(d_model, dropout):
        return nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def _init_weights(self):
        for head in [self.fY, self.fS, self.fR]:
            for module in head:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, zY, zS, zR, gate):
        yY = self.fY(zY).squeeze(-1)
        yS = self.fS(zS).squeeze(-1)
        yR = self.fR(zR).squeeze(-1)
        pred = yY + self.shared_scale * gate * yS + self.residual_scale * (1.0 - gate) * self.eta * yR
        return pred, yY, yS, yR


class IARDLossComputer(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.lambda_rating = float(configs.get("lambda_rating", 1.0))
        self.lambda_align = float(configs.get("lambda_align", 0.1))
        self.lambda_sep = float(configs.get("lambda_sep", 0.05))
        self.lambda_recon = float(configs.get("lambda_recon", 0.1))
        self.lambda_gate = float(configs.get("lambda_gate", 0.001))
        self.lambda_proto = float(configs.get("lambda_proto", 0.01))
        self.tau_c = float(configs.get("tau_c", 0.2))
        self.eps = float(configs.get("eps", 1e-8))
        self.detach_gate_for_align = bool(configs.get("detach_gate_for_align", True))
        self.detach_zx_for_recon = bool(configs.get("detach_zx_for_recon", False))

    def forward(self, output, ratings, prototypes):
        pred = output["pred"]
        zY = output["zY"]
        zX = output["zX"]
        zS = output["zS"]
        zR = output["zR"]
        gate = output["gate"]
        recon_zX = output["recon_zX"]

        rating_loss = F.mse_loss(pred, ratings)

        zY_norm = F.normalize(zY, dim=-1, eps=self.eps)
        zS_norm = F.normalize(zS, dim=-1, eps=self.eps)
        logits = zY_norm @ zS_norm.T
        logits = logits / self.tau_c
        labels = torch.arange(logits.size(0), device=logits.device)
        per_sample_nce = F.cross_entropy(logits, labels, reduction="none")
        gate_weight = gate.detach() if self.detach_gate_for_align else gate
        align_loss = torch.mean(gate_weight * per_sample_nce)

        normalized_zR = F.normalize(zR, dim=-1, eps=self.eps)
        normalized_zY = F.normalize(zY, dim=-1, eps=self.eps)
        normalized_zS = F.normalize(zS, dim=-1, eps=self.eps)
        y_r_cos = torch.sum(normalized_zY * normalized_zR, dim=-1)
        s_r_cos = torch.sum(normalized_zS * normalized_zR, dim=-1)
        sep_loss = torch.mean(y_r_cos.square()) + torch.mean(s_r_cos.square())

        recon_target = zX.detach() if self.detach_zx_for_recon else zX
        recon_loss = F.mse_loss(recon_zX, recon_target)

        entropy = -gate * torch.log(gate + self.eps) - (1.0 - gate) * torch.log(1.0 - gate + self.eps)
        gate_loss = -torch.mean(entropy)

        normalized_prototypes = F.normalize(prototypes, dim=-1, eps=self.eps)
        gram = normalized_prototypes @ normalized_prototypes.T
        identity = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
        proto_loss = torch.mean((gram - identity).square())

        loss = (
            self.lambda_rating * rating_loss
            + self.lambda_align * align_loss
            + self.lambda_sep * sep_loss
            + self.lambda_recon * recon_loss
            + self.lambda_gate * gate_loss
            + self.lambda_proto * proto_loss
        )

        return {
            "loss": loss,
            "rating_loss": rating_loss,
            "align_loss": align_loss,
            "sep_loss": sep_loss,
            "recon_loss": recon_loss,
            "gate_loss": gate_loss,
            "proto_loss": proto_loss,
        }


class IARDRM(AbstractRec):
    def __init__(self, configs, train_dataset):
        super().__init__()
        self.configs = configs
        self.num_users = int(train_dataset.num_users)
        self.num_items = int(train_dataset.num_items)
        self.d_id = int(configs.get("d_id", 64))
        self.d_text = int(configs.get("d_text", 768))
        self.d_model = int(configs.get("d_model", 128))
        self.num_layers = int(configs.get("num_layers", 2))
        self.num_intents = int(configs.get("num_intents", 8))
        self.eta = float(configs.get("eta", 0.1))
        self.dropout = float(configs.get("dropout", 0.1))
        self.tau_p = float(configs.get("tau_p", 0.2))
        self.gate_alpha = float(configs.get("gate_alpha", 5.0))
        self.shared_fusion_scale = float(configs.get("shared_fusion_scale", 1.0))
        self.residual_fusion_scale = float(configs.get("residual_fusion_scale", 1.0))
        self.fixed_gate_value = configs.get("fixed_gate_value")
        self.disentangler_mode = configs.get("disentangler_mode", "conditioned")
        self.history_encoder = str(configs.get("history_encoder", "mean")).lower()
        if self.history_encoder not in {"mean", "attention"}:
            raise ValueError("history_encoder must be either 'mean' or 'attention'.")
        self.review_emb_dim = int(configs.get("review_emb_dim", self.d_text // 4 if self.d_text % 4 == 0 else self.d_text))

        self.user_bias = nn.Embedding(self.num_users, 1)
        self.item_bias = nn.Embedding(self.num_items, 1)
        self.global_bias = nn.Parameter(torch.tensor(self._get_initial_global_bias(train_dataset), dtype=torch.float32))
        self._init_bias_terms()

        self.rating_encoder = RatingGraphEncoder(
            num_users=self.num_users,
            num_items=self.num_items,
            d_id=self.d_id,
            d_model=self.d_model,
            num_layers=self.num_layers,
        )
        self.review_encoder = ReviewProjectionEncoder(
            d_text=self.d_text,
            d_model=self.d_model,
            dropout=self.dropout,
        )
        self.history_attention_encoder = HistoryAttentionEncoder(
            raw_dim=self.review_emb_dim,
            d_model=self.d_model,
        )
        self.intent_extractor = PrototypeIntentExtractor(
            d_model=self.d_model,
            num_intents=self.num_intents,
            tau_p=self.tau_p,
        )
        self.disentangler = SharedResidualDisentangler(
            d_model=self.d_model,
            dropout=self.dropout,
            disentangler_mode=self.disentangler_mode,
        )
        self.gate = InconsistencyGate(
            d_model=self.d_model,
            dropout=self.dropout,
            gate_alpha=self.gate_alpha,
            fixed_gate_value=self.fixed_gate_value,
        )
        self.predictor = RatingPredictor(
            d_model=self.d_model,
            dropout=self.dropout,
            eta=self.eta,
            shared_scale=self.shared_fusion_scale,
            residual_scale=self.residual_fusion_scale,
        )
        self.loss_computer = IARDLossComputer(configs)

    @staticmethod
    def _get_initial_global_bias(train_dataset):
        ratings = getattr(train_dataset, "ratings", None)
        if ratings is None:
            return 0.0
        if torch.is_tensor(ratings):
            if ratings.numel() == 0:
                return 0.0
            return float(ratings.float().mean().item())
        try:
            ratings_tensor = torch.as_tensor(ratings, dtype=torch.float32)
        except (TypeError, ValueError):
            return 0.0
        if ratings_tensor.numel() == 0:
            return 0.0
        return float(ratings_tensor.mean().item())

    def _init_bias_terms(self):
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, *args, **kwargs):
        if args:
            user_ids, item_ids, review_emb, edge_index = args[:4]
            edge_weight = args[4] if len(args) > 4 else None
            user_history_emb = None
            user_history_mask = None
            item_history_emb = None
            item_history_mask = None
        else:
            user_ids = kwargs["user_ids"]
            item_ids = kwargs["item_ids"]
            review_emb = kwargs["review_emb"]
            edge_index = kwargs["edge_index"]
            edge_weight = kwargs.get("edge_weight")
            user_history_emb = kwargs.get("user_history_emb")
            user_history_mask = kwargs.get("user_history_mask")
            item_history_emb = kwargs.get("item_history_emb")
            item_history_mask = kwargs.get("item_history_mask")
        zY = self.rating_encoder(
            user_ids=user_ids,
            item_ids=item_ids,
            edge_index=edge_index,
            edge_weight=edge_weight,
        )
        if self.history_encoder == "attention":
            if user_history_emb is None or user_history_mask is None or item_history_emb is None or item_history_mask is None:
                raise ValueError("history_encoder='attention' requires user/item history embeddings and masks.")
            review_emb = self.history_attention_encoder(
                query=zY,
                user_history_emb=user_history_emb.to(zY.device),
                user_history_mask=user_history_mask.to(zY.device),
                item_history_emb=item_history_emb.to(zY.device),
                item_history_mask=item_history_mask.to(zY.device),
            )
        hX = self.review_encoder(review_emb)
        zX, intent_weights = self.intent_extractor(hX)
        zS, zR, recon_zX = self.disentangler(zX, zY)
        alignment, p_inc, gate = self.gate(zY, zS, zR)
        pred, yY, yS, yR = self.predictor(zY, zS, zR, gate)
        pred = pred + self.user_bias(user_ids).squeeze(-1) + self.item_bias(item_ids).squeeze(-1) + self.global_bias
        return {
            "pred": pred,
            "zY": zY,
            "hX": hX,
            "zX": zX,
            "zS": zS,
            "zR": zR,
            "intent_weights": intent_weights,
            "alignment": alignment,
            "p_inc": p_inc,
            "gate": gate,
            "yY": yY,
            "yS": yS,
            "yR": yR,
            "recon_zX": recon_zX,
        }

    def cal_loss(self, *args, **kwargs):
        batch_data = args[0] if args else kwargs["batch_data"]
        output = self.forward(
            user_ids=batch_data["user_ids"],
            item_ids=batch_data["item_ids"],
            review_emb=batch_data["review_emb"],
            edge_index=batch_data["edge_index"],
            edge_weight=batch_data.get("edge_weight"),
            user_history_emb=batch_data.get("user_history_emb"),
            user_history_mask=batch_data.get("user_history_mask"),
            item_history_emb=batch_data.get("item_history_emb"),
            item_history_mask=batch_data.get("item_history_mask"),
        )
        loss_dict = self.loss_computer(
            output=output,
            ratings=batch_data["ratings"],
            prototypes=self.intent_extractor.prototypes,
        )
        loss = loss_dict["loss"]
        numeric = {key: float(value.detach().item()) for key, value in loss_dict.items()}
        numeric["total_loss"] = numeric["loss"]
        return loss, numeric

    def predict_scores(self, *args, **kwargs):
        user_ids = args[0] if args else kwargs["user_ids"]
        return torch.zeros((user_ids.size(0), self.num_items), device=user_ids.device)


def run_iard_rm_sanity_check():
    class _DummyDataset:
        num_users = 10
        num_items = 20

    torch.manual_seed(42)
    configs = {
        "d_id": 64,
        "d_text": 768,
        "d_model": 128,
        "num_layers": 2,
        "num_intents": 8,
        "eta": 0.1,
        "dropout": 0.1,
        "tau_p": 0.2,
        "gate_alpha": 5.0,
        "lambda_rating": 1.0,
        "lambda_align": 0.1,
        "lambda_sep": 0.05,
        "lambda_recon": 0.1,
        "lambda_gate": 0.001,
        "lambda_proto": 0.01,
        "tau_c": 0.2,
        "detach_gate_for_align": True,
        "detach_zx_for_recon": False,
    }
    model = IARDRM(configs, _DummyDataset())
    batch_size = 4
    user_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    item_ids = torch.tensor([4, 5, 6, 7], dtype=torch.long)
    ratings = torch.tensor([4.0, 3.5, 2.0, 5.0], dtype=torch.float32)
    review_emb = torch.randn(batch_size, 768, dtype=torch.float32)
    edge_index = torch.tensor(
        [
            [0, 10 + 4, 1, 10 + 5, 2, 10 + 6, 3, 10 + 7],
            [10 + 4, 0, 10 + 5, 1, 10 + 6, 2, 10 + 7, 3],
        ],
        dtype=torch.long,
    )
    output = model(
        user_ids=user_ids,
        item_ids=item_ids,
        review_emb=review_emb,
        edge_index=edge_index,
        edge_weight=None,
    )
    expected_keys = {
        "pred",
        "zY",
        "hX",
        "zX",
        "zS",
        "zR",
        "intent_weights",
        "alignment",
        "p_inc",
        "gate",
        "yY",
        "yS",
        "yR",
        "recon_zX",
    }
    if set(output.keys()) != expected_keys:
        raise AssertionError(f"Unexpected output keys: {output.keys()}")
    if output["pred"].shape != (batch_size,):
        raise AssertionError(f"pred shape mismatch: {output['pred'].shape}")
    for key in ["zY", "zS", "zR"]:
        if output[key].shape != (batch_size, 128):
            raise AssertionError(f"{key} shape mismatch: {output[key].shape}")
    loss_dict = model.loss_computer(output=output, ratings=ratings, prototypes=model.intent_extractor.prototypes)
    loss = loss_dict["loss"]
    loss.backward()
    if torch.isnan(loss):
        raise AssertionError("Loss is NaN")
    for tensor_name, tensor_value in output.items():
        if torch.is_tensor(tensor_value) and torch.isnan(tensor_value).any():
            raise AssertionError(f"{tensor_name} contains NaN")
    attention_configs = dict(configs)
    attention_configs.update(
        {
            "d_text": 768 * 4,
            "review_emb_dim": 768,
            "history_encoder": "attention",
            "history_top_k": 3,
        }
    )
    attention_model = IARDRM(attention_configs, _DummyDataset())
    user_history_emb = torch.randn(batch_size, 3, 768, dtype=torch.float32)
    item_history_emb = torch.randn(batch_size, 3, 768, dtype=torch.float32)
    user_history_mask = torch.tensor(
        [[True, True, False], [False, False, False], [True, False, False], [True, True, True]],
        dtype=torch.bool,
    )
    item_history_mask = torch.tensor(
        [[True, False, False], [True, True, False], [False, False, False], [True, True, True]],
        dtype=torch.bool,
    )
    attention_output = attention_model(
        user_ids=user_ids,
        item_ids=item_ids,
        review_emb=torch.zeros(batch_size, 768 * 4, dtype=torch.float32),
        edge_index=edge_index,
        edge_weight=None,
        user_history_emb=user_history_emb,
        user_history_mask=user_history_mask,
        item_history_emb=item_history_emb,
        item_history_mask=item_history_mask,
    )
    if attention_output["pred"].shape != (batch_size,):
        raise AssertionError(f"attention pred shape mismatch: {attention_output['pred'].shape}")
    for tensor_name, tensor_value in attention_output.items():
        if torch.is_tensor(tensor_value) and torch.isnan(tensor_value).any():
            raise AssertionError(f"attention {tensor_name} contains NaN")
    return {
        "loss": float(loss.detach().item()),
        "pred_shape": tuple(output["pred"].shape),
        "attention_pred_shape": tuple(attention_output["pred"].shape),
        "zY_shape": tuple(output["zY"].shape),
        "zS_shape": tuple(output["zS"].shape),
        "zR_shape": tuple(output["zR"].shape),
    }
