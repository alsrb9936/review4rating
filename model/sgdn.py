from typing import Any, Dict, List, Mapping, Tuple
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from .abstract import AbstractRec

try:
    import dgl
    import dgl.function as fn
    import dgl.nn.pytorch as dglnn
    dgl_import_error = None
except Exception as exc:  # pragma: no cover - dependency is validated at runtime.
    dgl = None
    fn = None
    dglnn = None
    dgl_import_error = exc


class GCMCGraphConv(nn.Module):
    """Original SGDN relation module: node + review messages weighted by edge ``w``."""

    def __init__(self, in_feats: int, out_feats: int, review_dim: int, dropout: float):
        super().__init__()
        self.node_w = nn.Linear(in_feats, out_feats, bias=False)
        self.review_w = nn.Linear(review_dim, out_feats, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph, feat):
        etype = graph.canonical_etypes[0][1]
        rating = etype[-1]
        src_feat = feat[0][f"h_{rating}"] if isinstance(feat, tuple) else feat[f"h_{rating}"]
        src_idx = graph.edata["src_id"].to(src_feat.device)
        dst_idx = graph.edata["dst_id"].to(src_feat.device)
        src_h = self.node_w(src_feat[src_idx])
        review_h = self.review_w(graph.edata["review_feat"].to(src_feat.device))
        weight = self.dropout(graph.edata["w"].to(src_feat.device))
        msg = (src_h + review_h) * weight
        out = msg.new_zeros((graph.num_dst_nodes(), msg.size(1)))
        out.index_add_(0, dst_idx, msg)
        return out


class GCMCLayer(nn.Module):
    """Factor-wise DGL heterograph convolution with SGDN edge weighting."""

    def __init__(
        self,
        rating_vals: List[int],
        in_feats: int,
        out_feats: int,
        review_dim: int,
        num_factors: int,
        factor_idx: int,
        dropout: float,
        aggregate: str,
        edge_temperature: float,
    ):
        super().__init__()
        if dglnn is None:
            raise ImportError("SGDN requires DGL. Install it with `pip install dgl`.")
        self._dglnn = dglnn
        self.rating_vals = rating_vals
        self.num_factors = num_factors
        self.k = factor_idx
        self.edge_temperature = edge_temperature
        sub_conv = {}
        for rating in rating_vals:
            key = str(rating)
            sub_conv[key] = GCMCGraphConv(in_feats, out_feats, review_dim, dropout)
            sub_conv[f"rev-{key}"] = GCMCGraphConv(in_feats, out_feats, review_dim, dropout)
        self.conv = self._dglnn.HeteroGraphConv(sub_conv, aggregate=aggregate)

    def _etype_weight(
        self,
        graph,
        etype: Tuple[str, str, str],
        feat_dic: Mapping[str, Mapping[str, torch.Tensor]],
        review_feat_dic: Mapping[str, torch.Tensor],
        prototypes: torch.Tensor,
        eta: torch.Tensor,
    ) -> torch.Tensor:
        src_type, rel, dst_type = etype
        rating = rel[-1]
        device = prototypes.device
        src = graph.edges[etype].data["src_id"].to(device)
        dst = graph.edges[etype].data["dst_id"].to(device)

        row_feat = F.normalize(feat_dic[src_type][f"h_{rating}"][src], dim=1)
        col_feat = F.normalize(feat_dic[dst_type][f"h_{rating}"][dst], dim=1)
        row_all = feat_dic[src_type][f"h_sum{rating}"][src]
        col_all = feat_dic[dst_type][f"h_sum{rating}"][dst]

        tau = self.edge_temperature
        sim_k = (row_feat * col_feat).sum(dim=1) / tau
        sim_all = (row_all * col_all).sum(dim=2) / tau
        exp_sim = torch.exp(sim_k) / torch.exp(sim_all).sum(dim=1).clamp_min(1e-8)

        rating_reviews = review_feat_dic[rel].to(device)
        review_feat_k = rating_reviews[:, self.k, :]
        anchor_dot_k = (review_feat_k * prototypes[self.k]).sum(dim=1) / tau
        anchor_dot_all = (rating_reviews * prototypes.unsqueeze(0)).sum(dim=2) / tau
        exp_anchor_dot_k = torch.exp(anchor_dot_k) / torch.exp(anchor_dot_all).sum(dim=1).clamp_min(1e-8)

        rating_idx = self.rating_vals.index(int(rating))
        gate = torch.sigmoid(eta[rating_idx])
        edge_factor_weight = gate * exp_anchor_dot_k + (1.0 - gate) * exp_sim

        src_norm = torch.zeros(graph.num_nodes(src_type), device=device, dtype=edge_factor_weight.dtype)
        dst_norm = torch.zeros(graph.num_nodes(dst_type), device=device, dtype=edge_factor_weight.dtype)
        src_norm.index_add_(0, src, edge_factor_weight)
        dst_norm.index_add_(0, dst, edge_factor_weight)
        n_ij = torch.sqrt(src_norm[src] * dst_norm[dst]).clamp_min(1e-8)
        return (edge_factor_weight / n_ij).unsqueeze(1)

    def forward(
        self,
        graph,
        feat_dic: Mapping[str, Mapping[str, torch.Tensor]],
        review_feat_dic: Mapping[str, torch.Tensor],
        prototypes: torch.Tensor,
        eta: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        for etype in graph.canonical_etypes:
            graph.edges[etype].data["w"] = self._etype_weight(graph, etype, feat_dic, review_feat_dic, prototypes, eta)
        out = self.conv(graph, feat_dic)
        return {ntype: self._collapse_aggregate(value) for ntype, value in out.items()}

    @staticmethod
    def _collapse_aggregate(value: torch.Tensor) -> torch.Tensor:
        if value.dim() == 3:
            return value.sum(dim=1)
        return value


class MLPPredictor(nn.Module):
    """SGDN decoder on the ``('user', 'rate', 'movie')`` graph."""

    def __init__(self, in_units: int, num_factors: int, classification: bool, num_classes: int, dropout: float):
        super().__init__()
        self.num_factors = num_factors
        self.classification = classification
        self.mlp = nn.Sequential(
            nn.Linear(in_units * 2, 64, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64, bias=False),
            nn.GELU(),
        )
        self.predictor = nn.Linear(64, num_classes if classification else 1, bias=True)

    def forward(self, graph, user_out: torch.Tensor, item_out: torch.Tensor):
        with graph.local_scope():
            graph.nodes["user"].data["h"] = user_out
            graph.nodes["movie"].data["h"] = item_out
            graph.apply_edges(self._apply_edges, etype="rate")
            h_fea = self.mlp(graph.edges["rate"].data["cat"])
            return self.predictor(h_fea), h_fea

    @staticmethod
    def _apply_edges(edges):
        return {"cat": torch.cat([edges.src["h"], edges.dst["h"]], dim=1)}


def cal_c_loss(
    h_fea1: torch.Tensor,
    h_fea2: torch.Tensor,
    int_dist: torch.Tensor,
    rating_split: List[int],
    num_pos: int,
    temperature: float,
    num_neg: int,
) -> torch.Tensor:
    z1 = F.normalize(h_fea1, dim=1)
    z2 = F.normalize(h_fea2, dim=1)
    int_dist = F.normalize(int_dist, dim=1)
    total_loss = h_fea1.new_tensor(0.0)
    count = 0
    start = 0

    for segment_size in rating_split:
        end = start + int(segment_size)
        if end > int_dist.size(0):
            break
        if end - start < 2:
            start = end
            continue

        seg_z1 = z1[start:end]
        seg_z2 = z2[start:end]
        seg_dist = int_dist[start:end]
        group_size = seg_z1.size(0)
        k_pos = min(max(1, num_pos), group_size)
        k_neg = min(max(1, num_neg), group_size)

        intent_sim = seg_dist @ seg_dist.T
        _, pos_idx = torch.topk(intent_sim, k=k_pos, dim=1)
        pos_score = torch.exp((seg_z1.unsqueeze(1) * seg_z2[pos_idx]).sum(dim=2) / temperature).sum(dim=1)
        neg_idx = torch.randperm(group_size, device=h_fea1.device)[:k_neg]
        ttl_score = torch.exp((seg_z1 @ seg_z2[neg_idx].T) / temperature).sum(dim=1)
        total_loss = total_loss + (-torch.log(pos_score / ttl_score.clamp_min(1e-8))).mean()
        count += 1
        start = end

    return total_loss / max(count, 1)


class SGDN(AbstractRec):
    """DGL port of Self-supervised Graph Disentangled Networks."""

    def __init__(self, configs, train_dataset):
        super().__init__()
        if dgl is None:
            raise ImportError(
                "SGDN requires a working DGL installation. Install a DGL build compatible with your PyTorch version "
                "(and GraphBolt binary, for DGL 2.x)."
            ) from dgl_import_error
        self._dgl = dgl
        self.configs = configs
        self.train_dataset = train_dataset
        self.num_users = int(train_dataset.num_users)
        self.num_items = int(train_dataset.num_items)
        self.review_dim = int(configs.get("review_dim", configs.get("review_feat_size", configs.get("bert_whitening_dim", 64))))
        self.num_factors = int(configs.get("num_factors", configs.get("num_factor", 2)))
        self.num_layers = int(configs.get("num_layers", configs.get("num_layer", 1)))
        self.hidden_dim = int(configs.get("hidden_dim", configs.get("gcn_out_units", self.review_dim)))
        self.gcn_out_units = int(configs.get("gcn_out_units", self.hidden_dim))
        if self.hidden_dim % self.num_factors != 0:
            raise ValueError("SGDN hidden_dim must be divisible by num_factors.")
        self.factor_dim = self.hidden_dim // self.num_factors
        self.dropout = float(configs.get("dropout", 0.8))
        self.gcn_dropout = float(configs.get("gcn_dropout", self.dropout))
        self.pred_dropout = float(configs.get("pred_dropout", 0.0))
        self.cl_weight = float(configs.get("cl_weight", configs.get("lamda", 0.005)))
        self.temperature = float(configs.get("temperature", 0.2))
        self.edge_temperature = float(configs.get("edge_temperature", 0.5))
        self.num_pos = int(configs.get("num_pos", 10))
        self.num_neg = int(configs.get("num_neg", 2048))
        self.use_contrastive = bool(configs.get("use_contrastive", True))
        self.classification = bool(configs.get("classification", configs.get("train_classification", False)))
        self.init_pred_bias_with_rating_mean = bool(configs.get("init_pred_bias_with_rating_mean", True))
        self.debug_shapes = bool(configs.get("debug_shapes", False))
        self.rating_vals = [int(v) for v in configs.get("rating_values", [1, 2, 3, 4, 5])]
        self._shape_logged = False

        self.ufeats = nn.ModuleDict({
            str(rating): nn.ModuleList([nn.Embedding(self.num_users, self.factor_dim) for _ in range(self.num_factors)])
            for rating in self.rating_vals
        })
        self.ifeats = nn.ModuleDict({
            str(rating): nn.ModuleList([nn.Embedding(self.num_items, self.factor_dim) for _ in range(self.num_factors)])
            for rating in self.rating_vals
        })
        self.rfcs = nn.ModuleList([nn.Linear(self.review_dim, self.review_dim) for _ in range(self.num_factors)])
        self.eta = nn.Parameter(torch.zeros(len(self.rating_vals)))
        self.prototypes = nn.Parameter(torch.empty(self.num_factors, self.review_dim))

        layers = []
        for layer_idx in range(self.num_layers):
            factor_layers = nn.ModuleList()
            for factor_idx in range(self.num_factors):
                factor_layers.append(
                    GCMCLayer(
                        rating_vals=self.rating_vals,
                        in_feats=self.factor_dim,
                        out_feats=self.factor_dim,
                        review_dim=self.review_dim,
                        num_factors=self.num_factors,
                        factor_idx=factor_idx,
                        dropout=self.gcn_dropout,
                        aggregate="sum" if layer_idx == self.num_layers - 1 else "stack",
                        edge_temperature=self.edge_temperature,
                    )
                )
            layers.append(factor_layers)
        self.encoder_layers = nn.ModuleList(layers)
        self.decoder = MLPPredictor(self.hidden_dim, self.num_factors, self.classification, len(self.rating_vals), self.pred_dropout)
        self.rating_loss_fn = nn.CrossEntropyLoss() if self.classification else nn.MSELoss()

        self._init_parameters()
        self._init_prototypes(train_dataset)
        if self.init_pred_bias_with_rating_mean:
            self._init_pred_bias_with_train_mean(train_dataset)

    def _init_parameters(self) -> None:
        for modules in (self.ufeats, self.ifeats):
            for rating in self.rating_vals:
                module_list = modules[str(rating)]
                if not isinstance(module_list, nn.ModuleList):
                    continue
                for emb in module_list:
                    if isinstance(emb, nn.Embedding):
                        nn.init.xavier_uniform_(emb.weight)
        for rfc in self.rfcs:
            if isinstance(rfc, nn.Linear):
                nn.init.xavier_uniform_(rfc.weight)
                if rfc.bias is not None:
                    nn.init.zeros_(rfc.bias)
        nn.init.xavier_uniform_(self.prototypes)

    def _init_prototypes(self, train_dataset) -> None:
        review_feat_dic = getattr(train_dataset, "review_feat_dic", None)
        if not isinstance(review_feat_dic, Mapping):
            return
        chunks = [feat for feat in review_feat_dic.values() if isinstance(feat, torch.Tensor) and feat.numel() > 0]
        if not chunks:
            return
        review_feat = torch.cat(chunks, dim=0).detach().cpu().float()
        num_clusters = min(self.num_factors, review_feat.size(0))
        try:
            faiss = importlib.import_module("faiss")

            kmeans = faiss.Kmeans(d=self.review_dim, k=num_clusters, gpu=False, seed=int(self.configs.get("seed", 42)))
            kmeans.train(review_feat.numpy())
            centroids = torch.as_tensor(kmeans.centroids, dtype=self.prototypes.dtype)
            source = "FAISS"
        except Exception:
            try:
                from sklearn.cluster import KMeans

                kmeans = KMeans(n_clusters=num_clusters, random_state=int(self.configs.get("seed", 42)), n_init="auto")
                kmeans.fit(review_feat.numpy())
                centroids = torch.as_tensor(kmeans.cluster_centers_, dtype=self.prototypes.dtype)
                source = "sklearn"
            except Exception as exc:
                print(f"SGDN prototype KMeans init skipped: {exc}")
                return
        if num_clusters < self.num_factors:
            centroids = torch.cat([centroids, self.prototypes.detach().cpu()[num_clusters:]], dim=0)
        with torch.no_grad():
            self.prototypes.copy_(F.normalize(centroids, dim=1))
        print(f"SGDN initialized review prototypes with {source}: {tuple(self.prototypes.shape)}")

    def _init_pred_bias_with_train_mean(self, train_dataset) -> None:
        if self.classification:
            return
        ratings = getattr(train_dataset, "ratings", None)
        if not isinstance(ratings, torch.Tensor) or ratings.numel() == 0 or self.decoder.predictor.bias is None:
            return
        rating_mean = float(ratings.float().mean().item())
        with torch.no_grad():
            self.decoder.predictor.bias.fill_(rating_mean)

    def _to_device(self, value, device):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if dgl is not None and isinstance(value, self._dgl.DGLHeteroGraph):
            return value.to(device)
        if isinstance(value, list):
            return [self._to_device(sub_value, device) for sub_value in value]
        if isinstance(value, dict):
            return {key: self._to_device(sub_value, device) for key, sub_value in value.items()}
        return value

    def _prepare_batch(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        device = next(self.parameters()).device
        prepared = self._to_device(batch_data, device)
        if not isinstance(prepared, dict):
            raise TypeError("SGDN expects batch data as a dictionary.")
        return prepared

    def _factorize_review_features(self, enc_graph) -> Dict[str, torch.Tensor]:
        review_dic_fact = {}
        for _, rel, _ in enc_graph.canonical_etypes:
            review_feat = enc_graph.edges[rel].data["review_feat"]
            projected = [rfc(review_feat).unsqueeze(1) for rfc in self.rfcs]
            review_dic_fact[rel] = torch.cat(projected, dim=1)
        return review_dic_fact

    def _prepare_factor_features(self, factor_states: List[Dict[str, Dict[str, torch.Tensor]]], factor_idx: int) -> Dict[str, Dict[str, torch.Tensor]]:
        feat_dic = {"user": {}, "movie": {}}
        for rating in self.rating_vals:
            key = str(rating)
            user_stack = torch.stack([factor_states[k]["user"][key] for k in range(self.num_factors)], dim=1)
            item_stack = torch.stack([factor_states[k]["movie"][key] for k in range(self.num_factors)], dim=1)
            feat_dic["user"][f"h_{key}"] = factor_states[factor_idx]["user"][key]
            feat_dic["movie"][f"h_{key}"] = factor_states[factor_idx]["movie"][key]
            feat_dic["user"][f"h_sum{key}"] = F.normalize(user_stack, dim=2)
            feat_dic["movie"][f"h_sum{key}"] = F.normalize(item_stack, dim=2)
        return feat_dic

    def _rating_embedding(self, table: nn.ModuleDict, rating: int, factor_idx: int) -> torch.Tensor:
        module_list = table[str(rating)]
        if not isinstance(module_list, nn.ModuleList):
            raise TypeError("SGDN factor embedding table is malformed.")
        embedding = module_list[factor_idx]
        if not isinstance(embedding, nn.Embedding):
            raise TypeError("SGDN factor embedding entry is malformed.")
        return embedding.weight

    def _initial_factor_states(self) -> List[Dict[str, Dict[str, torch.Tensor]]]:
        states = []
        for factor_idx in range(self.num_factors):
            states.append({
                "user": {str(rating): self._rating_embedding(self.ufeats, rating, factor_idx) for rating in self.rating_vals},
                "movie": {str(rating): self._rating_embedding(self.ifeats, rating, factor_idx) for rating in self.rating_vals},
            })
        return states

    def _int_dist(self, dec_graph, review_feat_dic_fact: Mapping[str, torch.Tensor], user_out: torch.Tensor, item_out: torch.Tensor) -> torch.Tensor:
        review_feat = dec_graph.edges["rate"].data["review_feat"]
        review_all = torch.cat([rfc(review_feat).unsqueeze(1) for rfc in self.rfcs], dim=1)
        anchor_scores = (review_all * self.prototypes.unsqueeze(0)).sum(dim=2) / self.edge_temperature
        review_dist = F.softmax(anchor_scores, dim=1)
        src, dst = dec_graph.edges(etype="rate")
        user_factor = user_out[src].view(-1, self.num_factors, self.factor_dim)
        item_factor = item_out[dst].view(-1, self.num_factors, self.factor_dim)
        node_scores = F.cosine_similarity(user_factor, item_factor, dim=2) / self.edge_temperature
        node_dist = F.softmax(node_scores, dim=1)
        gate = torch.sigmoid(self.eta).mean()
        int_dist = gate * review_dist + (1.0 - gate) * node_dist
        return int_dist / int_dist.sum(dim=1, keepdim=True).clamp_min(1e-8)

    def encode(self, enc_graphs, dec_graph, review_feat_dic):
        assert len(enc_graphs) == self.num_factors, f"len(enc_graphs)={len(enc_graphs)} != num_factors={self.num_factors}"
        assert isinstance(dec_graph, self._dgl.DGLHeteroGraph)
        review_dic_fact = self._factorize_review_features(enc_graphs[0])
        factor_states = self._initial_factor_states()

        for layer_idx in range(len(self.encoder_layers)):
            factor_layers = self.encoder_layers[layer_idx]
            if not isinstance(factor_layers, nn.ModuleList):
                raise TypeError("SGDN encoder layer container is malformed.")
            next_states = []
            for factor_idx, layer in enumerate(factor_layers):
                if not isinstance(layer, GCMCLayer):
                    raise TypeError("SGDN encoder layer is malformed.")
                feat_dic = self._prepare_factor_features(factor_states, factor_idx)
                out = layer(enc_graphs[factor_idx], feat_dic, review_dic_fact, F.normalize(self.prototypes, dim=1), self.eta)
                next_states.append({
                    "user": {str(rating): out["user"] for rating in self.rating_vals},
                    "movie": {str(rating): out["movie"] for rating in self.rating_vals},
                })
            factor_states = next_states

        user_parts = []
        item_parts = []
        for factor_idx in range(self.num_factors):
            user_parts.append(torch.stack(list(factor_states[factor_idx]["user"].values()), dim=0).mean(dim=0))
            item_parts.append(torch.stack(list(factor_states[factor_idx]["movie"].values()), dim=0).mean(dim=0))
        user_out = torch.cat(user_parts, dim=1)
        item_out = torch.cat(item_parts, dim=1)
        int_dist = self._int_dist(dec_graph, review_dic_fact, user_out, item_out)
        return user_out, item_out, int_dist, review_dic_fact

    def _forward_once(self, batch: Dict[str, Any]):
        user_out, item_out, int_dist, review_dic_fact = self.encode(batch["enc_graphs"], batch["dec_graph"], batch["review_feat_dic"])
        pred_ratings, h_fea = self.decoder(batch["dec_graph"], user_out, item_out)
        self._assert_and_log_shapes(batch, pred_ratings, int_dist, review_dic_fact)
        return pred_ratings, h_fea, int_dist

    def _assert_and_log_shapes(self, batch, pred_ratings, int_dist, review_dic_fact) -> None:
        assert int_dist.shape == (batch["ratings"].numel(), self.num_factors), tuple(int_dist.shape)
        assert pred_ratings.shape[0] == batch["ratings"].shape[0], (tuple(pred_ratings.shape), tuple(batch["ratings"].shape))
        for rating in self.rating_vals:
            key = str(rating)
            expected_edges = batch["enc_graphs"][0].num_edges(key)
            assert review_dic_fact[key].shape == (expected_edges, self.num_factors, self.review_dim), tuple(review_dic_fact[key].shape)
        if not torch.isfinite(pred_ratings).all():
            raise FloatingPointError("SGDN prediction contains NaN/Inf.")
        if self.debug_shapes and not self._shape_logged:
            print("SGDN DGL shape check:", {
                "enc_graphs": len(batch["enc_graphs"]),
                "dec_edges": int(batch["dec_graph"].num_edges("rate")),
                "int_dist": tuple(int_dist.shape),
                "pred": tuple(pred_ratings.shape),
                "review_factor_shapes": {k: tuple(v.shape) for k, v in review_dic_fact.items()},
            })
            self._shape_logged = True

    def _rating_loss(self, pred_ratings: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        if self.classification:
            return self.rating_loss_fn(pred_ratings, batch["labels"])
        return self.rating_loss_fn(pred_ratings.squeeze(-1), batch["ratings"])

    def cal_loss(self, *args, **kwargs):
        batch_data = args[0] if args else kwargs["batch_data"]
        batch = self._prepare_batch(batch_data)
        pred_ratings1, h_fea1, int_dist1 = self._forward_once(batch)
        rating_loss = self._rating_loss(pred_ratings1, batch)

        cl_loss = rating_loss.new_tensor(0.0)
        if self.use_contrastive and self.cl_weight > 0:
            pred_ratings2, h_fea2, _ = self._forward_once(batch)
            rating_loss = (rating_loss + self._rating_loss(pred_ratings2, batch)) / 2.0
            cl_loss = cal_c_loss(h_fea1, h_fea2, int_dist1, batch["rating_split"], self.num_pos, self.temperature, self.num_neg)

        total_loss = rating_loss + self.cl_weight * cl_loss
        if not torch.isfinite(total_loss):
            raise FloatingPointError("SGDN loss contains NaN/Inf.")

        pred_for_diag = self._ratings_from_logits(pred_ratings1)
        loss_dict = {
            "total_loss": float(total_loss.detach().item()),
            "rating_loss": float(rating_loss.detach().item()),
            "cl_loss": float(cl_loss.detach().item()),
            "pred_min": float(pred_for_diag.detach().min().item()),
            "pred_max": float(pred_for_diag.detach().max().item()),
            "pred_mean": float(pred_for_diag.detach().mean().item()),
            "pred_std": float(pred_for_diag.detach().std(unbiased=False).item()),
        }
        return total_loss, loss_dict

    def _ratings_from_logits(self, pred_ratings: torch.Tensor) -> torch.Tensor:
        if self.classification:
            probs = F.softmax(pred_ratings, dim=1)
            rating_vals = pred_ratings.new_tensor(self.rating_vals, dtype=torch.float32)
            return (probs * rating_vals.view(1, -1)).sum(dim=1)
        return pred_ratings.squeeze(-1)

    def predict_ratings(self, batch_data):
        batch = self._prepare_batch(batch_data)
        pred_ratings, _, _ = self._forward_once(batch)
        return self._ratings_from_logits(pred_ratings)

    def predict_scores(self, *args, **kwargs):
        raise NotImplementedError("SGDN is a full-graph rating predictor; use predict_ratings(batch_data) for decoder edges.")
