from typing import Any, Dict, List, Mapping, Optional, Tuple
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

    def forward(
        self,
        graph,
        feat,
        weight: torch.Tensor,
        review_feat: Optional[torch.Tensor] = None,
        rating_key: Optional[str] = None,
        src_type: Optional[str] = None,
    ):
        etype = graph.canonical_etypes[0][1]
        rating = rating_key if rating_key is not None else etype[-1]
        if src_type is not None:
            src_feat = feat[src_type][f"h_{rating}"]
        else:
            src_feat = feat[0][f"h_{rating}"] if isinstance(feat, tuple) else feat[f"h_{rating}"]
        device = src_feat.device
        src, dst = graph.edges(etype=etype)
        src = src.to(device).long()
        dst = dst.to(device).long()
        src_h = self.node_w(src_feat[src])
        if review_feat is not None:
            msg = (src_h + self.review_w(review_feat.to(device))) * self.dropout(weight.to(device))
        else:
            msg = src_h * self.dropout(weight.to(device))
        out = msg.new_zeros((graph.num_dst_nodes(), msg.size(1)))
        out.index_add_(0, dst, msg)
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
        num_edges_for_eta: int,
    ):
        super().__init__()
        if dglnn is None:
            raise ImportError("SGDN requires DGL. Install it with `pip install dgl`.")
        self._dglnn = dglnn
        self.rating_vals = rating_vals
        self.num_factors = num_factors
        self.k = factor_idx
        self.edge_temperature = edge_temperature
        self.num_edges_for_eta = max(1, int(num_edges_for_eta))
        self.eta = nn.Parameter(torch.zeros(max(rating_vals), self.num_edges_for_eta))
        self.ufc = nn.Linear(out_feats, out_feats)
        self.ifc = nn.Linear(out_feats, out_feats)
        self.agg_act = nn.LeakyReLU(0.1)
        self.output_dropout = nn.Dropout(dropout)
        sub_conv = {}
        for rating in rating_vals:
            key = str(rating)
            sub_conv[key] = GCMCGraphConv(in_feats, out_feats, review_dim, dropout)
            sub_conv[f"rev-{key}"] = GCMCGraphConv(in_feats, out_feats, review_dim, dropout)
        self.sub_conv = nn.ModuleDict(sub_conv)
        self.aggregate = aggregate

    def _etype_signal(
        self,
        graph,
        etype: Tuple[str, str, str],
        feat_dic: Mapping[str, Mapping[str, torch.Tensor]],
        review_feat_dic: Mapping[str, torch.Tensor],
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        src_type, rel, dst_type = etype
        rating = rel[-1]
        device = prototypes.device
        src, dst = graph.edges(etype=etype)
        src = src.to(device).long()
        dst = dst.to(device).long()

        row_feat = F.normalize(feat_dic[src_type][f"h_{rating}"][src], dim=1)
        col_feat = F.normalize(feat_dic[dst_type][f"h_{rating}"][dst], dim=1)
        row_all = feat_dic[src_type][f"h_sum{rating}"][src]
        col_all = feat_dic[dst_type][f"h_sum{rating}"][dst]

        tau = self.edge_temperature
        sim_k = (row_feat * col_feat).sum(dim=1) / tau
        sim_all = (row_all * col_all).sum(dim=2) / tau
        exp_sim = torch.exp(sim_k) / torch.exp(sim_all).sum(dim=1).clamp_min(1e-8)

        if rating in review_feat_dic:
            rating_reviews = review_feat_dic[rating].to(device)
            review_feat_k = rating_reviews[:, self.k, :]
            anchor_dot_k = (review_feat_k * prototypes[self.k]).sum(dim=1) / tau
            anchor_dot_all = (rating_reviews * prototypes.unsqueeze(0)).sum(dim=2) / tau
            exp_anchor_dot_k = torch.exp(anchor_dot_k) / torch.exp(anchor_dot_all).sum(dim=1).clamp_min(1e-8)
            if exp_anchor_dot_k.numel() > self.eta.size(1):
                raise ValueError(
                    f"SGDN eta has {self.eta.size(1)} columns but relation {rel!r} has "
                    f"{exp_anchor_dot_k.numel()} edges. Increase num_edges_for_eta."
                )
            rating_row = max(0, int(rating) - 1)
            gate = torch.sigmoid(self.eta[rating_row, : exp_anchor_dot_k.shape[0]]).to(device)
            edge_factor_weight = gate * exp_anchor_dot_k + (1.0 - gate) * exp_sim
        else:
            edge_factor_weight = exp_sim

        return edge_factor_weight

    def forward(
        self,
        graph,
        feat_dic: Mapping[str, Mapping[str, torch.Tensor]],
        review_feat_dic: Mapping[str, torch.Tensor],
        prototypes: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        with graph.local_scope():
            device = prototypes.device
            norm_user_sum = torch.zeros(graph.num_nodes("user"), device=device)
            norm_movie_sum = torch.zeros(graph.num_nodes("movie"), device=device)
            edge_signals: Dict[Tuple[str, str, str], torch.Tensor] = {}

            for etype in graph.canonical_etypes:
                src_type, _, dst_type = etype
                edge_signal = self._etype_signal(graph, etype, feat_dic, review_feat_dic, prototypes)
                edge_signals[etype] = edge_signal
                src, dst = graph.edges(etype=etype)
                src = src.to(device).long()
                dst = dst.to(device).long()
                if src_type == "movie":
                    norm_movie_sum.index_add_(0, src, edge_signal)
                    norm_user_sum.index_add_(0, dst, edge_signal)
                else:
                    norm_user_sum.index_add_(0, src, edge_signal)
                    norm_movie_sum.index_add_(0, dst, edge_signal)

            norm_user_sum = norm_user_sum / 2.0
            norm_movie_sum = norm_movie_sum / 2.0
            weights: Dict[Tuple[str, str, str], torch.Tensor] = {}
            review_by_etype: Dict[Tuple[str, str, str], torch.Tensor] = {}
            reverse_weights = []
            for rating in sorted(self.rating_vals, reverse=True):
                reverse_etype = ("movie", f"rev-{rating}", "user")
                if reverse_etype not in edge_signals:
                    continue
                src, dst = graph.edges(etype=reverse_etype)
                src = src.to(device).long()
                dst = dst.to(device).long()
                edge_signal = edge_signals[reverse_etype]
                n_ij = torch.sqrt(norm_movie_sum[src] * norm_user_sum[dst]).clamp_min(1e-8)
                weight = (edge_signal / n_ij).unsqueeze(1)
                weights[reverse_etype] = weight
                key = str(rating)
                if key in review_feat_dic:
                    review_by_etype[reverse_etype] = review_feat_dic[key].to(device)[:, self.k, :]
                reverse_weights.append(weight)

            for rating in self.rating_vals:
                forward_etype = ("user", str(rating), "movie")
                if forward_etype not in edge_signals:
                    continue
                src, dst = graph.edges(etype=forward_etype)
                src = src.to(device).long()
                dst = dst.to(device).long()
                edge_signal = edge_signals[forward_etype]
                n_ij = torch.sqrt(norm_user_sum[src] * norm_movie_sum[dst]).clamp_min(1e-8)
                weights[forward_etype] = (edge_signal / n_ij).unsqueeze(1)
                key = str(rating)
                if key in review_feat_dic:
                    review_by_etype[forward_etype] = review_feat_dic[key].to(device)[:, self.k, :]

            out = self._manual_conv(graph, feat_dic, weights, review_by_etype)
            out = {ntype: self._post_process(ntype, value) for ntype, value in out.items()}
            if reverse_weights:
                int_dist = torch.cat(reverse_weights, dim=0)
            else:
                int_dist = prototypes.new_zeros((0, 1))
            return out, int_dist

    def _manual_conv(
        self,
        graph,
        feat_dic: Mapping[str, Mapping[str, torch.Tensor]],
        weights: Mapping[Tuple[str, str, str], torch.Tensor],
        review_by_etype: Mapping[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        per_node_type: Dict[str, List[torch.Tensor]] = {"user": [], "movie": []}
        for rating in self.rating_vals:
            for etype in (("user", str(rating), "movie"), ("movie", f"rev-{rating}", "user")):
                if etype not in weights:
                    continue
                src_type, rel, dst_type = etype
                subgraph = graph[etype]
                conv = self.sub_conv[rel]
                out = conv(subgraph, feat_dic, weights[etype], review_by_etype.get(etype), str(rating), src_type)
                per_node_type[dst_type].append(out)

        result = {}
        for ntype, outputs in per_node_type.items():
            if not outputs:
                sample = next(iter(next(iter(feat_dic.values())).values()))
                result[ntype] = sample.new_zeros((graph.num_nodes(ntype), sample.size(1)))
            elif self.aggregate == "stack":
                result[ntype] = torch.stack(outputs, dim=1)
            else:
                result[ntype] = torch.stack(outputs, dim=0).sum(dim=0)
        return result

    def _post_process(self, ntype: str, value: torch.Tensor) -> torch.Tensor:
        value = self.agg_act(value)
        value = self.output_dropout(value)
        return self.ufc(value) if ntype == "user" else self.ifc(value)


class MLPPredictor(nn.Module):
    """SGDN decoder on the ``('user', 'rate', 'movie')`` graph."""

    def __init__(self, in_units: int, num_factors: int, classification: bool, num_classes: int, dropout: float):
        super().__init__()
        self.num_factors = num_factors
        self.classification = classification
        self.mlp = nn.Sequential(
            nn.Linear(in_units * 2, 64, bias=False),
            nn.GELU(),
            nn.Linear(64, 64, bias=False),
        )
        self.dropout = nn.Dropout(dropout)
        self.predictor = nn.Linear(64, num_classes if classification else 1, bias=False)

    def forward(self, graph, user_out: torch.Tensor, item_out: torch.Tensor):
        src, dst = graph.edges(etype="rate")
        device = user_out.device
        src = src.to(device).long()
        dst = dst.to(device).long()
        h_fea = self.dropout(self.mlp(torch.cat([user_out[src], item_out[dst]], dim=1)))
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
        self.review_dim = int(getattr(train_dataset, "review_dim", configs.get("review_dim", configs.get("review_feat_size", configs.get("bert_whitening_dim", 64)))))
        for dim_key in ("review_dim", "review_feat_size", "bert_whitening_dim"):
            if dim_key in configs and int(configs.get(dim_key)) != self.review_dim:
                print(f"[SGDN] Sync {dim_key}={configs.get(dim_key)} -> actual review_dim={self.review_dim}")
            configs[dim_key] = self.review_dim
        self.num_factors = int(configs.get("num_factors", configs.get("num_factor", 2)))
        self.num_layers = int(configs.get("num_layers", configs.get("num_layer", 1)))
        if bool(configs.get("match_original_sgdn_dims", True)):
            self.hidden_dim = self.review_dim
            self.gcn_out_units = self.review_dim
            configs["hidden_dim"] = self.hidden_dim
            configs["gcn_out_units"] = self.gcn_out_units
            configs["gcn_agg_units"] = self.review_dim
        else:
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
        self.init_pred_bias_with_rating_mean = bool(configs.get("init_pred_bias_with_rating_mean", False))
        self.debug_shapes = bool(configs.get("debug_shapes", False))
        self.rating_vals = [int(v) for v in configs.get("rating_values", [1, 2, 3, 4, 5])]
        self.num_edges_for_eta = int(getattr(train_dataset, "num_train_edges", getattr(train_dataset, "ratings", torch.empty(0)).numel()))
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
                        num_edges_for_eta=self.num_edges_for_eta,
                    )
                )
            layers.append(factor_layers)
        self.encoder_layers = nn.ModuleList(layers)
        self.decoder = MLPPredictor(self.hidden_dim, self.num_factors, self.classification, len(self.rating_vals), self.pred_dropout)
        self.rating_loss_fn = nn.CrossEntropyLoss() if self.classification else nn.MSELoss()

        self._init_parameters()
        # Original SGDN calls reset_parameters() after KMeans prototype init, which overwrites centroids.
        # This port intentionally initializes learnable weights first and then applies KMeans so the
        # prototype prior survives model construction.
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
        if self.classification or self.decoder.predictor.bias is None:
            return
        ratings = getattr(train_dataset, "ratings", None)
        if not isinstance(ratings, torch.Tensor) or ratings.numel() == 0:
            return
        rating_mean = float(ratings.float().mean().item())
        with torch.no_grad():
            self.decoder.predictor.bias.fill_(rating_mean)

    def _to_device(self, value, device):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if dgl is not None and isinstance(value, self._dgl.DGLHeteroGraph):
            # Keep SGDN graph structure on CPU and move only edge index tensors/features as needed.
            # DGL 2.x can hit illegal-memory-access failures for this full heterograph on specific
            # CUDA devices (observed on cuda:3), while CPU graph metadata + GPU tensors is stable.
            return value
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

    def _factorize_review_features(self, review_feat_dic: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        review_dic_fact = {}
        for rating in self.rating_vals:
            key = str(rating)
            if key not in review_feat_dic:
                continue
            review_feat = review_feat_dic[key]
            projected = [rfc(review_feat).unsqueeze(1) for rfc in self.rfcs]
            review_dic_fact[key] = torch.cat(projected, dim=1)
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

    def encode(self, enc_graphs, dec_graph, review_feat_dic):
        assert len(enc_graphs) == self.num_factors, f"len(enc_graphs)={len(enc_graphs)} != num_factors={self.num_factors}"
        assert isinstance(dec_graph, self._dgl.DGLHeteroGraph)
        review_dic_fact = self._factorize_review_features(review_feat_dic)
        factor_states = self._initial_factor_states()
        int_dist_parts = []

        for layer_idx in range(len(self.encoder_layers)):
            factor_layers = self.encoder_layers[layer_idx]
            if not isinstance(factor_layers, nn.ModuleList):
                raise TypeError("SGDN encoder layer container is malformed.")
            next_states = []
            for factor_idx, layer in enumerate(factor_layers):
                if not isinstance(layer, GCMCLayer):
                    raise TypeError("SGDN encoder layer is malformed.")
                feat_dic = self._prepare_factor_features(factor_states, factor_idx)
                out, int_dist = layer(enc_graphs[factor_idx], feat_dic, review_dic_fact, F.normalize(self.prototypes, dim=1))
                if layer_idx == len(self.encoder_layers) - 1:
                    int_dist_parts.append(int_dist)
                user_by_rating = self._split_layer_output(out["user"])
                item_by_rating = self._split_layer_output(out["movie"])
                next_states.append({
                    "user": {str(rating): user_by_rating[str(rating)] for rating in self.rating_vals},
                    "movie": {str(rating): item_by_rating[str(rating)] for rating in self.rating_vals},
                })
            factor_states = next_states

        user_parts = []
        item_parts = []
        for factor_idx in range(self.num_factors):
            user_parts.append(torch.stack(list(factor_states[factor_idx]["user"].values()), dim=0).mean(dim=0))
            item_parts.append(torch.stack(list(factor_states[factor_idx]["movie"].values()), dim=0).mean(dim=0))
        user_out = torch.cat(user_parts, dim=1)
        item_out = torch.cat(item_parts, dim=1)
        int_dist = torch.cat(int_dist_parts, dim=1) if int_dist_parts else user_out.new_zeros((0, self.num_factors))
        return user_out, item_out, int_dist, review_dic_fact

    def _split_layer_output(self, value: torch.Tensor) -> Dict[str, torch.Tensor]:
        if value.dim() == 3:
            return {str(rating): value[:, idx, :] for idx, rating in enumerate(self.rating_vals)}
        return {str(rating): value for rating in self.rating_vals}

    def _forward_once(self, batch: Dict[str, Any]):
        user_out, item_out, int_dist, review_dic_fact = self.encode(batch["enc_graphs"], batch["dec_graph"], batch["review_feat_dic"])
        pred_ratings, h_fea = self.decoder(batch["dec_graph"], user_out, item_out)
        self._assert_and_log_shapes(batch, pred_ratings, int_dist, review_dic_fact)
        return pred_ratings, h_fea, int_dist

    def _assert_and_log_shapes(self, batch, pred_ratings, int_dist, review_dic_fact) -> None:
        expected_encoder_edges = int(sum(int(x) for x in batch["rating_split"]))
        assert int_dist.shape == (expected_encoder_edges, self.num_factors), tuple(int_dist.shape)
        assert pred_ratings.shape[0] == batch["ratings"].shape[0], (tuple(pred_ratings.shape), tuple(batch["ratings"].shape))
        for rating in self.rating_vals:
            key = str(rating)
            expected_edges = batch["enc_graphs"][0].num_edges(key)
            assert review_dic_fact[key].shape == (expected_edges, self.num_factors, self.review_dim), tuple(review_dic_fact[key].shape)
        if not torch.isfinite(pred_ratings).all():
            raise FloatingPointError("SGDN prediction contains NaN/Inf.")
        if not self._shape_logged:
            print("SGDN first-batch shape check:", {
                "review_dim": self.review_dim,
                "num_factors": self.num_factors,
                "prototypes.shape": tuple(self.prototypes.shape),
                "int_dist.shape": tuple(int_dist.shape),
                "pred_ratings.shape": tuple(pred_ratings.shape),
                "rating_split": [int(x) for x in batch["rating_split"]],
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


# DIFFERENCES_FROM_ORIGINAL_SGDN:
# - The original code loads review vectors from a `(user, item) -> tensor` pickle; this framework uses
#   `review_embedding` columns generated/cached by the review4rating BERT pipeline.
# - Decoder edges are kept grouped by descending rating so `rating_split=[5,4,3,2,1]` aligns with
#   contrastive segments; the upstream loader relies on its own data order.
# - KMeans prototypes are intentionally applied after parameter initialization so they are not overwritten
#   by `reset_parameters()`, fixing an upstream construction-order issue.
