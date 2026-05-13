import numpy as np
import torch

from .abstract_dataset import RecDataset

try:
    import dgl
    dgl_import_error = None
except Exception as exc:  # pragma: no cover - dependency is validated at runtime.
    dgl = None
    dgl_import_error = exc


class SGDNDataset(RecDataset):
    """Full-graph SGDN dataset using DGL heterographs.

    The original SGDN trains on a full encoder graph and predicts on a decoder
    bipartite graph.  This dataset preserves the project's one-batch full-graph
    loader contract while exposing the upstream-style objects:
    ``enc_graphs``, ``dec_graph``, ``review_feat_dic`` and ``rating_split``.
    """

    def __init__(self, df, configs, split="train"):
        super().__init__(df, configs, split)
        if dgl is None:
            raise ImportError(
                "SGDN requires a working DGL installation. Install a DGL build compatible with your PyTorch version "
                "(and GraphBolt binary, for DGL 2.x)."
            ) from dgl_import_error
        self._dgl = dgl

        self.rating_vals = [int(v) for v in configs.get("rating_values", [1, 2, 3, 4, 5])]
        self.review_dim = int(configs.get("review_dim", configs.get("review_feat_size", configs.get("bert_whitening_dim", 384))))
        self.num_factors = int(configs.get("num_factors", configs.get("num_factor", 2)))
        self.use_review_feat = bool(configs.get("use_review", True))

        self.enc_graphs = []
        self.dec_graph = None
        self.review_feat_dic = {}
        self.rating_split = []
        self.decoder_user_ids = None
        self.decoder_item_ids = None
        self.decoder_review_feat = None
        self.labels = None
        self.ratings = None

        self._build_decoder_graph(self.df)
        self._build_encoder_graphs(self.df)

    def _extract_review_tensor(self, frame) -> torch.Tensor:
        if not self.use_review_feat or "review_embedding" not in frame.columns:
            return torch.zeros((len(frame), self.review_dim), dtype=torch.float32)

        review_vectors = []
        for review_embedding in frame["review_embedding"].tolist():
            if isinstance(review_embedding, torch.Tensor):
                vector = review_embedding.detach().float().view(-1)
            elif review_embedding is None or (isinstance(review_embedding, float) and np.isnan(review_embedding)):
                vector = torch.zeros(self.review_dim, dtype=torch.float32)
            else:
                vector = torch.tensor(review_embedding, dtype=torch.float32).view(-1)
            if vector.numel() != self.review_dim:
                raise ValueError(f"SGDN expected review_dim={self.review_dim}, got {vector.numel()}.")
            review_vectors.append(vector)

        if not review_vectors:
            return torch.zeros((0, self.review_dim), dtype=torch.float32)
        return torch.stack(review_vectors, dim=0)

    def _build_decoder_graph(self, frame) -> None:
        if len(frame) > 0:
            ordered_frame = frame.assign(_sgdn_rating_order=frame["rating"].astype(float)).sort_values(
                "_sgdn_rating_order", ascending=False, kind="mergesort"
            ).drop(columns=["_sgdn_rating_order"])
        else:
            ordered_frame = frame

        ratings_np = ordered_frame["rating"].to_numpy(dtype=np.float32)
        user_ids_np = ordered_frame["user_id"].to_numpy(dtype=np.int64)
        item_ids_np = ordered_frame["item_id"].to_numpy(dtype=np.int64)

        self.decoder_user_ids = torch.tensor(user_ids_np, dtype=torch.long)
        self.decoder_item_ids = torch.tensor(item_ids_np, dtype=torch.long)
        self.ratings = torch.tensor(ratings_np, dtype=torch.float32)
        self.labels = torch.tensor(np.searchsorted(np.array(self.rating_vals, dtype=np.float32), ratings_np), dtype=torch.long)
        self.decoder_review_feat = self._extract_review_tensor(ordered_frame)

        graph_data = {
            ("user", "rate", "movie"): (self.decoder_user_ids, self.decoder_item_ids),
        }
        heterograph = getattr(self._dgl, "heterograph")
        self.dec_graph = heterograph(
            graph_data,
            num_nodes_dict={"user": self.num_users, "movie": self.num_items},
        )
        self.dec_graph.edges["rate"].data["review_feat"] = self.decoder_review_feat.clone()

    def _build_encoder_graphs(self, frame) -> None:
        user_ids = frame["user_id"].to_numpy(dtype=np.int64)
        item_ids = frame["item_id"].to_numpy(dtype=np.int64)
        ratings = frame["rating"].to_numpy(dtype=np.float32)
        review_feat = self._extract_review_tensor(frame)

        graph_data = {}
        edge_reviews = {}
        self.review_feat_dic = {}
        for rating in self.rating_vals:
            mask = ratings == float(rating)
            edge_users = torch.tensor(user_ids[mask], dtype=torch.long)
            edge_items = torch.tensor(item_ids[mask], dtype=torch.long)
            rating_review = review_feat[torch.from_numpy(mask)]
            rating_key = str(rating)

            graph_data[("user", rating_key, "movie")] = (edge_users, edge_items)
            graph_data[("movie", f"rev-{rating_key}", "user")] = (edge_items, edge_users)
            edge_reviews[rating_key] = rating_review
            self.review_feat_dic[rating_key] = rating_review

        self.enc_graphs = []
        for _ in range(self.num_factors):
            heterograph = getattr(self._dgl, "heterograph")
            graph = heterograph(
                graph_data,
                num_nodes_dict={"user": self.num_users, "movie": self.num_items},
            )
            for rating in self.rating_vals:
                rating_key = str(rating)
                rating_review = edge_reviews[rating_key]
                graph.edges[rating_key].data["review_feat"] = rating_review.clone()
                graph.edges[f"rev-{rating_key}"].data["review_feat"] = rating_review.clone()
                forward_src, forward_dst = graph.edges(etype=rating_key)
                reverse_src, reverse_dst = graph.edges(etype=f"rev-{rating_key}")
                graph.edges[rating_key].data["src_id"] = forward_src
                graph.edges[rating_key].data["dst_id"] = forward_dst
                graph.edges[f"rev-{rating_key}"].data["src_id"] = reverse_src
                graph.edges[f"rev-{rating_key}"].data["dst_id"] = reverse_dst
                graph.edges[rating_key].data["w"] = torch.ones((rating_review.size(0), 1), dtype=torch.float32)
                graph.edges[f"rev-{rating_key}"].data["w"] = torch.ones((rating_review.size(0), 1), dtype=torch.float32)
            self.enc_graphs.append(graph)

        self.rating_split = [int((ratings == float(rating)).sum()) for rating in sorted(self.rating_vals, reverse=True)]
        assert len(self.enc_graphs) == self.num_factors
        assert isinstance(self.dec_graph, self._dgl.DGLHeteroGraph)

    def _setup_evaluation(self, train_df, valid_df, test_df):
        if self.split in {"valid", "test"}:
            self._build_encoder_graphs(train_df)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {
            "enc_graphs": self.enc_graphs,
            "dec_graph": self.dec_graph,
            "review_feat_dic": self.review_feat_dic,
            "labels": self.labels,
            "ratings": self.ratings,
            "rating_split": self.rating_split,
            # Kept for trainer diagnostics and backward-compatible scripts.
            "decoder_user_ids": self.decoder_user_ids,
            "decoder_item_ids": self.decoder_item_ids,
            "decoder_review_feat": self.decoder_review_feat,
        }


def sgdn_collate_fn(batch):
    return batch[0]
