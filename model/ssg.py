from __future__ import annotations

# pyright: reportAny=false, reportArgumentType=false, reportImplicitOverride=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false

from collections.abc import Mapping, Sequence
from typing import cast

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .abstract import AbstractRec


def decov(x: torch.Tensor, y: torch.Tensor, diag: bool = False) -> torch.Tensor:
    bsz = x.size(0)
    x_centered = x - torch.mean(x, dim=0)[None, :]
    y_centered = y - torch.mean(y, dim=0)[None, :]
    mat = x_centered.t().mm(y_centered) / bsz
    loss = 0.5 * torch.norm(mat, p="fro") ** 2
    if diag:
        loss = loss - 0.5 * torch.norm(torch.diag(mat)) ** 2
    return cast(torch.Tensor, loss)


class TextCNN(nn.Module):
    def __init__(self, seq_len: int, vocab_size: int, emb_size: int, filter_sizes: Sequence[int], num_filters: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.filter_sizes = list(filter_sizes)
        self.num_filter_sizes = len(self.filter_sizes)
        self.num_filters = num_filters
        self.rembedding = nn.Embedding(vocab_size, emb_size)
        self.cnns = nn.ModuleList()
        self.pools = nn.ModuleList()
        for size in self.filter_sizes:
            self.cnns.append(nn.Conv2d(1, num_filters, kernel_size=(size, emb_size)))
            self.pools.append(nn.MaxPool2d(kernel_size=(seq_len - size + 1, 1), stride=(1, 1)))
        self.out_dim = self.num_filters * self.num_filter_sizes
        self.init_weights()

    def init_weights(self) -> None:
        nn.init.uniform_(self.rembedding.weight, -0.1, 0.1)
        for conv in self.cnns:
            nn.init.uniform_(conv.weight, -0.1, 0.1)
            nn.init.constant_(conv.bias, 0.1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.rembedding(inputs)
        pooled_out = []
        for idx in range(self.num_filter_sizes):
            h = F.relu(self.cnns[idx](inputs.view(-1, 1, self.seq_len, self.emb_size).contiguous()))
            pooled_out.append(self.pools[idx](h))
        return cast(torch.Tensor, torch.cat(pooled_out, 3).view(-1, self.num_filters * self.num_filter_sizes))


class TimeAttn(nn.Module):
    def __init__(self, dim: int, time_dim: int, beta: float, max_rel: int) -> None:
        super().__init__()
        self.beta = beta
        self.temperature = float(np.sqrt(dim * 1.0))
        vocab_size = max(150, max_rel + 1)
        self.pos_emb = nn.Embedding(vocab_size, time_dim)
        self.rel_emb = nn.Embedding(vocab_size, time_dim)
        self.fc_k = nn.Linear(dim, dim, bias=False)
        self.fc_q = nn.Linear(dim, dim, bias=False)
        self.fc_rp = nn.Linear(2 * time_dim, 1, bias=False)
        self.rate = nn.Parameter(torch.ones(1))
        self.init_weights()

    def init_weights(self) -> None:
        nn.init.uniform_(self.pos_emb.weight, -0.1, 0.1)
        nn.init.uniform_(self.rel_emb.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc_k.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc_q.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc_rp.weight, -0.1, 0.1)

    def forward(self, out: torch.Tensor, hn: torch.Tensor, pos_ind: torch.Tensor, rel_dt: torch.Tensor, abs_dt: torch.Tensor) -> torch.Tensor:
        del abs_dt
        pad_mask = pos_ind == 0
        pos_emb = self.pos_emb(pos_ind.clamp(min=0, max=self.pos_emb.num_embeddings - 1))
        rel_emb = self.rel_emb(rel_dt.clamp(min=0, max=self.rel_emb.num_embeddings - 1))
        attn_k = self.fc_k(out)
        attn_q = self.fc_q(hn)
        attn_0 = torch.bmm(attn_k, attn_q.unsqueeze(-1)).squeeze(-1) / self.temperature
        attn_1 = self.fc_rp(torch.cat([rel_emb, pos_emb], -1)).squeeze(-1)
        attn = attn_0 + self.beta * attn_1
        attn = attn.masked_fill(pad_mask, torch.finfo(attn.dtype).min)
        attn = F.softmax(attn, 1)
        attn = torch.where(torch.isfinite(attn), attn, torch.zeros_like(attn))
        return cast(torch.Tensor, torch.bmm(attn.unsqueeze(1), out).squeeze(1))


class GRUModule(nn.Module):
    def __init__(self, input_dim: int, gru_dim: int, time_dim: int, beta: float, max_rel: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, gru_dim, batch_first=True)
        self.attention = TimeAttn(gru_dim, time_dim, beta, max_rel)

    def forward(self, inputs: torch.Tensor, length: torch.Tensor, pos_ind: torch.Tensor, rel_dt: torch.Tensor, abs_dt: torch.Tensor) -> torch.Tensor:
        safe_length = length.clamp(min=1, max=inputs.size(1))
        sorted_len, sorted_idx = safe_length.sort(0, descending=True)
        index_sorted_idx = sorted_idx.view(-1, 1, 1).expand_as(inputs)
        sorted_inputs = inputs.gather(0, index_sorted_idx.long())
        packed = pack_padded_sequence(sorted_inputs, sorted_len.detach().cpu(), batch_first=True)
        out, hn = self.gru(packed)
        hn = torch.squeeze(hn, 0)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=inputs.size(1))
        _, ori_idx = sorted_idx.sort(0, descending=False)
        hn = hn.gather(0, ori_idx.view(-1, 1).expand_as(hn).long())
        out = out.gather(0, ori_idx.view(-1, 1, 1).expand_as(out).long())
        return self.attention(out, hn, pos_ind, rel_dt, abs_dt)


class SpecialSpmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices: torch.Tensor, values: torch.Tensor, shape: torch.Size, b: torch.Tensor) -> torch.Tensor:
        assert indices.requires_grad is False
        a = torch.sparse_coo_tensor(indices, values, shape, device=values.device)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[None, torch.Tensor | None, None, torch.Tensor | None]:
        a, b = ctx.saved_tensors
        grad_values = None
        grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices: torch.Tensor, values: torch.Tensor, shape: torch.Size, b: torch.Tensor) -> torch.Tensor:
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, emb_size: int, max_rating: int, att_dim: int, dropout: float, alpha: float, concat: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.zeros(size=(1, 2 * (out_features + att_dim))))
        self.re_W = nn.Parameter(torch.zeros(size=(emb_size, att_dim)))
        self.ra_W = nn.Parameter(torch.zeros(size=(max_rating, att_dim)))
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()
        self.init_weights()

    def init_weights(self) -> None:
        nn.init.uniform_(self.W, -0.1, 0.1)
        nn.init.uniform_(self.a, -0.1, 0.1)
        nn.init.uniform_(self.re_W, -0.1, 0.1)
        nn.init.uniform_(self.ra_W, -0.1, 0.1)

    def forward(self, inputs: torch.Tensor, adj: torch.Tensor, review: torch.Tensor, rating: torch.Tensor) -> torch.Tensor:
        device = inputs.device
        node_count = inputs.size(0)
        edge = adj.nonzero().t()
        h = torch.mm(inputs, self.W)
        if edge.size(1) == 0:
            return F.elu(h) if self.concat else h
        re_h = torch.mm(review, self.re_W)
        ra_h = torch.mm(rating, self.ra_W)
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :], re_h, ra_h), dim=1).t()
        edge_e = torch.exp(self.leakyrelu(self.a.mm(edge_h).squeeze()))
        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([node_count, node_count]), torch.ones(size=(node_count, 1), device=device))
        e_rowsum = e_rowsum + 1e-10
        edge_e = self.dropout(edge_e)
        h_prime = self.special_spmm(edge, edge_e, torch.Size([node_count, node_count]), h)
        h_prime = h_prime.div(e_rowsum)
        h_prime = h_prime + h
        return F.elu(h_prime) if self.concat else h_prime


class GraphModel(nn.Module):
    def __init__(self, input_size: int, node_num: int, node_emb: int, hid_dim: int, n_hops: int, max_rating: int, att_dim: int, n_heads: int, alpha: float, keep_prob: float) -> None:
        super().__init__()
        self.node_embedding = nn.Embedding(node_num, node_emb)
        self.attentions = nn.ModuleList()
        self.n_hops = n_hops
        for hop_idx in range(n_hops - 1):
            heads = nn.ModuleList()
            in_feat = node_emb if hop_idx == 0 else hid_dim * n_heads
            for _ in range(n_heads):
                heads.append(SpGraphAttentionLayer(in_feat, hid_dim, input_size, max_rating, att_dim, 1.0 - keep_prob, alpha, concat=True))
            self.attentions.append(heads)
        self.out_att = SpGraphAttentionLayer(hid_dim * n_heads if n_hops > 1 else node_emb, hid_dim, input_size, max_rating, att_dim, 1.0 - keep_prob, alpha, concat=False)
        self.init_weights()

    def init_weights(self) -> None:
        nn.init.uniform_(self.node_embedding.weight, -0.1, 0.1)

    def forward(self, nodes: torch.Tensor, edge_emb: torch.Tensor, ratings: torch.Tensor, adj: torch.Tensor, pairs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.node_embedding(nodes)
        for hop_idx in range(self.n_hops - 1):
            h = torch.cat([att(h, adj, edge_emb, ratings) for att in self.attentions[hop_idx]], dim=1)
        out = self.out_att(h, adj, edge_emb, ratings)
        return out[pairs[:, 0]], out[pairs[:, 1]]


class SSG(AbstractRec):
    def __init__(self, configs: Mapping[str, object], train_dataset: object) -> None:
        super().__init__()
        self.configs = configs
        self.review_num_u = int(getattr(train_dataset, "review_num_u"))
        self.review_num_i = int(getattr(train_dataset, "review_num_i"))
        self.review_len_u = int(getattr(train_dataset, "review_len_u"))
        self.review_len_i = int(getattr(train_dataset, "review_len_i"))
        self.review_len_g = int(getattr(train_dataset, "review_len_g", self.review_len_u))
        self.user_num = int(getattr(train_dataset, "num_users"))
        self.item_num = int(getattr(train_dataset, "num_items"))
        self.node_num = self.user_num + self.item_num
        self.filter_sizes = self._get_int_list("filter_sizes", [3])
        self.emb_size = int(configs.get("word_dim", 300))
        self.id_emb = int(configs.get("id_dim", 32))
        self.att_dim = int(configs.get("attention_size", 32))
        self.n_latent = int(configs.get("latent_dim", 8))
        self.gru_dim = int(configs.get("gru_dim", 32))
        self.time_dim = int(configs.get("time_dim", 32))
        self.num_filters = int(configs.get("num_filters", 100))
        self.keep_prob = float(configs.get("keep_prob", 1.0))
        self.alpha = float(configs.get("alpha", 0.25))
        self.beta = float(configs.get("beta", 1.0))
        self.n_hops = int(configs.get("n_hops", 2))
        self.n_heads = int(configs.get("n_heads", 8))
        self.node_emb = int(configs.get("graph_node_dim", 128))
        self.hid_dim = int(configs.get("graph_hidden_dim", 64))
        self.graph_att_dim = int(configs.get("graph_attention_dim", 32))
        self.max_rating = int(configs.get("max_rating", 5))
        self.max_rel = int(configs.get("max_rel_bucket", 100))
        self.decov_lambda = float(configs.get("decov_lambda", 0.01))
        self.l2_reg_lambda = float(configs.get("l2_lambda", 1.0))
        self.train_clip = self._get_bool("train_clip", False)
        self.test_clip = self._get_bool("test_clip", True)
        self.min_rating = float(configs.get("min_rating", 1.0))
        self.max_rating_value = float(configs.get("max_rating", 5.0))
        self.debug_shapes = self._get_bool("debug_shapes", False)
        self._shape_logged = False

        self.user_vocab_size = len(getattr(train_dataset, "vocabulary_user"))
        self.item_vocab_size = len(getattr(train_dataset, "vocabulary_item"))
        self.graph_vocab_size = len(getattr(train_dataset, "vocabulary"))
        self.set_dim = self.num_filters * len(self.filter_sizes)

        self.user_remb = nn.Embedding(self.user_vocab_size, self.emb_size)
        self.user_idemb_att = nn.Embedding(self.user_num + 2, self.id_emb)
        self.item_remb = nn.Embedding(self.item_vocab_size, self.emb_size)
        self.item_idemb_att = nn.Embedding(self.item_num + 2, self.id_emb)
        self.idemb = nn.Embedding(self.node_num, self.n_latent)

        self.user_cnns = nn.ModuleList()
        self.user_pools = nn.ModuleList()
        for size in self.filter_sizes:
            self.user_cnns.append(nn.Conv2d(1, self.num_filters, kernel_size=(size, self.emb_size)))
            self.user_pools.append(nn.MaxPool2d(kernel_size=(self.review_len_u - size + 1, 1), stride=(1, 1)))
        self.item_cnns = nn.ModuleList()
        self.item_pools = nn.ModuleList()
        for size in self.filter_sizes:
            self.item_cnns.append(nn.Conv2d(1, self.num_filters, kernel_size=(size, self.emb_size)))
            self.item_pools.append(nn.MaxPool2d(kernel_size=(self.review_len_i - size + 1, 1), stride=(1, 1)))

        self.Wau = Parameter(torch.Tensor(self.set_dim, self.att_dim))
        self.Wru = Parameter(torch.Tensor(self.id_emb, self.att_dim))
        self.Wpu = Parameter(torch.Tensor(self.att_dim, 1))
        self.bau = Parameter(torch.Tensor(self.att_dim))
        self.bbu = Parameter(torch.Tensor(1))
        self.Wai = Parameter(torch.Tensor(self.set_dim, self.att_dim))
        self.Wri = Parameter(torch.Tensor(self.id_emb, self.att_dim))
        self.Wpi = Parameter(torch.Tensor(self.att_dim, 1))
        self.bai = Parameter(torch.Tensor(self.att_dim))
        self.bbi = Parameter(torch.Tensor(1))

        self.u_dropout = nn.Dropout(1.0 - self.keep_prob)
        self.i_dropout = nn.Dropout(1.0 - self.keep_prob)
        self.u_gru = GRUModule(self.set_dim, self.gru_dim, self.time_dim, self.beta, self.max_rel)
        self.i_gru = GRUModule(self.set_dim, self.gru_dim, self.time_dim, self.beta, self.max_rel)
        self.u_fc = nn.Linear(self.set_dim + self.gru_dim + self.hid_dim, self.n_latent)
        self.i_fc = nn.Linear(self.set_dim + self.gru_dim + self.hid_dim, self.n_latent)
        self.fm_dropout = nn.Dropout(1.0 - self.keep_prob)
        self.Wmul = Parameter(torch.Tensor(self.n_latent, 1))
        self.biases = Parameter(torch.Tensor(self.node_num))
        self.gbias = Parameter(torch.Tensor(1))
        self.mse_loss = nn.MSELoss()
        self.graph_cnn = TextCNN(self.review_len_g, self.graph_vocab_size, self.emb_size, self.filter_sizes, self.num_filters)
        self.graph_view = GraphModel(self.set_dim, self.node_num, self.node_emb, self.hid_dim, self.n_hops, self.max_rating, self.graph_att_dim, self.n_heads, self.alpha, self.keep_prob)
        self.init_weights()
        self._maybe_load_word2vec(train_dataset)
        total_params = sum(parameter.numel() for parameter in self.parameters())
        print(f"SSG parameters: total={total_params}, trainable={sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)}")

    def _get_int_list(self, key: str, default: Sequence[int]) -> list[int]:
        value = self.configs.get(key, default)
        if isinstance(value, str):
            return [int(part.strip()) for part in value.strip().strip("[]").split(",") if part.strip()]
        if isinstance(value, Sequence):
            return [int(part) for part in value]
        return list(default)

    def _get_bool(self, key: str, default: bool) -> bool:
        value = self.configs.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def init_weights(self) -> None:
        nn.init.uniform_(self.user_remb.weight, -0.1, 0.1)
        nn.init.uniform_(self.user_idemb_att.weight, -0.1, 0.1)
        nn.init.uniform_(self.item_remb.weight, -0.1, 0.1)
        nn.init.uniform_(self.item_idemb_att.weight, -0.1, 0.1)
        nn.init.uniform_(self.idemb.weight, -0.1, 0.1)
        for conv in self.user_cnns:
            nn.init.uniform_(conv.weight, -0.1, 0.1)
            nn.init.constant_(conv.bias, 0.1)
        for conv in self.item_cnns:
            nn.init.uniform_(conv.weight, -0.1, 0.1)
            nn.init.constant_(conv.bias, 0.1)
        for parameter in [self.Wau, self.Wru, self.Wpu, self.Wai, self.Wri, self.Wpi, self.Wmul]:
            nn.init.uniform_(parameter, -0.1, 0.1)
        for parameter in [self.bau, self.bbu, self.bai, self.bbi, self.biases, self.gbias]:
            nn.init.constant_(parameter, 0.1)
        nn.init.uniform_(self.u_fc.weight, -0.1, 0.1)
        nn.init.constant_(self.u_fc.bias, 0.1)
        nn.init.uniform_(self.i_fc.weight, -0.1, 0.1)
        nn.init.constant_(self.i_fc.bias, 0.1)

    def _maybe_load_word2vec(self, train_dataset: object) -> None:
        path = str(self.configs.get("word2vec", self.configs.get("glove_path", "")))
        if not path or not os.path.exists(path):
            return
        if not path.endswith(".bin"):
            return
        try:
            self.user_remb.weight.data.copy_(torch.tensor(self._read_binary_word2vec(path, getattr(train_dataset, "vocabulary_user"), self.emb_size), dtype=torch.float32))
            self.item_remb.weight.data.copy_(torch.tensor(self._read_binary_word2vec(path, getattr(train_dataset, "vocabulary_item"), self.emb_size), dtype=torch.float32))
            self.graph_cnn.rembedding.weight.data.copy_(torch.tensor(self._read_binary_word2vec(path, getattr(train_dataset, "vocabulary"), self.emb_size), dtype=torch.float32))
            print("SSG loaded binary word2vec embeddings")
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            print(f"SSG word2vec load skipped: {exc}")

    @staticmethod
    def _read_binary_word2vec(path: str, vocabulary: Mapping[str, int], embedding_dim: int) -> np.ndarray:
        init_w = np.random.uniform(-1.0, 1.0, (len(vocabulary), embedding_dim)).astype(np.float32)
        with open(path, "rb") as handle:
            header = handle.readline()
            vocab_size, layer1_size = map(int, header.split())
            binary_len = np.dtype("float32").itemsize * layer1_size
            if layer1_size != embedding_dim:
                raise ValueError(f"word2vec dim {layer1_size} != configured {embedding_dim}")
            for _ in range(vocab_size):
                word_bytes = []
                while True:
                    ch = handle.read(1)
                    if ch == b" ":
                        word = b"".join(word_bytes).decode("latin1")
                        break
                    if ch != b"\n":
                        word_bytes.append(ch)
                vector = np.frombuffer(handle.read(binary_len), dtype="float32")
                if word in vocabulary:
                    init_w[int(vocabulary[word])] = vector
        return init_w

    @staticmethod
    def broad_mm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        bsz, rows, dim = x.size()
        dim_y, cols = y.size()
        assert dim == dim_y
        return torch.mm(x.view(-1, dim), y).view(bsz, rows, cols)

    def _assert_and_log_shapes(self, batch: Mapping[str, torch.Tensor]) -> None:
        if not self.debug_shapes or self._shape_logged:
            return
        keys = ["nodes", "reviews", "ratings", "adj", "pairs", "input_u", "input_i", "reuid", "reiid", "u_pos_ind", "u_rel_dt", "u_abs_dt"]
        print("SSG model input shapes:", {key: tuple(batch[key].shape) for key in keys if key in batch})
        self._shape_logged = True

    def forward_original(self, input_u: torch.Tensor, input_i: torch.Tensor, reuid: torch.Tensor, reiid: torch.Tensor, u_s_renum: torch.Tensor, i_s_renum: torch.Tensor, u_pos_ind: torch.Tensor, i_pos_ind: torch.Tensor, u_rel_dt: torch.Tensor, i_rel_dt: torch.Tensor, u_abs_dt: torch.Tensor, i_abs_dt: torch.Tensor, nodes: torch.Tensor, reviews: torch.Tensor, ratings: torch.Tensor, adj: torch.Tensor, pairs: torch.Tensor, uid: torch.Tensor, iid: torch.Tensor, y: torch.Tensor, clip: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_emb = self.graph_cnn(reviews)
        graph_ufeas, graph_ifeas = self.graph_view(nodes, edge_emb, ratings, adj, pairs)

        embedding_users = self.user_remb(input_u)
        embedding_items = self.item_remb(input_i)
        pooled_out_u = []
        for idx in range(len(self.filter_sizes)):
            h = F.relu(self.user_cnns[idx](embedding_users.view(-1, 1, self.review_len_u, self.emb_size)))
            pooled_out_u.append(self.user_pools[idx](h))
        reviews_u = torch.cat(pooled_out_u, 3).view(-1, self.review_num_u, self.set_dim)
        pooled_out_i = []
        for idx in range(len(self.filter_sizes)):
            h = F.relu(self.item_cnns[idx](embedding_items.view(-1, 1, self.review_len_i, self.emb_size)))
            pooled_out_i.append(self.item_pools[idx](h))
        reviews_i = torch.cat(pooled_out_i, 3).view(-1, self.review_num_i, self.set_dim)

        iid_a = F.relu(self.item_idemb_att(reuid.clamp(min=0, max=self.item_num + 1)))
        u_j = self.broad_mm(F.relu(self.broad_mm(reviews_u, self.Wau) + self.broad_mm(iid_a, self.Wru) + self.bau), self.Wpu) + self.bbu
        u_a = F.softmax(u_j, 1)
        uid_a = F.relu(self.user_idemb_att(reiid.clamp(min=0, max=self.user_num + 1)))
        i_j = self.broad_mm(F.relu(self.broad_mm(reviews_i, self.Wai) + self.broad_mm(uid_a, self.Wri) + self.bai), self.Wpi) + self.bbi
        i_a = F.softmax(i_j, 1)

        u_set = torch.sum(reviews_u * u_a, 1)
        i_set = torch.sum(reviews_i * i_a, 1)
        u_hn = self.u_gru(reviews_u, u_s_renum, u_pos_ind, u_rel_dt, u_abs_dt)
        i_hn = self.i_gru(reviews_i, i_s_renum, i_pos_ind, i_rel_dt, i_abs_dt)

        decov_loss = decov(u_set, u_hn) + decov(i_set, i_hn)
        decov_loss = decov_loss + decov(u_hn, graph_ufeas) + decov(i_hn, graph_ifeas)
        decov_loss = decov_loss + decov(u_set, graph_ufeas) + decov(i_set, graph_ifeas)

        u_feas = self.u_fc(torch.cat([u_set, u_hn, graph_ufeas], dim=1))
        i_feas = self.i_fc(torch.cat([i_set, i_hn, graph_ifeas], dim=1))
        uid_emb = self.idemb(uid).view(-1, self.n_latent)
        iid_emb = self.idemb(iid).view(-1, self.n_latent)
        u_feas = u_feas + uid_emb
        i_feas = i_feas + iid_emb
        fm = F.relu(u_feas * i_feas)
        mul = torch.matmul(fm, self.Wmul)
        pred = torch.sum(mul, 1, keepdim=True)
        u_bias = torch.gather(self.biases, 0, uid).view(-1, 1)
        i_bias = torch.gather(self.biases, 0, iid).view(-1, 1)
        pred = (pred + u_bias + i_bias + self.gbias).view(-1)
        if clip:
            pred = torch.clamp(pred, self.min_rating, self.max_rating_value)
        y = y.float().view(-1)
        mse = 0.5 * self.mse_loss(pred, y)
        l2_loss = 0.5 * torch.sum(self.Wau ** 2) + 0.5 * torch.sum(self.Wru ** 2) + 0.5 * torch.sum(self.Wai ** 2) + 0.5 * torch.sum(self.Wri ** 2)
        loss = mse + self.l2_reg_lambda * l2_loss + self.decov_lambda * decov_loss
        mae = torch.mean(torch.abs(pred - y))
        rmse = torch.sqrt(torch.mean((pred - y) ** 2))
        self._last_losses = {"mse_loss": mse.detach(), "l2_loss": l2_loss.detach(), "decov_loss": decov_loss.detach(), "total_loss": loss.detach(), "mae": mae.detach(), "rmse": rmse.detach()}
        return loss, mae, rmse, pred

    def _batch_from_kwargs(self, kwargs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        input_u = kwargs["input_u"] if "input_u" in kwargs else kwargs["user_review"]
        input_i = kwargs["input_i"] if "input_i" in kwargs else kwargs["item_review"]
        reuid = kwargs["reuid"] if "reuid" in kwargs else kwargs["user_review_item_ids"]
        reiid = kwargs["reiid"] if "reiid" in kwargs else kwargs["item_review_user_ids"]
        u_s_renum = kwargs["u_s_renum"] if "u_s_renum" in kwargs else kwargs["user_seq_len"]
        i_s_renum = kwargs["i_s_renum"] if "i_s_renum" in kwargs else kwargs["item_seq_len"]
        u_pos_ind = kwargs["u_pos_ind"] if "u_pos_ind" in kwargs else kwargs["user_pos_ind"]
        i_pos_ind = kwargs["i_pos_ind"] if "i_pos_ind" in kwargs else kwargs["item_pos_ind"]
        u_rel_dt = kwargs["u_rel_dt"] if "u_rel_dt" in kwargs else kwargs["user_rel_dt"]
        i_rel_dt = kwargs["i_rel_dt"] if "i_rel_dt" in kwargs else kwargs["item_rel_dt"]
        u_abs_dt = kwargs["u_abs_dt"] if "u_abs_dt" in kwargs else kwargs["user_abs_dt"]
        i_abs_dt = kwargs["i_abs_dt"] if "i_abs_dt" in kwargs else kwargs["item_abs_dt"]
        nodes = kwargs["nodes"] if "nodes" in kwargs else kwargs["graph_nodes"]
        reviews = kwargs["reviews"] if "reviews" in kwargs else kwargs["graph_reviews"]
        graph_ratings = kwargs["ratings"] if "ratings" in kwargs else kwargs["graph_ratings"]
        adj = kwargs["adj"] if "adj" in kwargs else kwargs["graph_adj"]
        return {
            "input_u": input_u,
            "input_i": input_i,
            "reuid": reuid,
            "reiid": reiid,
            "u_s_renum": u_s_renum,
            "i_s_renum": i_s_renum,
            "u_pos_ind": u_pos_ind,
            "i_pos_ind": i_pos_ind,
            "u_rel_dt": u_rel_dt,
            "i_rel_dt": i_rel_dt,
            "u_abs_dt": u_abs_dt,
            "i_abs_dt": i_abs_dt,
            "nodes": nodes,
            "reviews": reviews,
            "ratings": graph_ratings,
            "adj": adj,
            "pairs": kwargs["pairs"],
            "user_id": kwargs["user_id"],
            "item_id": kwargs["item_id"],
            "rating": kwargs.get("rating", torch.zeros_like(kwargs["user_id"], dtype=torch.float32)),
        }

    def forward(self, **kwargs: torch.Tensor) -> torch.Tensor:
        batch = self._batch_from_kwargs(kwargs)
        self._assert_and_log_shapes(batch)
        clip = self.train_clip if self.training else self.test_clip
        _, _, _, pred = self.forward_original(
            batch["input_u"].long(), batch["input_i"].long(), batch["reuid"].long(), batch["reiid"].long(),
            batch["u_s_renum"].long(), batch["i_s_renum"].long(), batch["u_pos_ind"].long(), batch["i_pos_ind"].long(),
            batch["u_rel_dt"].long(), batch["i_rel_dt"].long(), batch["u_abs_dt"].float(), batch["i_abs_dt"].float(),
            batch["nodes"].long(), batch["reviews"].long(), batch["ratings"].float(), batch["adj"].float(), batch["pairs"].long(),
            batch["user_id"].long(), batch["item_id"].long(), batch["rating"].float(), clip,
        )
        return pred.view(-1, 1)

    def cal_loss(self, batch_data: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        batch = self._batch_from_kwargs(batch_data)
        self._assert_and_log_shapes(batch)
        loss, mae, rmse, _ = self.forward_original(
            batch["input_u"].long(), batch["input_i"].long(), batch["reuid"].long(), batch["reiid"].long(),
            batch["u_s_renum"].long(), batch["i_s_renum"].long(), batch["u_pos_ind"].long(), batch["i_pos_ind"].long(),
            batch["u_rel_dt"].long(), batch["i_rel_dt"].long(), batch["u_abs_dt"].float(), batch["i_abs_dt"].float(),
            batch["nodes"].long(), batch["reviews"].long(), batch["ratings"].float(), batch["adj"].float(), batch["pairs"].long(),
            batch["user_id"].long(), batch["item_id"].long(), batch["rating"].float(), self.train_clip,
        )
        last = getattr(self, "_last_losses", {})
        return loss, {
            "mse_loss": float(last.get("mse_loss", torch.tensor(0.0)).item()),
            "l2_loss": float(last.get("l2_loss", torch.tensor(0.0)).item()),
            "decov_loss": float(last.get("decov_loss", torch.tensor(0.0)).item()),
            "mae": float(mae.detach().item()),
            "rmse": float(rmse.detach().item()),
            "total_loss": float(loss.detach().item()),
        }
