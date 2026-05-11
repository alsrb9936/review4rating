# pyright: reportAny=false, reportArgumentType=false, reportDeprecated=false, reportImplicitOverride=false, reportIncompatibleMethodOverride=false, reportUnannotatedClassAttribute=false, reportUninitializedInstanceVariable=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnnecessaryCast=false, reportUnnecessaryComparison=false, reportUnusedCallResult=false, reportUnusedVariable=false

from collections.abc import Mapping, Sequence
from typing import Dict, List, Optional, Tuple, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .abstract import AbstractRec


class SSG(AbstractRec):
    def __init__(self, configs: Mapping[str, object], train_dataset: object) -> None:
        super().__init__()
        self.configs = configs
        self.train_dataset = train_dataset

        self.review_length = self._get_int_config("review_length", 40)
        self.review_count = self._get_int_config("review_count", 10)
        self.seq_count = self._get_int_config("seq_count", 10)
        self.word_dim = self._get_int_config("word_dim", 300)
        self.filter_sizes = self._get_int_list_config("filter_sizes", [3])
        self.num_filters = self._get_int_config("num_filters", 100)
        self.id_dim = self._get_int_config("id_dim", 32)
        self.attention_size = self._get_int_config("attention_size", 32)
        self.gru_dim = self._get_int_config("gru_dim", 100)
        self.time_dim = self._get_int_config("time_dim", 32)
        self.latent_dim = self._get_int_config("latent_dim", 32)
        self.review_dim = self._get_int_config("review_dim", self._get_int_config("bert_whitening_dim", 64))
        self.review_input_mode = str(self.configs.get("review_input_mode", "token"))
        if self.review_input_mode not in {"token", "embedding"}:
            raise ValueError("review_input_mode must be 'token' or 'embedding'.")

        self.use_set_view = self._get_bool_config("use_set_view", True)
        self.use_sequence_view = self._get_bool_config("use_sequence_view", True)
        self.use_graph_view = self._get_bool_config("use_graph_view", False)

        self.graph_hidden_dim = self._get_int_config("graph_hidden_dim", 32)
        self.graph_node_dim = self._get_int_config("graph_node_dim", 32)
        self.graph_attention_dim = self._get_int_config("graph_attention_dim", 32)
        self.n_hops = self._get_int_config("n_hops", 2)
        self.n_heads = self._get_int_config("n_heads", 2)
        self.alpha = self._get_float_config("alpha", 0.2)

        self.dropout_prob = self._get_float_config("dropout_prob", 0.5)
        self.decov_lambda = self._get_float_config("decov_lambda", 0.01)
        self.l2_lambda = self._get_float_config("l2_lambda", 0.001)

        self.num_users = int(getattr(train_dataset, "num_users"))
        self.num_items = int(getattr(train_dataset, "num_items"))
        self.pad_idx = int(getattr(train_dataset, "pad_idx", 0))

        embedding_matrix = torch.as_tensor(
            getattr(train_dataset, "embedding_matrix"),
            dtype=torch.float32,
        )
        if embedding_matrix.size(1) != self.word_dim:
            raise ValueError(
                "Configured word_dim={} does not match embedding dim={}".format(
                    self.word_dim,
                    embedding_matrix.size(1),
                )
            )
        self.embedding_matrix = embedding_matrix

        self.word_embedding = nn.Embedding.from_pretrained(
            self.embedding_matrix,
            freeze=False,
            padding_idx=self.pad_idx,
        )

        self.cnn_out_dim = self.num_filters * len(self.filter_sizes)
        self.user_convs = nn.ModuleList(
            [
                nn.Conv1d(self.word_dim, self.num_filters, kernel_size=kernel_size)
                for kernel_size in self.filter_sizes
            ]
        )
        self.item_convs = nn.ModuleList(
            [
                nn.Conv1d(self.word_dim, self.num_filters, kernel_size=kernel_size)
                for kernel_size in self.filter_sizes
            ]
        )
        self.user_embedding_projection = nn.Linear(self.review_dim, self.cnn_out_dim)
        self.item_embedding_projection = nn.Linear(self.review_dim, self.cnn_out_dim)

        self.user_review_fc = nn.Linear(self.cnn_out_dim, self.attention_size)
        self.item_review_fc = nn.Linear(self.cnn_out_dim, self.attention_size)
        self.user_id_attention_fc = nn.Linear(self.id_dim, self.attention_size)
        self.item_id_attention_fc = nn.Linear(self.id_dim, self.attention_size)
        self.user_attention_fc = nn.Linear(self.attention_size, 1)
        self.item_attention_fc = nn.Linear(self.attention_size, 1)

        self.user_id_embedding = nn.Embedding(self.num_users, self.id_dim)
        self.item_id_embedding = nn.Embedding(self.num_items, self.id_dim)
        self.user_review_item_id_embedding = nn.Embedding(
            self.num_items + 1,
            self.id_dim,
            padding_idx=self.num_items,
        )
        self.item_review_user_id_embedding = nn.Embedding(
            self.num_users + 1,
            self.id_dim,
            padding_idx=self.num_users,
        )

        self.user_gru = nn.GRU(self.cnn_out_dim, self.gru_dim, batch_first=True)
        self.item_gru = nn.GRU(self.cnn_out_dim, self.gru_dim, batch_first=True)
        self.position_embedding = nn.Embedding(150, self.time_dim)
        self.relative_time_embedding = nn.Embedding(150, self.time_dim)
        self.user_content_query = nn.Linear(self.gru_dim, self.gru_dim)
        self.item_content_query = nn.Linear(self.gru_dim, self.gru_dim)
        self.user_temporal_fc = nn.Linear(self.time_dim, 1)
        self.item_temporal_fc = nn.Linear(self.time_dim, 1)
        self.beta = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        self.graph_user_embedding = nn.Embedding(self.num_users, self.graph_node_dim)
        self.graph_item_embedding = nn.Embedding(self.num_items, self.graph_node_dim)
        self.graph_review_projection = nn.Linear(self.cnn_out_dim, self.graph_attention_dim)
        self.graph_rating_projection = nn.Linear(5, self.graph_attention_dim)
        self.graph_src_projection = nn.Linear(self.graph_node_dim, self.graph_attention_dim)
        self.graph_dst_projection = nn.Linear(self.graph_node_dim, self.graph_attention_dim)
        self.graph_attention_heads = nn.ModuleList(
            [nn.Linear(self.graph_attention_dim, 1) for _ in range(self.n_heads)]
        )
        self.graph_message_projection = nn.Linear(
            self.graph_node_dim + self.graph_attention_dim,
            self.graph_hidden_dim,
        )
        self.graph_update = nn.Linear(
            self.graph_node_dim + self.graph_hidden_dim,
            self.graph_node_dim,
        )
        self.graph_output_projection = nn.Linear(self.graph_node_dim, self.graph_hidden_dim)

        user_fusion_in_dim = 0
        item_fusion_in_dim = 0
        if self.use_set_view:
            user_fusion_in_dim += self.cnn_out_dim
            item_fusion_in_dim += self.cnn_out_dim
        if self.use_sequence_view:
            user_fusion_in_dim += self.gru_dim
            item_fusion_in_dim += self.gru_dim
        if self.use_graph_view:
            user_fusion_in_dim += self.graph_hidden_dim
            item_fusion_in_dim += self.graph_hidden_dim
        if user_fusion_in_dim == 0 or item_fusion_in_dim == 0:
            raise ValueError("At least one SSG view must be active.")

        self.user_fusion = nn.Linear(user_fusion_in_dim, self.latent_dim)
        self.item_fusion = nn.Linear(item_fusion_in_dim, self.latent_dim)
        self.user_id_latent = nn.Linear(self.id_dim, self.latent_dim)
        self.item_id_latent = nn.Linear(self.id_dim, self.latent_dim)

        self.user_bias = nn.Embedding(self.num_users, 1)
        self.item_bias = nn.Embedding(self.num_items, 1)
        self.predict_layer = nn.Linear(self.latent_dim, 1)

        ratings = getattr(train_dataset, "ratings", None)
        if ratings is None:
            self.global_bias = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        else:
            ratings_tensor = torch.as_tensor(ratings, dtype=torch.float32)
            self.global_bias = nn.Parameter(ratings_tensor.mean().view(1))

        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.loss_fn = nn.MSELoss()
        self._cached_user_set = None
        self._cached_item_set = None
        self._cached_user_seq = None
        self._cached_item_seq = None
        self._cached_user_graph = None
        self._cached_item_graph = None
        self._shape_logged = False

        self.init_weights()

        total_params = sum(parameter.numel() for parameter in self.parameters())
        trainable_params = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        print("SSG parameters: total={}, trainable={}".format(total_params, trainable_params))

    def _get_int_config(self, key: str, default: int) -> int:
        return int(cast(Union[int, float, str], self.configs.get(key, default)))

    def _get_float_config(self, key: str, default: float) -> float:
        return float(cast(Union[int, float, str], self.configs.get(key, default)))

    def _get_bool_config(self, key: str, default: bool) -> bool:
        value = self.configs.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def _get_int_list_config(self, key: str, default: Sequence[int]) -> List[int]:
        value = self.configs.get(key, default)
        if isinstance(value, str):
            stripped = value.strip().strip("[]")
            if not stripped:
                return list(default)
            return [int(part.strip()) for part in stripped.split(",") if part.strip()]
        if isinstance(value, Sequence):
            return [int(cast(Union[int, float, str], part)) for part in value]
        return list(default)

    def init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding) and module is not self.word_embedding:
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].fill_(0.0)

    def _encode_review_tokens(
        self,
        review_tokens: torch.Tensor,
        convs: nn.ModuleList,
    ) -> torch.Tensor:
        if review_tokens.dim() != 3:
            raise ValueError("Expected review tokens with shape [B, R, L].")
        batch_size, review_count, review_length = review_tokens.shape
        if review_length != self.review_length:
            raise ValueError(
                "Expected review_length={}, got {}".format(self.review_length, review_length)
            )
        flattened = review_tokens.reshape(batch_size * review_count, review_length)
        embedded = cast(torch.Tensor, self.word_embedding(flattened)).transpose(1, 2)

        conv_outputs: List[torch.Tensor] = []
        for conv in convs:
            conv_out = cast(torch.Tensor, self.relu(conv(embedded)))
            pooled = cast(torch.Tensor, torch.amax(conv_out, dim=2))
            conv_outputs.append(pooled)
        combined = cast(torch.Tensor, torch.cat(conv_outputs, dim=1))
        return combined.reshape(batch_size, review_count, self.cnn_out_dim)

    def _encode_review_inputs(
        self,
        reviews: torch.Tensor,
        convs: nn.ModuleList,
        projection: nn.Linear,
    ) -> torch.Tensor:
        if self.review_input_mode == "embedding":
            if reviews.dim() != 3 or reviews.size(-1) != self.review_dim:
                raise ValueError(f"Expected review embeddings with shape [B, R, {self.review_dim}].")
            return cast(torch.Tensor, projection(reviews.float()))
        return self._encode_review_tokens(reviews.long(), convs)

    def _masked_id_attention(
        self,
        review_features: torch.Tensor,
        review_ids: torch.Tensor,
        review_tokens: torch.Tensor,
        review_fc: nn.Linear,
        id_embedding: nn.Embedding,
        id_fc: nn.Linear,
        attention_fc: nn.Linear,
        padding_value: int,
    ) -> torch.Tensor:
        safe_review_ids = review_ids.clamp(min=0, max=padding_value)
        review_id_features = cast(torch.Tensor, id_embedding(safe_review_ids))
        review_proj = cast(torch.Tensor, review_fc(review_features))
        id_proj = cast(torch.Tensor, id_fc(review_id_features))
        attention_logits = cast(torch.Tensor, attention_fc(self.relu(review_proj + id_proj)))

        review_mask = review_tokens.ne(self.pad_idx).any(dim=2)
        review_mask = review_mask & safe_review_ids.ne(padding_value)
        mask = review_mask.unsqueeze(-1)
        attention_logits = attention_logits.masked_fill(~mask, -1e9)
        attention = cast(torch.Tensor, torch.softmax(attention_logits, dim=1))
        attention = attention * mask.to(attention.dtype)
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return cast(torch.Tensor, torch.sum(attention * review_features, dim=1))

    def _build_sequence_mask(
        self,
        sequence_features: torch.Tensor,
        sequence_lengths: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, seq_len, _ = sequence_features.shape
        if sequence_lengths is not None:
            clipped_lengths = sequence_lengths.clamp(min=0, max=seq_len)
            positions = torch.arange(seq_len, device=sequence_features.device).unsqueeze(0)
            return positions < clipped_lengths.unsqueeze(1)
        return sequence_features.abs().sum(dim=2).gt(0)

    def _sequence_attention(
        self,
        sequence_features: torch.Tensor,
        sequence_lengths: Optional[torch.Tensor],
        pos_ind: Optional[torch.Tensor],
        rel_dt: Optional[torch.Tensor],
        gru: nn.GRU,
        query_projection: nn.Linear,
        temporal_fc: nn.Linear,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = sequence_features.shape
        if seq_len == 0:
            return sequence_features.new_zeros(batch_size, self.gru_dim)

        gru_outputs, hidden = gru(sequence_features)
        last_hidden = hidden[-1]
        content_query = cast(torch.Tensor, query_projection(last_hidden)).unsqueeze(2)
        content_attn = cast(
            torch.Tensor,
            torch.bmm(gru_outputs, content_query).squeeze(2) / (self.gru_dim ** 0.5),
        )

        if pos_ind is None:
            pos_ind = torch.arange(seq_len, device=sequence_features.device).unsqueeze(0).expand(batch_size, -1)
        if rel_dt is None:
            rel_dt = torch.zeros(batch_size, seq_len, dtype=torch.long, device=sequence_features.device)
        pos_ind = pos_ind.clamp(min=0, max=149)
        rel_dt = rel_dt.clamp(min=0, max=149)

        temporal_emb = cast(torch.Tensor, self.position_embedding(pos_ind))
        temporal_emb = temporal_emb + cast(torch.Tensor, self.relative_time_embedding(rel_dt))
        temporal_attn = cast(torch.Tensor, temporal_fc(self.leaky_relu(temporal_emb)).squeeze(-1))

        attention_logits = content_attn + self.beta * temporal_attn
        mask = self._build_sequence_mask(gru_outputs, sequence_lengths)
        attention_logits = attention_logits.masked_fill(~mask, -1e9)
        attention = cast(torch.Tensor, torch.softmax(attention_logits, dim=1))
        attention = attention * mask.to(attention.dtype)
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return cast(torch.Tensor, torch.sum(attention.unsqueeze(-1) * gru_outputs, dim=1))

    def _pool_graph_reviews(self, graph_reviews: torch.Tensor) -> torch.Tensor:
        if self.review_input_mode == "embedding":
            if graph_reviews.dim() != 2 or graph_reviews.size(1) != self.review_dim:
                raise ValueError(f"Expected graph review embeddings with shape [E, {self.review_dim}].")
            return cast(torch.Tensor, self.user_embedding_projection(graph_reviews.float()))
        if graph_reviews.dim() == 2:
            graph_reviews = graph_reviews.unsqueeze(1)
        if graph_reviews.dim() != 3:
            raise ValueError("Expected graph_reviews with shape [E, L] or [E, R, L].")
        edge_features = self._encode_review_tokens(graph_reviews, self.user_convs)
        return cast(torch.Tensor, edge_features.mean(dim=1))

    def _graph_encoder(
        self,
        user_id: torch.Tensor,
        item_id: torch.Tensor,
        graph_adj: Optional[torch.Tensor],
        graph_reviews: Optional[torch.Tensor],
        graph_ratings: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = user_id.size(0)
        zero_user = self.user_id_embedding.weight.new_zeros(batch_size, self.graph_hidden_dim)
        zero_item = self.item_id_embedding.weight.new_zeros(batch_size, self.graph_hidden_dim)
        if not self.use_graph_view or graph_adj is None or graph_reviews is None or graph_ratings is None:
            return zero_user, zero_item

        if graph_adj.dim() == 2:
            graph_adj = graph_adj.unsqueeze(0).expand(batch_size, -1, -1)
        if graph_reviews.dim() == 2:
            graph_reviews = graph_reviews.unsqueeze(0).expand(batch_size, -1, -1)
        if graph_ratings.dim() == 1:
            graph_ratings = graph_ratings.unsqueeze(0).expand(batch_size, -1)
        if graph_adj.dim() != 3 or graph_adj.size(0) != batch_size or graph_adj.size(1) != 2:
            raise ValueError("Expected graph_adj with shape [B, 2, E] or [2, E].")
        if graph_reviews.dim() != 3 or graph_reviews.size(0) != batch_size:
            raise ValueError("Expected graph_reviews with shape [B, E, L] or [E, L].")
        if graph_ratings.dim() != 2 or graph_ratings.size(0) != batch_size:
            raise ValueError("Expected graph_ratings with shape [B, E] or [E].")

        user_outputs: List[torch.Tensor] = []
        item_outputs: List[torch.Tensor] = []

        for sample_idx in range(batch_size):
            sample_adj = graph_adj[sample_idx]
            sample_reviews = graph_reviews[sample_idx]
            sample_ratings = graph_ratings[sample_idx]
            valid_mask = sample_adj[0].ge(0) & sample_adj[1].ge(0) & sample_ratings.gt(0)
            if not bool(valid_mask.any()):
                user_outputs.append(zero_user[sample_idx])
                item_outputs.append(zero_item[sample_idx])
                continue

            src_users = sample_adj[0, valid_mask].clamp(min=0, max=self.num_users - 1)
            dst_items = sample_adj[1, valid_mask].clamp(min=0, max=self.num_items - 1)
            edge_reviews = self._pool_graph_reviews(sample_reviews[valid_mask])

            rating_index = sample_ratings[valid_mask].long().clamp(min=1, max=5) - 1
            edge_rating_one_hot = F.one_hot(rating_index, num_classes=5).float()

            user_nodes = cast(torch.Tensor, self.graph_user_embedding.weight)
            item_nodes = cast(torch.Tensor, self.graph_item_embedding.weight)
            node_states = torch.cat([user_nodes, item_nodes], dim=0)

            src_node_index = src_users
            dst_node_index = self.num_users + dst_items
            bidir_src = torch.cat([src_node_index, dst_node_index], dim=0)
            bidir_dst = torch.cat([dst_node_index, src_node_index], dim=0)

            edge_review_proj = cast(torch.Tensor, self.graph_review_projection(edge_reviews))
            edge_rating_proj = cast(torch.Tensor, self.graph_rating_projection(edge_rating_one_hot))
            edge_context = self.leaky_relu(edge_review_proj + edge_rating_proj)
            edge_context = torch.cat([edge_context, edge_context], dim=0)

            for _ in range(self.n_hops):
                src_states = node_states[bidir_src]
                dst_states = node_states[bidir_dst]
                attention_hidden = self.leaky_relu(
                    cast(torch.Tensor, self.graph_src_projection(src_states))
                    + cast(torch.Tensor, self.graph_dst_projection(dst_states))
                    + edge_context
                )

                head_scores: List[torch.Tensor] = []
                for head in self.graph_attention_heads:
                    head_scores.append(cast(torch.Tensor, head(attention_hidden)))
                attention_scores = torch.mean(torch.cat(head_scores, dim=1), dim=1)
                attention_scores = torch.sigmoid(attention_scores)

                message_inputs = torch.cat([src_states, edge_context], dim=1)
                messages = cast(torch.Tensor, self.graph_message_projection(message_inputs))
                messages = attention_scores.unsqueeze(1) * messages

                aggregated = messages.new_zeros(node_states.size(0), self.graph_hidden_dim)
                aggregated.index_add_(0, bidir_dst, messages)
                updated = cast(torch.Tensor, self.graph_update(torch.cat([node_states, aggregated], dim=1)))
                node_states = self.dropout(self.relu(updated))

            sample_user = user_id[sample_idx].clamp(min=0, max=self.num_users - 1)
            sample_item = item_id[sample_idx].clamp(min=0, max=self.num_items - 1)
            user_outputs.append(cast(torch.Tensor, self.graph_output_projection(node_states[sample_user])))
            item_outputs.append(cast(torch.Tensor, self.graph_output_projection(node_states[self.num_users + sample_item])))

        return torch.stack(user_outputs, dim=0), torch.stack(item_outputs, dim=0)

    def _decov(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.size(0) == 0:
            return x.new_zeros(())
        x_centered = x - x.mean(dim=0, keepdim=True)
        y_centered = y - y.mean(dim=0, keepdim=True)
        cov = torch.matmul(x_centered.transpose(0, 1), y_centered) / float(x.size(0))
        return 0.5 * cov.pow(2).sum()

    def _collect_decov_loss(
        self,
        set_view: Optional[torch.Tensor],
        seq_view: Optional[torch.Tensor],
        graph_view: Optional[torch.Tensor],
    ) -> torch.Tensor:
        loss = self.global_bias.new_zeros(())
        active_views = [
            view
            for view in [set_view, seq_view, graph_view]
            if view is not None
        ]
        for first_index in range(len(active_views)):
            for second_index in range(first_index + 1, len(active_views)):
                loss = loss + self._decov(active_views[first_index], active_views[second_index])
        return cast(torch.Tensor, loss)

    def _assert_and_log_shapes(
        self,
        user_review: torch.Tensor,
        user_seq_reviews: Optional[torch.Tensor],
        graph_adj: Optional[torch.Tensor],
        graph_reviews: Optional[torch.Tensor],
    ) -> None:
        if user_review.dim() != 3:
            raise ValueError("Expected user_review with shape [B, R, L].")
        if user_seq_reviews is not None and user_seq_reviews.dim() != 3:
            raise ValueError("Expected user_seq_reviews with shape [B, S, L].")
        expected_last_dim = self.review_dim if self.review_input_mode == "embedding" else self.review_length
        if user_review.size(-1) != expected_last_dim:
            raise ValueError(f"Expected user_review last dim {expected_last_dim}, got {user_review.size(-1)}.")
        if user_seq_reviews is not None and user_seq_reviews.size(-1) != expected_last_dim:
            raise ValueError(f"Expected user_seq_reviews last dim {expected_last_dim}, got {user_seq_reviews.size(-1)}.")
        if graph_adj is not None and (graph_adj.dim() != 3 or graph_adj.size(1) != 2):
            raise ValueError("Expected graph_adj with shape [B, 2, E].")
        if graph_reviews is not None and graph_reviews.dim() != 3:
            raise ValueError("Expected graph_reviews with shape [B, E, L].")

        debug_shapes = self._get_bool_config("debug_shapes", False)
        if debug_shapes and not self._shape_logged:
            print(
                "SSG shape check:",
                {
                    "user_review": tuple(user_review.shape),
                    "user_seq_reviews": None if user_seq_reviews is None else tuple(user_seq_reviews.shape),
                    "graph_adj": None if graph_adj is None else tuple(graph_adj.shape),
                    "graph_reviews": None if graph_reviews is None else tuple(graph_reviews.shape),
                },
            )
            self._shape_logged = True

    def _l2_regularization(self) -> torch.Tensor:
        reg = self.global_bias.new_zeros(())
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad or name == "word_embedding.weight":
                continue
            reg = reg + parameter.pow(2).sum()
        return cast(torch.Tensor, reg)

    def forward(
        self,
        user_id: torch.Tensor,
        item_id: torch.Tensor,
        user_review: torch.Tensor,
        item_review: torch.Tensor,
        user_review_item_ids: torch.Tensor,
        item_review_user_ids: torch.Tensor,
        user_seq_reviews: Optional[torch.Tensor] = None,
        item_seq_reviews: Optional[torch.Tensor] = None,
        user_seq_len: Optional[torch.Tensor] = None,
        item_seq_len: Optional[torch.Tensor] = None,
        user_pos_ind: Optional[torch.Tensor] = None,
        item_pos_ind: Optional[torch.Tensor] = None,
        user_rel_dt: Optional[torch.Tensor] = None,
        item_rel_dt: Optional[torch.Tensor] = None,
        user_abs_dt: Optional[torch.Tensor] = None,
        item_abs_dt: Optional[torch.Tensor] = None,
        graph_adj: Optional[torch.Tensor] = None,
        graph_reviews: Optional[torch.Tensor] = None,
        graph_ratings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del user_abs_dt, item_abs_dt
        self._assert_and_log_shapes(user_review, user_seq_reviews, graph_adj, graph_reviews)

        user_review_features = self._encode_review_inputs(user_review, self.user_convs, self.user_embedding_projection)
        item_review_features = self._encode_review_inputs(item_review, self.item_convs, self.item_embedding_projection)

        user_set = self._masked_id_attention(
            review_features=user_review_features,
            review_ids=user_review_item_ids,
            review_tokens=user_review,
            review_fc=self.user_review_fc,
            id_embedding=self.user_review_item_id_embedding,
            id_fc=self.user_id_attention_fc,
            attention_fc=self.user_attention_fc,
            padding_value=self.num_items,
        )
        item_set = self._masked_id_attention(
            review_features=item_review_features,
            review_ids=item_review_user_ids,
            review_tokens=item_review,
            review_fc=self.item_review_fc,
            id_embedding=self.item_review_user_id_embedding,
            id_fc=self.item_id_attention_fc,
            attention_fc=self.item_attention_fc,
            padding_value=self.num_users,
        )

        if user_seq_reviews is not None:
            user_seq_features = self._encode_review_inputs(user_seq_reviews, self.user_convs, self.user_embedding_projection)
        else:
            user_seq_features = user_review_features[:, : min(user_review_features.size(1), self.seq_count), :]
        if item_seq_reviews is not None:
            item_seq_features = self._encode_review_inputs(item_seq_reviews, self.item_convs, self.item_embedding_projection)
        else:
            item_seq_features = item_review_features[:, : min(item_review_features.size(1), self.seq_count), :]
        if user_pos_ind is not None:
            user_pos_ind = user_pos_ind[:, : user_seq_features.size(1)]
        if item_pos_ind is not None:
            item_pos_ind = item_pos_ind[:, : item_seq_features.size(1)]
        if user_rel_dt is not None:
            user_rel_dt = user_rel_dt[:, : user_seq_features.size(1)]
        if item_rel_dt is not None:
            item_rel_dt = item_rel_dt[:, : item_seq_features.size(1)]

        user_seq = self._sequence_attention(
            sequence_features=user_seq_features,
            sequence_lengths=user_seq_len,
            pos_ind=user_pos_ind,
            rel_dt=user_rel_dt,
            gru=self.user_gru,
            query_projection=self.user_content_query,
            temporal_fc=self.user_temporal_fc,
        )
        item_seq = self._sequence_attention(
            sequence_features=item_seq_features,
            sequence_lengths=item_seq_len,
            pos_ind=item_pos_ind,
            rel_dt=item_rel_dt,
            gru=self.item_gru,
            query_projection=self.item_content_query,
            temporal_fc=self.item_temporal_fc,
        )

        user_graph, item_graph = self._graph_encoder(
            user_id=user_id,
            item_id=item_id,
            graph_adj=graph_adj,
            graph_reviews=graph_reviews,
            graph_ratings=graph_ratings,
        )

        user_views: List[torch.Tensor] = []
        item_views: List[torch.Tensor] = []
        self._cached_user_set = user_set if self.use_set_view else None
        self._cached_item_set = item_set if self.use_set_view else None
        self._cached_user_seq = user_seq if self.use_sequence_view else None
        self._cached_item_seq = item_seq if self.use_sequence_view else None
        self._cached_user_graph = user_graph if self.use_graph_view else None
        self._cached_item_graph = item_graph if self.use_graph_view else None

        if self.use_set_view:
            user_views.append(user_set)
            item_views.append(item_set)
        if self.use_sequence_view:
            user_views.append(user_seq)
            item_views.append(item_seq)
        if self.use_graph_view:
            user_views.append(user_graph)
            item_views.append(item_graph)

        user_latent = cast(torch.Tensor, self.user_fusion(self.dropout(torch.cat(user_views, dim=1))))
        item_latent = cast(torch.Tensor, self.item_fusion(self.dropout(torch.cat(item_views, dim=1))))

        user_latent = user_latent + cast(torch.Tensor, self.user_id_latent(self.user_id_embedding(user_id)))
        item_latent = item_latent + cast(torch.Tensor, self.item_id_latent(self.item_id_embedding(item_id)))

        interaction = self.relu(user_latent * item_latent)
        pred = cast(torch.Tensor, self.predict_layer(self.dropout(interaction)))
        pred = pred + cast(torch.Tensor, self.user_bias(user_id))
        pred = pred + cast(torch.Tensor, self.item_bias(item_id))
        pred = pred + self.global_bias
        return pred

    def cal_loss(
        self,
        batch_data: Union[Tuple[torch.Tensor, ...], Mapping[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if isinstance(batch_data, Mapping):
            user_id = batch_data["user_id"]
            item_id = batch_data["item_id"]
            user_review = batch_data["user_review"]
            item_review = batch_data["item_review"]
            user_review_item_ids = batch_data["user_review_item_ids"]
            item_review_user_ids = batch_data["item_review_user_ids"]
            user_seq_reviews = batch_data.get("user_seq_reviews")
            item_seq_reviews = batch_data.get("item_seq_reviews")
            user_seq_len = batch_data.get("user_seq_len")
            item_seq_len = batch_data.get("item_seq_len")
            user_pos_ind = batch_data.get("user_pos_ind")
            item_pos_ind = batch_data.get("item_pos_ind")
            user_rel_dt = batch_data.get("user_rel_dt")
            item_rel_dt = batch_data.get("item_rel_dt")
            user_abs_dt = batch_data.get("user_abs_dt")
            item_abs_dt = batch_data.get("item_abs_dt")
            graph_adj = batch_data.get("graph_adj")
            graph_reviews = batch_data.get("graph_reviews")
            graph_ratings = batch_data.get("graph_ratings")
            ratings = batch_data["rating"]
        else:
            if len(batch_data) < 7:
                raise ValueError("Expected at least 7 tensors in batch_data.")
            user_id, item_id, user_review, item_review, user_review_item_ids, item_review_user_ids = batch_data[:6]
            ratings = batch_data[-1]
            optional_tensors = cast(List[Optional[torch.Tensor]], list(batch_data[6:-1]))
            while len(optional_tensors) < 13:
                optional_tensors.append(None)
            (
                user_seq_reviews,
                item_seq_reviews,
                user_seq_len,
                item_seq_len,
                user_pos_ind,
                item_pos_ind,
                user_rel_dt,
                item_rel_dt,
                user_abs_dt,
                item_abs_dt,
                graph_adj,
                graph_reviews,
                graph_ratings,
            ) = optional_tensors[:13]

        predictions = self.forward(
            user_id=user_id,
            item_id=item_id,
            user_review=user_review,
            item_review=item_review,
            user_review_item_ids=user_review_item_ids,
            item_review_user_ids=item_review_user_ids,
            user_seq_reviews=user_seq_reviews,
            item_seq_reviews=item_seq_reviews,
            user_seq_len=cast(Optional[torch.Tensor], user_seq_len),
            item_seq_len=cast(Optional[torch.Tensor], item_seq_len),
            user_pos_ind=cast(Optional[torch.Tensor], user_pos_ind),
            item_pos_ind=cast(Optional[torch.Tensor], item_pos_ind),
            user_rel_dt=cast(Optional[torch.Tensor], user_rel_dt),
            item_rel_dt=cast(Optional[torch.Tensor], item_rel_dt),
            user_abs_dt=cast(Optional[torch.Tensor], user_abs_dt),
            item_abs_dt=cast(Optional[torch.Tensor], item_abs_dt),
            graph_adj=cast(Optional[torch.Tensor], graph_adj),
            graph_reviews=cast(Optional[torch.Tensor], graph_reviews),
            graph_ratings=cast(Optional[torch.Tensor], graph_ratings),
        )

        mse_loss = self.loss_fn(predictions, ratings.view(-1, 1).float())
        l2_loss = self._l2_regularization()
        decov_loss = self._collect_decov_loss(
            cast(Optional[torch.Tensor], self._cached_user_set),
            cast(Optional[torch.Tensor], self._cached_user_seq),
            cast(Optional[torch.Tensor], self._cached_user_graph),
        ) + self._collect_decov_loss(
            cast(Optional[torch.Tensor], self._cached_item_set),
            cast(Optional[torch.Tensor], self._cached_item_seq),
            cast(Optional[torch.Tensor], self._cached_item_graph),
        )
        total_loss = mse_loss + self.l2_lambda * l2_loss + self.decov_lambda * decov_loss

        loss_dict = {
            "mse_loss": float(mse_loss.detach().item()),
            "l2_loss": float(l2_loss.detach().item()),
            "decov_loss": float(decov_loss.detach().item()),
            "total_loss": float(total_loss.detach().item()),
        }
        return total_loss, loss_dict

    def predict_scores(
        self,
        user_id: torch.Tensor,
        item_id: torch.Tensor,
        user_review: torch.Tensor,
        item_review: torch.Tensor,
        user_review_item_ids: torch.Tensor,
        item_review_user_ids: torch.Tensor,
        user_seq_reviews: Optional[torch.Tensor] = None,
        item_seq_reviews: Optional[torch.Tensor] = None,
        user_seq_len: Optional[torch.Tensor] = None,
        item_seq_len: Optional[torch.Tensor] = None,
        user_pos_ind: Optional[torch.Tensor] = None,
        item_pos_ind: Optional[torch.Tensor] = None,
        user_rel_dt: Optional[torch.Tensor] = None,
        item_rel_dt: Optional[torch.Tensor] = None,
        user_abs_dt: Optional[torch.Tensor] = None,
        item_abs_dt: Optional[torch.Tensor] = None,
        graph_adj: Optional[torch.Tensor] = None,
        graph_reviews: Optional[torch.Tensor] = None,
        graph_ratings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward(
            user_id=user_id,
            item_id=item_id,
            user_review=user_review,
            item_review=item_review,
            user_review_item_ids=user_review_item_ids,
            item_review_user_ids=item_review_user_ids,
            user_seq_reviews=user_seq_reviews,
            item_seq_reviews=item_seq_reviews,
            user_seq_len=user_seq_len,
            item_seq_len=item_seq_len,
            user_pos_ind=user_pos_ind,
            item_pos_ind=item_pos_ind,
            user_rel_dt=user_rel_dt,
            item_rel_dt=item_rel_dt,
            user_abs_dt=user_abs_dt,
            item_abs_dt=item_abs_dt,
            graph_adj=graph_adj,
            graph_reviews=graph_reviews,
            graph_ratings=graph_ratings,
        )
