import torch
import torch.nn as nn
from .abstract import AbstractRec

class NeuMF(AbstractRec):
    def __init__(self, configs, train_dataset):
        super().__init__()
        self.configs = configs
        self.train_dataset = train_dataset

        self.num_users = train_dataset.num_users
        self.num_items = train_dataset.num_items
        self.mf_embedding_size = configs.get("mf_embedding_size", configs["embedding_size"])
        self.mlp_embedding_size = configs.get("mlp_embedding_size", configs["embedding_size"])
        self.mlp_hidden_size = configs.get("mlp_hidden_size", [self.mlp_embedding_size, self.mlp_embedding_size])

        self.user_mf_embedding = nn.Embedding(self.num_users, self.mf_embedding_size)
        self.item_mf_embedding = nn.Embedding(self.num_items, self.mf_embedding_size)
        self.user_mlp_embedding = nn.Embedding(self.num_users, self.mlp_embedding_size)
        self.item_mlp_embedding = nn.Embedding(self.num_items, self.mlp_embedding_size)

        self.dropout_prob = configs.get("dropout_prob", 0.2)
        self.embedding_dropout = nn.Dropout(p=self.dropout_prob)
        
        mlp_layers = []
        input_dim = self.mlp_embedding_size * 2
        for i, hidden_dim in enumerate(self.mlp_hidden_size):
            mlp_layers.append(nn.Linear(input_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(p=self.dropout_prob))
            input_dim = hidden_dim
        self.mlp_layers = nn.Sequential(*mlp_layers)

        mlp_output_dim = input_dim
        self.predict_layer = nn.Linear(self.mf_embedding_size + mlp_output_dim, 1)

        self.loss_fn = nn.MSELoss()

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.user_mf_embedding.weight)
        nn.init.xavier_normal_(self.item_mf_embedding.weight)
        nn.init.xavier_normal_(self.user_mlp_embedding.weight)
        nn.init.xavier_normal_(self.item_mlp_embedding.weight)

        for module in self.mlp_layers:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.xavier_normal_(self.predict_layer.weight)
        if self.predict_layer.bias is not None:
            nn.init.zeros_(self.predict_layer.bias)

    def forward(self, *args, **kwargs):
        if args:
            user_ids, item_ids = args[:2]
        else:
            user_ids = kwargs["user_ids"]
            item_ids = kwargs["item_ids"]
        user_mf_e = self.embedding_dropout(self.user_mf_embedding(user_ids))
        item_mf_e = self.embedding_dropout(self.item_mf_embedding(item_ids))
        mf_output = user_mf_e * item_mf_e

        user_mlp_e = self.embedding_dropout(self.user_mlp_embedding(user_ids))
        item_mlp_e = self.embedding_dropout(self.item_mlp_embedding(item_ids))
        mlp_input = torch.cat([user_mlp_e, item_mlp_e], dim=-1)
        mlp_output = self.mlp_layers(mlp_input)

        final_input = torch.cat([mf_output, mlp_output], dim=-1)
        rating_pred = self.predict_layer(final_input).squeeze(-1)
        return rating_pred

    def cal_loss(self, *args, **kwargs):
        batch_data = args[0] if args else kwargs["batch_data"]
        user_ids, item_ids, ratings = batch_data

        rating_pred = self.forward(user_ids, item_ids)
        mse_loss = self.loss_fn(rating_pred, ratings)

        loss_dict = {
            'mse_loss': mse_loss.item(),
            'total_loss': mse_loss.item()
        }

        return mse_loss, loss_dict

    def predict_scores(self, *args, **kwargs):
        user_ids = args[0] if args else kwargs["user_ids"]
        user_mf_e = self.user_mf_embedding(user_ids)
        user_mlp_e = self.user_mlp_embedding(user_ids)

        batch_size = user_ids.size(0)
        all_item_mf_e = self.item_mf_embedding.weight
        all_item_mlp_e = self.item_mlp_embedding.weight

        mf_output = user_mf_e.unsqueeze(1) * all_item_mf_e.unsqueeze(0)

        user_mlp_expand = user_mlp_e.unsqueeze(1).expand(batch_size, self.num_items, self.mlp_embedding_size)
        item_mlp_expand = all_item_mlp_e.unsqueeze(0).expand(batch_size, self.num_items, self.mlp_embedding_size)
        mlp_input = torch.cat([user_mlp_expand, item_mlp_expand], dim=-1)
        mlp_output = self.mlp_layers(mlp_input.reshape(batch_size * self.num_items, -1)).reshape(batch_size, self.num_items, -1)

        final_input = torch.cat([mf_output, mlp_output], dim=-1)
        scores = self.predict_layer(final_input.reshape(batch_size * self.num_items, -1)).squeeze(-1)
        scores = scores.reshape(batch_size, self.num_items)
        return scores
