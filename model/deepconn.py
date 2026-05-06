import torch
import torch.nn as nn

from .abstract import AbstractRec


class DeepCoNN(AbstractRec):
    def __init__(self, configs, train_dataset):
        super().__init__()
        self.configs = configs
        self.train_dataset = train_dataset

        self.review_length = int(configs.get("review_length", 40))
        self.review_count = int(configs.get("review_count", 10))
        self.word_dim = int(configs.get("word_dim", 50))
        self.kernel_count = int(configs.get("kernel_count", 100))
        self.kernel_size = int(configs.get("kernel_size", 3))
        self.latent_dim = int(configs.get("cnn_out_dim", 50))
        self.dropout_prob = float(configs.get("dropout_prob", 0.5))
        self.pad_idx = int(getattr(train_dataset, "pad_idx", 0))

        embedding_weight = train_dataset.embedding_matrix
        if embedding_weight.size(1) != self.word_dim:
            raise ValueError(
                f"Configured word_dim={self.word_dim} does not match "
                f"loaded embedding dim={embedding_weight.size(1)}"
            )

        self.embedding = nn.Embedding.from_pretrained(
            embedding_weight,
            freeze=True,
            padding_idx=self.pad_idx,
        )

        # user branch
        self.user_conv = nn.Conv1d(
            in_channels=self.word_dim,
            out_channels=self.kernel_count,
            kernel_size=self.kernel_size,
            padding=(self.kernel_size - 1) // 2,
        )
        self.user_fc = nn.Linear(self.kernel_count, self.latent_dim)

        # item branch
        self.item_conv = nn.Conv1d(
            in_channels=self.word_dim,
            out_channels=self.kernel_count,
            kernel_size=self.kernel_size,
            padding=(self.kernel_size - 1) // 2,
        )
        self.item_fc = nn.Linear(self.kernel_count, self.latent_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_prob)

        # final predictor
        self.predict_layer = nn.Linear(self.latent_dim * 2, 1)

        self.loss_fn = nn.MSELoss()

    def _encode_reviews(self, review_tokens, conv, fc):
        """
        review_tokens: [B, R, L]
        1) flatten reviews -> [B, R*L]
        2) embedding -> [B, R*L, D]
        3) conv1d input -> [B, D, R*L]
        4) conv + global max pool -> [B, K]
        5) fc -> [B, latent_dim]
        """
        batch_size = review_tokens.size(0)

        x = review_tokens.reshape(batch_size, self.review_count * self.review_length)
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.relu(conv(x))

        x = torch.amax(x, dim=2)

        x = self.dropout(x)
        x = self.relu(fc(x))
        x = self.dropout(x)

        return x  # [B, latent_dim]

    def forward(self, *args, **kwargs):
        if args:
            user_review, item_review = args[:2]
        else:
            user_review = kwargs["user_review"]
            item_review = kwargs["item_review"]

        user_latent = self._encode_reviews(user_review, self.user_conv, self.user_fc)
        item_latent = self._encode_reviews(item_review, self.item_conv, self.item_fc)

        x = torch.cat([user_latent, item_latent], dim=1)

        # [B, 2*latent_dim] -> [B, 1]
        return self.predict_layer(x)

    def cal_loss(self, *args, **kwargs):
        batch_data = args[0] if args else kwargs["batch_data"]
        user_review, item_review, ratings = batch_data

        predictions = self.forward(user_review, item_review)
        mse_loss = self.loss_fn(predictions, ratings.view(-1, 1))

        loss_dict = {
            "mse_loss": float(mse_loss.detach().item()),
            "total_loss": float(mse_loss.detach().item()),
        }
        return mse_loss, loss_dict

    def predict_scores(self, *args, **kwargs):
        if args:
            user_review, item_review = args[:2]
        else:
            user_review = kwargs["user_review"]
            item_review = kwargs["item_review"]
        return self.forward(user_review, item_review)