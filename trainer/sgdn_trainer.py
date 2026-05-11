from tqdm import tqdm
import numpy as np

from .base_trainer import BaseTrainer


class SGDNTrainer(BaseTrainer):
    def __init__(self, model, train_dataloader, valid_dataloader, test_dataloader, configs):
        super().__init__(model, train_dataloader, valid_dataloader, test_dataloader, configs)

    def train_epoch(self, epoch_idx):
        self.model.train()
        epoch_loss_dict = None

        for batch in tqdm(self.train_dataloader, desc=f"Epoch {epoch_idx + 1}"):
            loss, loss_dict = self.model.cal_loss(batch)

            if epoch_loss_dict is None:
                epoch_loss_dict = {key: 0.0 for key in loss_dict}

            self.optimizer.zero_grad()
            loss.backward()
            grad_clip = self.configs.get("grad_clip", 1.0)
            if grad_clip is not None:
                import torch

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.optimizer.step()

            for key, value in loss_dict.items():
                epoch_loss_dict[key] += value

        if epoch_loss_dict is None:
            return {}

        num_batches = max(len(self.train_dataloader), 1)
        return {key: value / num_batches for key, value in epoch_loss_dict.items()}

    def evaluate(self, dataloader, phase="valid"):
        self.model.eval()
        all_predictions = []
        all_ratings = []

        for batch in dataloader:
            predictions = self.model.predict_ratings(batch)
            ratings = batch["ratings"]
            all_predictions.append(predictions.detach().cpu().numpy())
            all_ratings.append(ratings.cpu().numpy())

        predictions = np.concatenate(all_predictions)
        ratings = np.concatenate(all_ratings)

        return self._build_metrics(predictions, ratings)

    def _predict_batch(self, batch):
        return self.model.predict_ratings(batch)
