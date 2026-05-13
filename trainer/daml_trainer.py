import json
import os

from tqdm import tqdm
import torch

from .base_trainer import BaseTrainer
from metric import print_results


class DAMLTrainer(BaseTrainer):
    def __init__(self, model, train_dataloader, valid_dataloader, test_dataloader, configs):
        super().__init__(model, train_dataloader, valid_dataloader, test_dataloader, configs)
        self.lr_decay = float(configs.get("gamma", configs.get("lr_decay", 0.95)))
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lr_decay)

    def train_epoch(self, epoch_idx):
        self.model.train()
        epoch_loss_dict = None

        for batch in tqdm(self.train_dataloader, desc=f"Epoch {epoch_idx + 1}"):
            user_id = batch["user_id"].to(self.device)
            item_id = batch["item_id"].to(self.device)
            user_doc = batch["user_doc"].to(self.device)
            item_doc = batch["item_doc"].to(self.device)
            ratings = batch["rating"].to(self.device)

            batch_data = (user_id, item_id, user_doc, item_doc, ratings)
            loss, loss_dict = self.model.cal_loss(batch_data)

            if epoch_loss_dict is None:
                epoch_loss_dict = {key: 0.0 for key in loss_dict}

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            for key in epoch_loss_dict:
                epoch_loss_dict[key] += loss_dict[key]

        if epoch_loss_dict is None:
            return {}

        for key in epoch_loss_dict:
            epoch_loss_dict[key] /= len(self.train_dataloader)

        return epoch_loss_dict

    def _predict_batch(self, batch):
        user_id = batch["user_id"].to(self.device)
        item_id = batch["item_id"].to(self.device)
        user_doc = batch["user_doc"].to(self.device)
        item_doc = batch["item_doc"].to(self.device)
        return self.model.forward(user_id, item_id, user_doc, item_doc)
