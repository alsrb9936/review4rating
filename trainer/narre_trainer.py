"""NARRE trainer."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportMissingParameterType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportImplicitOverride=false, reportUnannotatedClassAttribute=false, reportUnusedCallResult=false

import json
import os

from tqdm import tqdm
import torch

from .base_trainer import BaseTrainer
from metric import print_results


class NARRETrainer(BaseTrainer):
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
            user_review = batch["user_review"].to(self.device)
            item_review = batch["item_review"].to(self.device)
            user_review_item_ids = batch["user_review_item_ids"].to(self.device)
            item_review_user_ids = batch["item_review_user_ids"].to(self.device)
            ratings = batch["rating"].to(self.device)

            batch_data = (
                user_id,
                item_id,
                user_review,
                item_review,
                user_review_item_ids,
                item_review_user_ids,
                ratings,
            )
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
        user_review = batch["user_review"].to(self.device)
        item_review = batch["item_review"].to(self.device)
        user_review_item_ids = batch["user_review_item_ids"].to(self.device)
        item_review_user_ids = batch["item_review_user_ids"].to(self.device)
        return self.model.forward(
            user_id,
            item_id,
            user_review,
            item_review,
            user_review_item_ids,
            item_review_user_ids,
        )

    def train(self):
        print(f"Starting training for {self.epoch} epochs")
        print(f"Model: {self.model_name}")
        print(f"Dataset: {self.dataset_name}")
        print(f"Device: {self.device}")
        print(f"Best metric: {self.best_metric_name}")

        for epoch in range(self.epoch):
            train_loss_dict = self.train_epoch(epoch)

            log_entry = {
                "epoch": epoch + 1,
                "train": train_loss_dict,
            }

            print(f"Epoch {epoch + 1}/{self.epoch} - Total Loss: {train_loss_dict.get('total_loss', 0):.4f}")

            if (epoch + 1) % self.eval_step == 0:
                valid_metrics = self.evaluate(self.valid_dataloader, phase="valid")
                log_entry["valid"] = valid_metrics
                current_metric = valid_metrics.get(self.best_metric_name, float("inf"))

                print("  Validation Metrics:")
                print_results(valid_metrics)

                if current_metric < self.best_valid_metric:
                    self.best_valid_metric = current_metric
                    self.patience_counter = 0
                    self.save_checkpoint("best_model.pt")
                    print(f"  New Best! {self.best_metric_name}: {current_metric:.4f}")
                else:
                    self.patience_counter += 1
                    print(f"  Patience: {self.patience_counter}/{self.early_stop_patience}")

                if self.patience_counter >= self.early_stop_patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    self.train_log.append(log_entry)
                    self.scheduler.step()
                    break

            self.train_log.append(log_entry)
            self.scheduler.step()

        print("Training completed!")
        self.save_checkpoint("final_model.pt")
        with open(os.path.join(self.result_path, "training_log.json"), "w") as f:
            json.dump(self.train_log, f, indent=2)

        return self.best_valid_metric
