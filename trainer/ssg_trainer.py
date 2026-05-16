"""SSG trainer."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportMissingParameterType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportImplicitOverride=false, reportUnannotatedClassAttribute=false, reportUnusedCallResult=false

import json
import os
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .base_trainer import BaseTrainer
from metric import print_results


class SSGTrainer(BaseTrainer):
    def __init__(self, model, train_dataloader, valid_dataloader, test_dataloader, configs):
        super().__init__(model, train_dataloader, valid_dataloader, test_dataloader, configs)
        self.gamma = float(configs.get("gamma", configs.get("lr_decay", 0.95)))
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.gamma)

    def _prepare_inputs(self, batch):
        user_id = batch["user_id"].to(self.device)
        item_id = batch["item_id"].to(self.device)
        ratings = batch["rating"].to(self.device)
        input_u = batch["input_u"].to(self.device)
        input_i = batch["input_i"].to(self.device)
        reuid = batch["reuid"].to(self.device)
        reiid = batch["reiid"].to(self.device)
        nodes = batch["nodes"].to(self.device)
        reviews = batch["reviews"].to(self.device)
        graph_ratings = batch["ratings"].to(self.device)
        adj = batch["adj"].to(self.device)
        pairs = batch["pairs"].to(self.device)

        user_seq_reviews = batch.get("user_seq_reviews")
        item_seq_reviews = batch.get("item_seq_reviews")
        user_seq_len = batch.get("user_seq_len")
        item_seq_len = batch.get("item_seq_len")
        user_pos_ind = batch.get("user_pos_ind")
        item_pos_ind = batch.get("item_pos_ind")
        user_rel_dt = batch.get("user_rel_dt")
        item_rel_dt = batch.get("item_rel_dt")
        user_abs_dt = batch.get("user_abs_dt")
        item_abs_dt = batch.get("item_abs_dt")

        optional_tensors = [
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
        ]
        optional_tensors = [tensor.to(self.device) if tensor is not None else None for tensor in optional_tensors]
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
        ) = optional_tensors

        graph_adj = batch.get("graph_adj")
        graph_reviews = batch.get("graph_reviews")
        graph_ratings = batch.get("graph_ratings")
        if graph_adj is not None:
            graph_adj = graph_adj.to(self.device)
        if graph_reviews is not None:
            graph_reviews = graph_reviews.to(self.device)
        if graph_ratings is not None:
            graph_ratings = graph_ratings.to(self.device)

        return {
            "user_id": user_id,
            "item_id": item_id,
            "rating": ratings,
            "input_u": input_u,
            "input_i": input_i,
            "reuid": reuid,
            "reiid": reiid,
            "nodes": nodes,
            "reviews": reviews,
            "ratings": graph_ratings,
            "adj": adj,
            "pairs": pairs,
            "user_review": input_u,
            "item_review": input_i,
            "user_review_item_ids": reuid,
            "item_review_user_ids": reiid,
            "user_seq_reviews": user_seq_reviews,
            "item_seq_reviews": item_seq_reviews,
            "user_seq_len": user_seq_len,
            "item_seq_len": item_seq_len,
            "user_pos_ind": user_pos_ind,
            "item_pos_ind": item_pos_ind,
            "user_rel_dt": user_rel_dt,
            "item_rel_dt": item_rel_dt,
            "user_abs_dt": user_abs_dt,
            "item_abs_dt": item_abs_dt,
            "graph_adj": graph_adj,
            "graph_reviews": graph_reviews,
            "graph_ratings": graph_ratings,
        }

    def train_epoch(self, epoch_idx):
        self.model.train()
        epoch_loss_dict = None

        for batch in tqdm(self.train_dataloader, desc=f"Epoch {epoch_idx + 1}"):
            inputs = self._prepare_inputs(batch)
            loss, loss_dict = self.model.cal_loss(inputs)

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
        inputs = self._prepare_inputs(batch)
        inputs.pop("rating", None)
        return self.model.forward(**inputs)

    @torch.no_grad()
    def evaluate(self, dataloader, phase="valid"):
        self.model.eval()
        all_predictions = []
        all_ratings = []
        all_user_ids = []

        for batch in dataloader:
            predictions = self._predict_batch(batch).view(-1)
            all_predictions.append(predictions.cpu().numpy())
            all_ratings.append(batch["rating"].cpu().numpy())

            if self.use_ranking_metrics:
                user_ids = batch.get("user_id")
                if user_ids is not None:
                    all_user_ids.append(user_ids.cpu().numpy())

        if not all_ratings:
            empty = np.array([], dtype=np.float64)
            return self._build_eval_metrics(empty, empty, user_ids=None, phase=phase)

        predictions = np.concatenate(all_predictions)
        ratings = np.concatenate(all_ratings)

        user_ids = None
        if self.use_ranking_metrics and all_user_ids:
            user_ids = np.concatenate(all_user_ids)

        test_clip = getattr(self.model, "test_clip", False)
        train_clip = getattr(self.model, "train_clip", False)
        model_clip = test_clip if phase != "train" else train_clip

        if model_clip:
            clipped = np.clip(predictions, self.min_rating, self.max_rating)
        else:
            clipped = predictions

        metrics = self._build_eval_metrics(predictions, ratings, user_ids=user_ids, phase=phase)
        metrics.update({
            "pred_min": float(np.min(predictions)),
            "pred_max": float(np.max(predictions)),
            "clipped_pred_min": float(np.min(clipped)),
            "clipped_pred_max": float(np.max(clipped)),
            "ssg_model_clip_applied": bool(model_clip),
        })

        if phase == "test":
            print(f"[SSG Test] raw_pred: min={predictions.min():.4f}, max={predictions.max():.4f}, mean={predictions.mean():.4f}")
            print(f"[SSG Test] ssg_model_clip={model_clip}, clipped_pred: min={clipped.min():.4f}, max={clipped.max():.4f}, mean={clipped.mean():.4f}")

        return metrics

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
