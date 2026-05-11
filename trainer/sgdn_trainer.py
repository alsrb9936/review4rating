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
        avg = {key: value / num_batches for key, value in epoch_loss_dict.items()}

        cl_weight = float(self.configs.get("cl_weight", 0.1))
        disentangle_weight = float(self.configs.get("disentangle_weight", 0.01))
        if "cl_loss" in avg:
            avg["weighted_cl_loss"] = avg["cl_loss"] * cl_weight
        if "disentangle_loss" in avg:
            avg["weighted_disentangle_loss"] = avg["disentangle_loss"] * disentangle_weight

        print(f"  [SGDN Losses] rating={avg.get('rating_loss', 0):.4f}, "
              f"cl={avg.get('cl_loss', 0):.4f} (w={avg.get('weighted_cl_loss', 0):.4f}), "
              f"disentangle={avg.get('disentangle_loss', 0):.4f} (w={avg.get('weighted_disentangle_loss', 0):.4f}), "
              f"total={avg.get('total_loss', 0):.4f}")

        return avg

    def evaluate(self, dataloader, phase="valid"):
        self.model.eval()
        all_predictions = []
        all_ratings = []

        for batch in dataloader:
            predictions = self.model.predict_ratings(batch)
            ratings = batch["ratings"]
            all_predictions.append(predictions.detach().cpu().numpy())
            all_ratings.append(ratings.cpu().numpy())

        if not all_ratings:
            empty = np.array([], dtype=np.float64)
            return self._build_eval_metrics(empty, empty, phase=phase)

        predictions = np.concatenate(all_predictions)
        ratings = np.concatenate(all_ratings)

        metrics = self._build_eval_metrics(predictions, ratings, phase=phase)
        clipped = np.clip(predictions, self.min_rating, self.max_rating)
        clipped_errors = clipped - ratings
        metrics.update({
            "mse_clipped": float(np.mean(np.square(clipped_errors))),
            "rmse_clipped": float(np.sqrt(np.mean(np.square(clipped_errors)))),
            "mae_clipped": float(np.mean(np.abs(clipped_errors))),
            "raw_pred_min": float(np.min(predictions)),
            "raw_pred_max": float(np.max(predictions)),
            "raw_pred_mean": float(np.mean(predictions)),
            "raw_pred_std": float(np.std(predictions)),
        })

        if phase == "test":
            raw = predictions
            print(f"[SGDN Test] raw_pred: min={raw.min():.4f}, max={raw.max():.4f}, mean={raw.mean():.4f}")
            print(f"[SGDN Test] eval_clip={self.eval_clip}, clipped_pred: min={clipped.min():.4f}, max={clipped.max():.4f}, mean={clipped.mean():.4f}")

        return metrics

    def _predict_batch(self, batch):
        return self.model.predict_ratings(batch)
