import numpy as np
from tqdm import tqdm

from .base_trainer import BaseTrainer


class SCGRGCLTrainer(BaseTrainer):
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

        if not all_ratings:
            empty = np.array([], dtype=np.float64)
            return self._build_eval_metrics(empty, empty, phase=phase)

        predictions = np.concatenate(all_predictions)
        ratings = np.concatenate(all_ratings)
        metrics = self._build_eval_metrics(predictions, ratings, phase=phase)

        if getattr(self.model, "current_gate_stats", None):
            metrics.update({key: float(value) for key, value in self.model.current_gate_stats.items()})

        if phase == "test":
            raw = predictions
            clipped = np.clip(raw, self.min_rating, self.max_rating) if self.eval_clip else raw
            print(f"[SCG_RGCL Test] raw_pred: min={raw.min():.4f}, max={raw.max():.4f}, mean={raw.mean():.4f}")
            print(
                f"[SCG_RGCL Test] eval_clip={self.eval_clip}, "
                f"final_pred: min={clipped.min():.4f}, max={clipped.max():.4f}, mean={clipped.mean():.4f}"
            )
            if getattr(self.model, "current_gate_stats", None):
                stats = self.model.current_gate_stats
                print(
                    "[SCG_RGCL Test] gate_stats: "
                    f"mean={stats.get('mean_gate', float('nan')):.4f}, "
                    f"std={stats.get('std_gate', float('nan')):.4f}, "
                    f"min={stats.get('min_gate_value', float('nan')):.4f}, "
                    f"max={stats.get('max_gate_value', float('nan')):.4f}, "
                    f"mean_edge_misalign={stats.get('mean_edge_misalign_score', float('nan')):.4f}"
                )

        return metrics

    def _predict_batch(self, batch):
        return self.model.predict_ratings(batch)
