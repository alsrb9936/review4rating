import numpy as np

from .scg_rgcl_trainer import SCGRGCLTrainer


class MARGCLTrainer(SCGRGCLTrainer):
    """Trainer for MA-RGCL; keeps RGCL full-graph training semantics."""

    def evaluate(self, dataloader, phase="valid"):
        self.model.eval()
        all_predictions = []
        all_ratings = []
        all_user_ids = []

        for batch in dataloader:
            predictions = self.model.predict_ratings(batch)
            ratings = batch["ratings"]
            all_predictions.append(predictions.detach().cpu().numpy())
            all_ratings.append(ratings.cpu().numpy())

            if self.use_ranking_metrics:
                user_ids = batch.get("decoder_user_ids")
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

        metrics = self._build_eval_metrics(predictions, ratings, user_ids=user_ids, phase=phase)

        if getattr(self.model, "current_gate_stats", None):
            metrics.update({key: float(value) for key, value in self.model.current_gate_stats.items()})

        if phase == "test" and getattr(self.model, "current_gate_stats", None):
            raw = predictions
            clipped = np.clip(raw, self.min_rating, self.max_rating) if self.eval_clip else raw
            print(f"[MA_RGCL Test] raw_pred: min={raw.min():.4f}, max={raw.max():.4f}, mean={raw.mean():.4f}")
            print(
                f"[MA_RGCL Test] eval_clip={self.eval_clip}, "
                f"final_pred: min={clipped.min():.4f}, max={clipped.max():.4f}, mean={clipped.mean():.4f}"
            )
            stats = self.model.current_gate_stats
            print(
                "[MA_RGCL Test] gate_stats: "
                f"mean={stats.get('mean_gate', float('nan')):.4f}, "
                f"std={stats.get('std_gate', float('nan')):.4f}, "
                f"min={stats.get('min_gate_value', float('nan')):.4f}, "
                f"max={stats.get('max_gate_value', float('nan')):.4f}, "
                f"mean_gamma={stats.get('mean_gamma', float('nan')):.4f}, "
                f"mean_misalign={stats.get('mean_edge_misalign_score', float('nan')):.4f}"
            )
        return metrics
