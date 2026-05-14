import csv
import os

import numpy as np
import torch
from tqdm import tqdm

from .iard_rm_trainer import IARDRMTrainer, _group_metrics, _intent_entropy, _summarize_distribution, _to_numpy


def _safe_corr(left, right):
    if left.size < 2 or right.size < 2:
        return 0.0
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


class IARMRMSentiTrainer(IARDRMTrainer):
    def _build_consistency_group_metrics(self, consistent_values, predictions, ratings):
        grouped = {}
        for prefix, mask in [
            ("consistent", consistent_values.astype(bool)),
            ("inconsistent", ~consistent_values.astype(bool)),
        ]:
            if np.any(mask):
                metrics = _group_metrics(predictions[mask], ratings[mask])
            else:
                metrics = {"rmse": 0.0, "mae": 0.0, "count": 0.0}
            grouped[f"{prefix}_rmse"] = metrics["rmse"]
            grouped[f"{prefix}_mae"] = metrics["mae"]
            grouped[f"{prefix}_count"] = metrics["count"]
        return grouped

    def _build_rating_bin_q_inc(self, ratings, q_inc_values):
        grouped = {}
        rounded_ratings = np.rint(ratings).astype(np.int64)
        for rating_bin in range(1, 6):
            mask = rounded_ratings == rating_bin
            grouped[f"rating_{rating_bin}_mean_q_inc"] = float(np.mean(q_inc_values[mask])) if np.any(mask) else 0.0
        return grouped

    def _write_eval_csv(self, phase, records):
        if not records:
            return None
        file_path = os.path.join(self.result_path, f"{phase}_sample_analysis.csv")
        fieldnames = list(records[0].keys())
        with open(file_path, "w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        return file_path

    def train_epoch(self, epoch_idx):
        self.model.train()
        epoch_loss_sums = None
        all_predictions = []
        all_ratings = []
        all_gates = []
        all_p_inc = []
        all_alignment = []
        all_intent_entropy = []
        all_empty_history = []
        all_q_agree = []
        all_q_inc = []
        all_is_consistent = []
        edge_index, edge_weight = self._get_graph_inputs(self.train_dataloader)

        for batch in tqdm(self.train_dataloader, desc=f"Epoch {epoch_idx + 1}"):
            user_ids = batch["user_ids"].to(self.device)
            item_ids = batch["item_ids"].to(self.device)
            ratings = batch["ratings"].to(self.device)
            review_emb = batch["review_emb"].to(self.device)
            q_agree = batch["q_agree"].to(self.device)
            user_history_emb = batch.get("user_history_emb")
            user_history_mask = batch.get("user_history_mask")
            item_history_emb = batch.get("item_history_emb")
            item_history_mask = batch.get("item_history_mask")

            output = self.model(
                user_ids=user_ids,
                item_ids=item_ids,
                review_emb=review_emb,
                edge_index=edge_index,
                edge_weight=edge_weight,
                user_history_emb=user_history_emb.to(self.device) if user_history_emb is not None else None,
                user_history_mask=user_history_mask.to(self.device) if user_history_mask is not None else None,
                item_history_emb=item_history_emb.to(self.device) if item_history_emb is not None else None,
                item_history_mask=item_history_mask.to(self.device) if item_history_mask is not None else None,
            )
            loss_dict = self.loss_computer(
                output=output,
                ratings=ratings,
                prototypes=self.model.intent_extractor.prototypes,
                q_agree=q_agree,
            )
            loss = loss_dict["loss"]

            if epoch_loss_sums is None:
                epoch_loss_sums = {key: 0.0 for key in loss_dict}
                epoch_loss_sums["total_loss"] = 0.0

            self.optimizer.zero_grad()
            loss.backward()
            grad_clip = self.configs.get("grad_clip", None)
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.optimizer.step()

            for key, value in loss_dict.items():
                epoch_loss_sums[key] += float(value.detach().item())
            epoch_loss_sums["total_loss"] += float(loss.detach().item())

            all_predictions.append(output["pred"].detach().cpu().numpy())
            all_ratings.append(ratings.detach().cpu().numpy())
            all_gates.append(_to_numpy(output["gate"]))
            all_p_inc.append(_to_numpy(output["p_inc"]))
            all_alignment.append(_to_numpy(output["alignment"]))
            all_intent_entropy.append(_to_numpy(_intent_entropy(output["intent_weights"])))
            all_q_agree.append(batch["q_agree"].detach().cpu().numpy())
            all_q_inc.append(batch["q_inc"].detach().cpu().numpy())
            all_is_consistent.append(batch["is_consistent"].detach().cpu().numpy().astype(bool))
            empty_history_mask = batch.get("empty_history_mask")
            if empty_history_mask is not None:
                all_empty_history.append(empty_history_mask.detach().cpu().numpy().astype(np.float32))

        if epoch_loss_sums is None:
            return {}

        num_batches = max(len(self.train_dataloader), 1)
        averaged = {key: value / num_batches for key, value in epoch_loss_sums.items()}
        predictions = np.concatenate(all_predictions)
        ratings = np.concatenate(all_ratings)
        gate_values = np.concatenate(all_gates)
        p_inc_values = np.concatenate(all_p_inc)
        alignment_values = np.concatenate(all_alignment)
        intent_entropy_values = np.concatenate(all_intent_entropy)
        q_agree_values = np.concatenate(all_q_agree)
        q_inc_values = np.concatenate(all_q_inc)
        is_consistent_values = np.concatenate(all_is_consistent)
        train_metrics = self._build_metrics(predictions, ratings)
        averaged.update(train_metrics)
        averaged["mean_gate"], averaged["std_gate"] = _summarize_distribution(gate_values)
        averaged["mean_p_inc"], averaged["std_p_inc"] = _summarize_distribution(p_inc_values)
        averaged["mean_alignment"], averaged["std_alignment"] = _summarize_distribution(alignment_values)
        averaged["mean_intent_entropy"], _ = _summarize_distribution(intent_entropy_values)
        averaged["mean_q_agree"], averaged["std_q_agree"] = _summarize_distribution(q_agree_values)
        averaged["mean_q_inc"], averaged["std_q_inc"] = _summarize_distribution(q_inc_values)
        averaged["corr_gate_q_agree"] = _safe_corr(gate_values, q_agree_values)
        averaged.update(self._build_consistency_group_metrics(is_consistent_values, predictions, ratings))
        averaged.update(self._build_rating_bin_q_inc(ratings, q_inc_values))
        if all_empty_history:
            averaged["mean_empty_history_ratio"] = float(np.concatenate(all_empty_history).mean())
        return averaged

    @torch.no_grad()
    def evaluate(self, dataloader, phase="valid"):
        self.model.eval()
        all_predictions = []
        all_ratings = []
        all_user_ids = []
        all_item_ids = []
        all_gates = []
        all_p_inc = []
        all_alignment = []
        all_intent_entropy = []
        all_yY = []
        all_yS = []
        all_yR = []
        all_empty_history = []
        all_review_source = []
        all_q_agree = []
        all_q_inc = []
        all_is_consistent = []
        all_sent_score = []
        edge_index, edge_weight = self._get_graph_inputs(dataloader)

        for batch in dataloader:
            user_ids = batch["user_ids"].to(self.device)
            item_ids = batch["item_ids"].to(self.device)
            ratings = batch["ratings"].to(self.device)
            review_emb = batch["review_emb"].to(self.device)
            user_history_emb = batch.get("user_history_emb")
            user_history_mask = batch.get("user_history_mask")
            item_history_emb = batch.get("item_history_emb")
            item_history_mask = batch.get("item_history_mask")
            review_source_used = batch.get("review_source_used")
            if phase in {"valid", "test"} and review_source_used is not None:
                if not torch.all(review_source_used == 1):
                    raise RuntimeError(f"IARM-RM-Senti {phase} must use train-history context, not target reviews.")

            output = self.model(
                user_ids=user_ids,
                item_ids=item_ids,
                review_emb=review_emb,
                edge_index=edge_index,
                edge_weight=edge_weight,
                user_history_emb=user_history_emb.to(self.device) if user_history_emb is not None else None,
                user_history_mask=user_history_mask.to(self.device) if user_history_mask is not None else None,
                item_history_emb=item_history_emb.to(self.device) if item_history_emb is not None else None,
                item_history_mask=item_history_mask.to(self.device) if item_history_mask is not None else None,
            )

            all_predictions.append(output["pred"].detach().cpu().numpy())
            all_ratings.append(ratings.detach().cpu().numpy())
            all_user_ids.append(user_ids.detach().cpu().numpy())
            all_item_ids.append(item_ids.detach().cpu().numpy())
            all_gates.append(_to_numpy(output["gate"]))
            all_p_inc.append(_to_numpy(output["p_inc"]))
            all_alignment.append(_to_numpy(output["alignment"]))
            all_intent_entropy.append(_to_numpy(_intent_entropy(output["intent_weights"])))
            all_yY.append(_to_numpy(output["yY"]))
            all_yS.append(_to_numpy(output["yS"]))
            all_yR.append(_to_numpy(output["yR"]))
            all_q_agree.append(batch["q_agree"].detach().cpu().numpy())
            all_q_inc.append(batch["q_inc"].detach().cpu().numpy())
            all_is_consistent.append(batch["is_consistent"].detach().cpu().numpy().astype(bool))
            all_sent_score.append(batch["sent_score"].detach().cpu().numpy())
            empty_history_mask = batch.get("empty_history_mask")
            if empty_history_mask is not None:
                all_empty_history.append(empty_history_mask.detach().cpu().numpy().astype(np.float32))
            if review_source_used is not None:
                all_review_source.append(review_source_used.detach().cpu().numpy())

        if not all_ratings:
            empty = np.array([], dtype=np.float64)
            return self._build_eval_metrics(empty, empty, phase=phase)

        raw_predictions = np.concatenate(all_predictions)
        ratings = np.concatenate(all_ratings)
        user_ids = np.concatenate(all_user_ids)
        item_ids = np.concatenate(all_item_ids)
        gate_values = np.concatenate(all_gates)
        p_inc_values = np.concatenate(all_p_inc)
        alignment_values = np.concatenate(all_alignment)
        intent_entropy_values = np.concatenate(all_intent_entropy)
        yY_values = np.concatenate(all_yY)
        yS_values = np.concatenate(all_yS)
        yR_values = np.concatenate(all_yR)
        q_agree_values = np.concatenate(all_q_agree)
        q_inc_values = np.concatenate(all_q_inc)
        is_consistent_values = np.concatenate(all_is_consistent)
        sent_score_values = np.concatenate(all_sent_score)
        empty_history_values = np.concatenate(all_empty_history) if all_empty_history else None
        review_source_values = np.concatenate(all_review_source) if all_review_source else None
        predictions = np.clip(raw_predictions, self.min_rating, self.max_rating) if self.eval_clip else raw_predictions
        metrics = self._build_eval_metrics(raw_predictions, ratings, phase=phase)
        metrics["mean_gate"], metrics["std_gate"] = _summarize_distribution(gate_values)
        metrics["mean_p_inc"], metrics["std_p_inc"] = _summarize_distribution(p_inc_values)
        metrics["mean_alignment"], metrics["std_alignment"] = _summarize_distribution(alignment_values)
        metrics["mean_intent_entropy"], _ = _summarize_distribution(intent_entropy_values)
        metrics["mean_q_agree"], metrics["std_q_agree"] = _summarize_distribution(q_agree_values)
        metrics["mean_q_inc"], metrics["std_q_inc"] = _summarize_distribution(q_inc_values)
        metrics["corr_gate_q_agree"] = _safe_corr(gate_values, q_agree_values)
        metrics.update(self._build_consistency_group_metrics(is_consistent_values, predictions, ratings))
        metrics.update(self._build_rating_bin_q_inc(ratings, q_inc_values))
        if empty_history_values is not None:
            metrics["mean_empty_history_ratio"] = float(empty_history_values.mean())
        if review_source_values is not None and phase in {"valid", "test"}:
            metrics["history_source_ratio"] = float(np.mean(review_source_values == 1))
            print(
                f"IARM-RM-Senti {phase} leakage check: history_source_ratio={metrics['history_source_ratio']:.4f}, "
                f"target_review_used={bool(np.any(review_source_values == 0))}, "
                f"empty_history_ratio={metrics.get('mean_empty_history_ratio', 0.0):.4f}"
            )
        metrics.update(self._build_alignment_group_metrics(alignment_values, predictions, ratings))

        if phase == "test":
            abs_error = np.abs(predictions - ratings)
            records = []
            for idx in range(len(predictions)):
                records.append(
                    {
                        "user_id": int(user_ids[idx]),
                        "item_id": int(item_ids[idx]),
                        "rating": float(ratings[idx]),
                        "pred_unclipped": float(raw_predictions[idx]),
                        "pred": float(predictions[idx]),
                        "abs_error": float(abs_error[idx]),
                        "gate": float(gate_values[idx]),
                        "p_inc": float(p_inc_values[idx]),
                        "q_agree": float(q_agree_values[idx]),
                        "q_inc": float(q_inc_values[idx]),
                        "is_consistent": bool(is_consistent_values[idx]),
                        "sent_score": float(sent_score_values[idx]),
                        "alignment": float(alignment_values[idx]),
                        "intent_entropy": float(intent_entropy_values[idx]),
                        "yY": float(yY_values[idx]),
                        "yS": float(yS_values[idx]),
                        "yR": float(yR_values[idx]),
                        "empty_history_mask": bool(empty_history_values[idx]) if empty_history_values is not None else False,
                        "review_source_used": int(review_source_values[idx]) if review_source_values is not None else -1,
                    }
                )
            csv_path = self._write_eval_csv(phase=phase, records=records)
            if csv_path is not None:
                print(f"saved {phase} sample analysis to {csv_path}")
        return metrics


def run_iarm_senti_trainer_sanity_check():
    from data.iarm_rm_senti_dataset import IARMRMSentiDataset
    from model.iarm_rm_senti import IARMRMSenti

    torch.manual_seed(42)
    configs = {
        "num_users": 10,
        "num_items": 20,
        "d_id": 64,
        "d_text": 768,
        "d_model": 128,
        "num_layers": 2,
        "num_intents": 8,
        "eta": 0.1,
        "dropout": 0.1,
        "tau_p": 0.2,
        "gate_alpha": 5.0,
        "lambda_rating": 1.0,
        "lambda_align": 0.1,
        "lambda_sep": 0.05,
        "lambda_recon": 0.1,
        "lambda_gate": 0.001,
        "lambda_proto": 0.01,
        "lambda_gate_anchor": 0.1,
        "use_q_agree_for_align": True,
        "tau_c": 0.2,
        "detach_gate_for_align": True,
        "detach_zx_for_recon": False,
        "review_context_mode": "history",
        "history_aggregation": "mean",
        "history_encoder": "attention",
        "history_top_k": 2,
        "history_temporal": False,
    }
    import pandas as pd

    frame = pd.DataFrame(
        {
            "user_id": [0, 1, 2, 3],
            "item_id": [4, 5, 6, 7],
            "rating": [4.0, 3.5, 2.0, 5.0],
            "review_embedding": [torch.randn(768) for _ in range(4)],
            "review_score": [[0.05, 0.10, 0.85], [0.20, 0.65, 0.15], [0.75, 0.10, 0.15], [0.30, 0.20, 0.50]],
            "review_sentiment": ["positive", "neutral", "negative", "positive"],
            "rating_sentiment": ["positive", "positive", "negative", "positive"],
            "is_consistent": [True, False, True, True],
        }
    )
    dataset = IARMRMSentiDataset(frame, configs, split="train")
    model = IARMRMSenti(configs, dataset)
    batch = {key: torch.stack([dataset[idx][key] for idx in range(len(dataset))]) for key in dataset[0]}
    summary = self_inspect_iarm_batch(model, batch, dataset.edge_index, dataset.edge_weight)
    return summary


def self_inspect_iarm_batch(model, batch, edge_index, edge_weight=None):
    output = model(
        user_ids=batch["user_ids"],
        item_ids=batch["item_ids"],
        review_emb=batch["review_emb"],
        edge_index=edge_index,
        edge_weight=edge_weight,
        user_history_emb=batch["user_history_emb"],
        user_history_mask=batch["user_history_mask"],
        item_history_emb=batch["item_history_emb"],
        item_history_mask=batch["item_history_mask"],
    )
    loss_dict = model.loss_computer(
        output=output,
        ratings=batch["ratings"],
        prototypes=model.intent_extractor.prototypes,
        q_agree=batch["q_agree"],
    )
    loss_dict["loss"].backward()
    return {
        "loss": float(loss_dict["loss"].detach().item()),
        "gate_anchor_loss": float(loss_dict["gate_anchor_loss"].detach().item()),
        "mean_q_agree": float(batch["q_agree"].mean().item()),
        "mean_gate": float(output["gate"].detach().mean().item()),
    }
