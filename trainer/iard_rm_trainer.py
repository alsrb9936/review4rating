import csv
import os

import numpy as np
import torch
from tqdm import tqdm

from .base_trainer import BaseTrainer


def _to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def _summarize_distribution(values):
    return float(np.mean(values)), float(np.std(values))


def _intent_entropy(intent_weights, eps=1e-8):
    return -(intent_weights * torch.log(intent_weights + eps)).sum(dim=-1)


def _group_metrics(predictions, ratings):
    errors = predictions - ratings
    return {
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mae": float(np.mean(np.abs(errors))),
        "count": float(len(predictions)),
    }


def inspect_iard_batch(model, loss_computer, batch, edge_index, edge_weight=None, backward=True):
    device = next(model.parameters()).device
    user_ids = batch["user_ids"].to(device)
    item_ids = batch["item_ids"].to(device)
    ratings = batch["ratings"].to(device)
    review_emb = batch["review_emb"].to(device)
    edge_index = edge_index.to(device)
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)

    model.zero_grad(set_to_none=True)
    output = model(
        user_ids=user_ids,
        item_ids=item_ids,
        review_emb=review_emb,
        edge_index=edge_index,
        edge_weight=edge_weight,
    )
    loss_dict = loss_computer(
        output=output,
        ratings=ratings,
        prototypes=model.intent_extractor.prototypes,
    )
    if backward:
        loss_dict["loss"].backward()

    gate_values = _to_numpy(output["gate"])
    p_inc_values = _to_numpy(output["p_inc"])
    alignment_values = _to_numpy(output["alignment"])
    intent_entropy = _to_numpy(_intent_entropy(output["intent_weights"]))

    has_nan = False
    for value in output.values():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            has_nan = True
            break
    if not has_nan:
        for value in loss_dict.values():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                has_nan = True
                break

    numeric_loss_dict = {key: float(value.detach().item()) for key, value in loss_dict.items()}
    mean_gate, std_gate = _summarize_distribution(gate_values)
    mean_p_inc, std_p_inc = _summarize_distribution(p_inc_values)
    mean_alignment, std_alignment = _summarize_distribution(alignment_values)
    mean_intent_entropy, std_intent_entropy = _summarize_distribution(intent_entropy)

    return {
        "pred_shape": tuple(output["pred"].shape),
        "zY_shape": tuple(output["zY"].shape),
        "zS_shape": tuple(output["zS"].shape),
        "zR_shape": tuple(output["zR"].shape),
        "loss_dict": numeric_loss_dict,
        "mean_gate": mean_gate,
        "std_gate": std_gate,
        "mean_p_inc": mean_p_inc,
        "std_p_inc": std_p_inc,
        "mean_alignment": mean_alignment,
        "std_alignment": std_alignment,
        "mean_intent_entropy": mean_intent_entropy,
        "std_intent_entropy": std_intent_entropy,
        "has_nan": bool(has_nan),
    }


class IARDRMTrainer(BaseTrainer):
    def __init__(self, model, train_dataloader, valid_dataloader, test_dataloader, configs):
        super().__init__(model, train_dataloader, valid_dataloader, test_dataloader, configs)
        self.loss_computer = self.model.loss_computer

    def _get_graph_inputs(self, dataloader):
        dataset = dataloader.dataset
        edge_index = dataset.edge_index.to(self.device)
        edge_weight = dataset.edge_weight
        if edge_weight is not None:
            edge_weight = edge_weight.to(self.device)
        return edge_index, edge_weight

    def inspect_batch(self, dataloader=None, backward=True):
        active_dataloader = dataloader or self.train_dataloader
        batch = next(iter(active_dataloader))
        edge_index, edge_weight = self._get_graph_inputs(active_dataloader)
        self.model.train()
        return inspect_iard_batch(
            model=self.model,
            loss_computer=self.loss_computer,
            batch=batch,
            edge_index=edge_index,
            edge_weight=edge_weight,
            backward=backward,
        )

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

    def _build_alignment_group_metrics(self, alignments, predictions, ratings):
        if alignments.size == 0:
            return {}
        low_threshold = float(np.quantile(alignments, 0.2))
        high_threshold = float(np.quantile(alignments, 0.8))
        low_mask = alignments <= low_threshold
        high_mask = alignments >= high_threshold
        mid_mask = (~low_mask) & (~high_mask)

        grouped = {}
        for prefix, mask in [("low_alignment", low_mask), ("mid_alignment", mid_mask), ("high_alignment", high_mask)]:
            if np.any(mask):
                metrics = _group_metrics(predictions[mask], ratings[mask])
            else:
                metrics = {"rmse": 0.0, "mae": 0.0, "count": 0.0}
            grouped[f"{prefix}_rmse"] = metrics["rmse"]
            grouped[f"{prefix}_mae"] = metrics["mae"]
            grouped[f"{prefix}_count"] = metrics["count"]
        return grouped

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
        edge_index, edge_weight = self._get_graph_inputs(self.train_dataloader)

        for batch in tqdm(self.train_dataloader, desc=f"Epoch {epoch_idx + 1}"):
            user_ids = batch["user_ids"].to(self.device)
            item_ids = batch["item_ids"].to(self.device)
            ratings = batch["ratings"].to(self.device)
            review_emb = batch["review_emb"].to(self.device)

            output = self.model(
                user_ids=user_ids,
                item_ids=item_ids,
                review_emb=review_emb,
                edge_index=edge_index,
                edge_weight=edge_weight,
            )
            loss_dict = self.loss_computer(
                output=output,
                ratings=ratings,
                prototypes=self.model.intent_extractor.prototypes,
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

            predictions = output["pred"].detach().cpu().numpy()
            all_predictions.append(predictions)
            all_ratings.append(ratings.detach().cpu().numpy())
            all_gates.append(_to_numpy(output["gate"]))
            all_p_inc.append(_to_numpy(output["p_inc"]))
            all_alignment.append(_to_numpy(output["alignment"]))
            all_intent_entropy.append(_to_numpy(_intent_entropy(output["intent_weights"])))
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
        train_metrics = self._build_metrics(predictions, ratings)
        averaged.update(train_metrics)
        averaged["mean_gate"], averaged["std_gate"] = _summarize_distribution(gate_values)
        averaged["mean_p_inc"], averaged["std_p_inc"] = _summarize_distribution(p_inc_values)
        averaged["mean_alignment"], averaged["std_alignment"] = _summarize_distribution(alignment_values)
        averaged["mean_intent_entropy"], _ = _summarize_distribution(intent_entropy_values)
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
        edge_index, edge_weight = self._get_graph_inputs(dataloader)

        for batch in dataloader:
            user_ids = batch["user_ids"].to(self.device)
            item_ids = batch["item_ids"].to(self.device)
            ratings = batch["ratings"].to(self.device)
            review_emb = batch["review_emb"].to(self.device)

            output = self.model(
                user_ids=user_ids,
                item_ids=item_ids,
                review_emb=review_emb,
                edge_index=edge_index,
                edge_weight=edge_weight,
            )
            predictions = output["pred"]

            all_predictions.append(predictions.detach().cpu().numpy())
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
            empty_history_mask = batch.get("empty_history_mask")
            if empty_history_mask is not None:
                all_empty_history.append(empty_history_mask.detach().cpu().numpy().astype(np.float32))
            review_source_used = batch.get("review_source_used")
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
        empty_history_values = np.concatenate(all_empty_history) if all_empty_history else None
        review_source_values = np.concatenate(all_review_source) if all_review_source else None
        predictions = (
            np.clip(raw_predictions, self.min_rating, self.max_rating)
            if self.eval_clip
            else raw_predictions
        )
        metrics = self._build_eval_metrics(raw_predictions, ratings, phase=phase)
        metrics["mean_gate"], metrics["std_gate"] = _summarize_distribution(gate_values)
        metrics["mean_p_inc"], metrics["std_p_inc"] = _summarize_distribution(p_inc_values)
        metrics["mean_alignment"], metrics["std_alignment"] = _summarize_distribution(alignment_values)
        metrics["mean_intent_entropy"], _ = _summarize_distribution(intent_entropy_values)
        if empty_history_values is not None:
            metrics["mean_empty_history_ratio"] = float(empty_history_values.mean())
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

    def _predict_batch(self, batch):
        edge_index, edge_weight = self._get_graph_inputs(self.valid_dataloader)
        output = self.model(
            user_ids=batch["user_ids"].to(self.device),
            item_ids=batch["item_ids"].to(self.device),
            review_emb=batch["review_emb"].to(self.device),
            edge_index=edge_index,
            edge_weight=edge_weight,
        )
        return output["pred"]


def run_iard_trainer_sanity_check():
    from data.iard_rm_dataset import IARDRMDataset
    from model.iard_rm import IARDRM

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
        "tau_c": 0.2,
        "detach_gate_for_align": True,
        "detach_zx_for_recon": False,
        "review_context_mode": "history",
        "history_aggregation": "mean",
        "history_temporal": False,
    }
    import pandas as pd

    frame = pd.DataFrame(
        {
            "user_id": [0, 1, 2, 3],
            "item_id": [4, 5, 6, 7],
            "rating": [4.0, 3.5, 2.0, 5.0],
            "review_embedding": [torch.randn(768) for _ in range(4)],
        }
    )
    dataset = IARDRMDataset(frame, configs, split="train")
    model = IARDRM(configs, dataset)
    batch = {
        "user_ids": dataset.user_ids[:4],
        "item_ids": dataset.item_ids[:4],
        "ratings": dataset.ratings[:4],
        "review_emb": dataset.review_emb[:4],
        "empty_history_mask": dataset.empty_history_mask[:4],
        "review_source_used": dataset.review_source_used[:4],
    }
    output = model(
        user_ids=batch["user_ids"],
        item_ids=batch["item_ids"],
        review_emb=batch["review_emb"],
        edge_index=dataset.edge_index,
        edge_weight=dataset.edge_weight,
    )
    loss_dict = model.loss_computer(output=output, ratings=batch["ratings"], prototypes=model.intent_extractor.prototypes)
    loss = loss_dict["loss"]
    loss.backward()
    if torch.isnan(loss):
        raise AssertionError("Loss is NaN")
    return {
        "loss": float(loss.detach().item()),
        "mean_gate": float(output["gate"].detach().mean().item()),
        "mean_alignment": float(output["alignment"].detach().mean().item()),
    }


def run_iard_real_batch_check(trainer, dataloader=None, backward=True):
    return trainer.inspect_batch(dataloader=dataloader, backward=backward)
