import torch
import torch.nn.functional as F

from .iard_rm import IARDRM, IARDLossComputer


class IARMRMSentiLossComputer(IARDLossComputer):
    def __init__(self, configs):
        super().__init__(configs)
        self.lambda_gate_anchor = float(configs.get("lambda_gate_anchor", 0.0))
        self.use_q_agree_for_align = bool(configs.get("use_q_agree_for_align", False))

    def forward(self, output, ratings, prototypes, q_agree=None):
        pred = output["pred"]
        zY = output["zY"]
        zX = output["zX"]
        zS = output["zS"]
        zR = output["zR"]
        gate = output["gate"]
        recon_zX = output["recon_zX"]

        rating_loss = F.mse_loss(pred, ratings)

        zY_norm = F.normalize(zY, dim=-1, eps=self.eps)
        zS_norm = F.normalize(zS, dim=-1, eps=self.eps)
        logits = zY_norm @ zS_norm.T
        logits = logits / self.tau_c
        labels = torch.arange(logits.size(0), device=logits.device)
        per_sample_nce = F.cross_entropy(logits, labels, reduction="none")

        if self.use_q_agree_for_align and q_agree is not None:
            align_weight = q_agree.detach().to(gate.device, dtype=gate.dtype)
        else:
            align_weight = gate.detach() if self.detach_gate_for_align else gate
        align_loss = torch.mean(align_weight * per_sample_nce)

        normalized_zR = F.normalize(zR, dim=-1, eps=self.eps)
        normalized_zY = F.normalize(zY, dim=-1, eps=self.eps)
        normalized_zS = F.normalize(zS, dim=-1, eps=self.eps)
        y_r_cos = torch.sum(normalized_zY * normalized_zR, dim=-1)
        s_r_cos = torch.sum(normalized_zS * normalized_zR, dim=-1)
        sep_loss = torch.mean(y_r_cos.square()) + torch.mean(s_r_cos.square())

        recon_target = zX.detach() if self.detach_zx_for_recon else zX
        recon_loss = F.mse_loss(recon_zX, recon_target)

        entropy = -gate * torch.log(gate + self.eps) - (1.0 - gate) * torch.log(1.0 - gate + self.eps)
        gate_loss = -torch.mean(entropy)

        normalized_prototypes = F.normalize(prototypes, dim=-1, eps=self.eps)
        gram = normalized_prototypes @ normalized_prototypes.T
        identity = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
        proto_loss = torch.mean((gram - identity).square())

        if q_agree is None:
            gate_anchor_loss = gate.new_tensor(0.0)
        else:
            q_target = q_agree.detach().to(gate.device, dtype=gate.dtype).clamp(self.eps, 1.0 - self.eps)
            gate_anchor_loss = F.binary_cross_entropy(gate.clamp(self.eps, 1.0 - self.eps), q_target)

        loss = (
            self.lambda_rating * rating_loss
            + self.lambda_align * align_loss
            + self.lambda_sep * sep_loss
            + self.lambda_recon * recon_loss
            + self.lambda_gate * gate_loss
            + self.lambda_proto * proto_loss
            + self.lambda_gate_anchor * gate_anchor_loss
        )

        return {
            "loss": loss,
            "rating_loss": rating_loss,
            "align_loss": align_loss,
            "sep_loss": sep_loss,
            "recon_loss": recon_loss,
            "gate_loss": gate_loss,
            "proto_loss": proto_loss,
            "gate_anchor_loss": gate_anchor_loss,
        }


class IARMRMSenti(IARDRM):
    def __init__(self, configs, train_dataset):
        super().__init__(configs, train_dataset)
        self.loss_computer = IARMRMSentiLossComputer(configs)

    def cal_loss(self, *args, **kwargs):
        batch_data = args[0] if args else kwargs["batch_data"]
        output = self.forward(
            user_ids=batch_data["user_ids"],
            item_ids=batch_data["item_ids"],
            review_emb=batch_data["review_emb"],
            edge_index=batch_data["edge_index"],
            edge_weight=batch_data.get("edge_weight"),
            user_history_emb=batch_data.get("user_history_emb"),
            user_history_mask=batch_data.get("user_history_mask"),
            item_history_emb=batch_data.get("item_history_emb"),
            item_history_mask=batch_data.get("item_history_mask"),
        )
        loss_dict = self.loss_computer(
            output=output,
            ratings=batch_data["ratings"],
            prototypes=self.intent_extractor.prototypes,
            q_agree=batch_data.get("q_agree"),
        )
        loss = loss_dict["loss"]
        numeric = {key: float(value.detach().item()) for key, value in loss_dict.items()}
        numeric["total_loss"] = numeric["loss"]
        return loss, numeric


def run_iarm_rm_senti_sanity_check():
    from data.iarm_rm_senti_dataset import IARMRMSentiDataset

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
    output = model(
        user_ids=batch["user_ids"],
        item_ids=batch["item_ids"],
        review_emb=batch["review_emb"],
        edge_index=dataset.edge_index,
        edge_weight=dataset.edge_weight,
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
    loss = loss_dict["loss"]
    loss.backward()
    if torch.isnan(loss):
        raise AssertionError("Loss is NaN")
    return {
        "loss": float(loss.detach().item()),
        "gate_anchor_loss": float(loss_dict["gate_anchor_loss"].detach().item()),
        "mean_q_agree": float(batch["q_agree"].mean().item()),
        "mean_gate": float(output["gate"].detach().mean().item()),
    }
