import torch
import torch.nn as nn
import torch.nn.functional as F

from .iard_rm import IARDRM, IARDLossComputer


class ReviewEmbeddingDecoder(nn.Module):
    def __init__(self, d_model=128, d_text=768, dropout=0.1):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_text),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.decoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, zY):
        return self.decoder(zY)


class IARDRMGenLossComputer(IARDLossComputer):
    def __init__(self, configs):
        super().__init__(configs)
        self.lambda_gen = float(configs.get("lambda_gen", 0.05))
        self.normalize_gen_loss = bool(configs.get("normalize_gen_loss", True))

    def forward(self, output, ratings):
        loss_dict = super().forward(output=output, ratings=ratings)
        generated_review_emb = output.get("generated_review_emb")
        target_review_emb = output.get("target_review_emb")

        if (
            generated_review_emb is None
            or target_review_emb is None
            or generated_review_emb.shape != target_review_emb.shape
        ):
            gen_loss = ratings.new_tensor(0.0)
        elif self.normalize_gen_loss:
            gen_loss = F.mse_loss(
                F.normalize(generated_review_emb, dim=-1, eps=self.eps),
                F.normalize(target_review_emb.detach(), dim=-1, eps=self.eps),
            )
        else:
            gen_loss = F.mse_loss(generated_review_emb, target_review_emb.detach())

        loss_dict["gen_loss"] = gen_loss
        loss_dict["loss"] = loss_dict["loss"] + self.lambda_gen * gen_loss
        return loss_dict


class IARDRMGen(IARDRM):
    """
    IARD-RM variant that never needs the target interaction review at inference.

    The original IARD-RM review branch consumes a review embedding from the batch.
    For interaction-level rating prediction this is unavailable for validation/test
    targets, so this variant decodes a pseudo review embedding from the rating graph
    interaction representation zY and feeds that generated embedding to the review
    branch in both training and evaluation.
    """

    def __init__(self, configs, train_dataset):
        super().__init__(configs, train_dataset)
        self.review_decoder = ReviewEmbeddingDecoder(
            d_model=self.d_model,
            d_text=self.d_text,
            dropout=self.dropout,
        )
        self.gen_teacher_forcing_ratio = float(configs.get("gen_teacher_forcing_ratio", 0.0))
        if not 0.0 <= self.gen_teacher_forcing_ratio <= 1.0:
            raise ValueError("gen_teacher_forcing_ratio must be in [0, 1].")
        self.loss_computer = IARDRMGenLossComputer(configs)

    def _select_review_embedding(self, generated_review_emb, review_emb):
        if (
            self.training
            and self.gen_teacher_forcing_ratio > 0.0
            and review_emb is not None
            and review_emb.shape == generated_review_emb.shape
        ):
            ratio = self.gen_teacher_forcing_ratio
            return ratio * review_emb + (1.0 - ratio) * generated_review_emb
        return generated_review_emb

    def forward(self, *args, **kwargs):
        if args:
            user_ids, item_ids, review_emb, edge_index = args[:4]
            edge_weight = args[4] if len(args) > 4 else None
            user_history_emb = None
            user_history_mask = None
            item_history_emb = None
            item_history_mask = None
        else:
            user_ids = kwargs["user_ids"]
            item_ids = kwargs["item_ids"]
            review_emb = kwargs.get("review_emb")
            edge_index = kwargs["edge_index"]
            edge_weight = kwargs.get("edge_weight")
            user_history_emb = kwargs.get("user_history_emb")
            user_history_mask = kwargs.get("user_history_mask")
            item_history_emb = kwargs.get("item_history_emb")
            item_history_mask = kwargs.get("item_history_mask")

        zY = self.rating_encoder(
            user_ids=user_ids,
            item_ids=item_ids,
            edge_index=edge_index,
            edge_weight=edge_weight,
        )
        generated_review_emb = self.review_decoder(zY)
        effective_review_emb = self._select_review_embedding(generated_review_emb, review_emb)

        del user_history_emb, user_history_mask, item_history_emb, item_history_mask

        hX = self.review_encoder(effective_review_emb)
        zX = hX
        zS, zR, recon_zX = self.disentangler(hX, zY)
        alignment, p_inc, gate = self.gate(zY, zS, zR)
        pred, yY, yS, yR = self.predictor(zY, zS, zR, gate)
        pred = pred + self.user_bias(user_ids).squeeze(-1) + self.item_bias(item_ids).squeeze(-1) + self.global_bias
        return {
            "pred": pred,
            "zY": zY,
            "hX": hX,
            "zX": zX,
            "zS": zS,
            "zR": zR,
            "alignment": alignment,
            "p_inc": p_inc,
            "gate": gate,
            "yY": yY,
            "yS": yS,
            "yR": yR,
            "recon_zX": recon_zX,
            "generated_review_emb": generated_review_emb,
            "target_review_emb": review_emb,
        }


def run_iard_rm_gen_sanity_check():
    from data.iard_rm_gen_dataset import IARDRMGenDataset

    torch.manual_seed(42)
    configs = {
        "num_users": 10,
        "num_items": 20,
        "d_id": 64,
        "d_text": 192,
        "review_emb_dim": 192,
        "d_model": 128,
        "num_layers": 2,
        "eta": 0.1,
        "dropout": 0.1,
        "gate_alpha": 5.0,
        "lambda_rating": 1.0,
        "lambda_align": 0.0,
        "lambda_sep": 0.05,
        "lambda_recon": 0.1,
        "lambda_gen": 0.05,
        "tau_c": 0.2,
        "detach_gate_for_align": True,
        "detach_zx_for_recon": False,
        "review_context_mode": "target",
        "history_aggregation": "mean",
        "history_encoder": "mean",
        "history_top_k": 2,
        "history_temporal": False,
        "retain_rui": False,
    }
    import pandas as pd

    frame = pd.DataFrame(
        {
            "user_id": [0, 1, 2, 3],
            "item_id": [4, 5, 6, 7],
            "rating": [4.0, 3.5, 2.0, 5.0],
            "review_embedding": [torch.randn(192) for _ in range(4)],
        }
    )
    dataset = IARDRMGenDataset(frame, configs, split="train")
    model = IARDRMGen(configs, dataset)
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
    loss_dict = model.loss_computer(output=output, ratings=batch["ratings"])
    loss = loss_dict["loss"]
    loss.backward()
    if torch.isnan(loss):
        raise AssertionError("Loss is NaN")
    return {
        "loss": float(loss.detach().item()),
        "gen_loss": float(loss_dict["gen_loss"].detach().item()),
        "mean_gate": float(output["gate"].detach().mean().item()),
    }
