import os
from typing import Optional

import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from collections.abc import Sequence

def _load_language_model(model_name, gpu_id):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name) 

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
    model.to(device)
    model.eval()

    return tokenizer, model, device

def _mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embedding_batch(model_name: str, texts: Sequence[str], batch_size: int = 32, gpu_id:int = 0):
    tokenizer, model, device = _load_language_model(model_name, gpu_id)
    
    embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for start in tqdm(
        range(0, len(texts), batch_size),
        total=total_batches, 
        desc="Getting Embedding", 
        unit="batch"
    ):  
        batch = list(texts[start : start + batch_size])
        encoded_input = tokenizer(
            batch, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
            )

        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        with torch.no_grad():
            model_output = model(**encoded_input)

        sentence_embeddings = _mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        embeddings.extend(sentence_embeddings.cpu().tolist())

    return embeddings


def _pool_bert_output(model_output, attention_mask: torch.Tensor, pooling: str) -> torch.Tensor:
    hidden = getattr(model_output, "last_hidden_state", model_output[0])
    if pooling == "cls":
        return hidden[:, 0, :]
    if pooling == "mean":
        return _mean_pooling(model_output, attention_mask)
    raise ValueError("bert_whitening_pooling must be 'cls' or 'mean'.")


def _compute_whitening_stats(vecs: torch.Tensor, output_dim: Optional[int]) -> tuple[torch.Tensor, torch.Tensor]:
    if vecs.dim() != 2:
        raise ValueError("Whitening expects a 2D embedding matrix.")
    mean = vecs.mean(dim=0, keepdim=True)
    centered = vecs - mean
    denom = max(vecs.size(0) - 1, 1)
    cov = centered.T @ centered / float(denom)
    u, s, _ = torch.linalg.svd(cov, full_matrices=False)
    selected_dim = u.size(1) if output_dim is None else int(output_dim)
    selected_dim = min(selected_dim, u.size(1))
    s = s[:selected_dim].clamp_min(1e-9)
    kernel = u[:, :selected_dim] @ torch.diag(torch.rsqrt(s))
    bias = -mean
    return kernel.cpu(), bias.cpu()


def _load_whitening_stats(stats_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    suffix = os.path.splitext(stats_path)[1].lower()
    if suffix == ".pt":
        try:
            loaded = torch.load(stats_path, map_location="cpu", weights_only=True)
        except TypeError:
            loaded = torch.load(stats_path, map_location="cpu")
        if not isinstance(loaded, dict) or "kernel" not in loaded or "bias" not in loaded:
            raise ValueError("Whitening stats .pt must contain {'kernel', 'bias'}.")
        return torch.as_tensor(loaded["kernel"], dtype=torch.float32), torch.as_tensor(loaded["bias"], dtype=torch.float32)
    if os.path.isdir(stats_path):
        kernel = torch.as_tensor(np.load(os.path.join(stats_path, "kernel.npy")), dtype=torch.float32)
        bias = torch.as_tensor(np.load(os.path.join(stats_path, "bias.npy")), dtype=torch.float32)
        return kernel, bias
    raise ValueError("bert_whitening_stats_path must be a .pt file or directory with kernel.npy/bias.npy.")


def _save_whitening_stats(stats_path: str, kernel: torch.Tensor, bias: torch.Tensor) -> None:
    suffix = os.path.splitext(stats_path)[1].lower()
    if suffix == ".pt":
        parent = os.path.dirname(stats_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save({"kernel": kernel.cpu(), "bias": bias.cpu()}, stats_path)
        return
    os.makedirs(stats_path, exist_ok=True)
    np.save(os.path.join(stats_path, "kernel.npy"), kernel.cpu().numpy())
    np.save(os.path.join(stats_path, "bias.npy"), bias.cpu().numpy())


def get_bert_whitening_embedding_batch(
    model_name: str,
    texts: Sequence[str],
    batch_size: int = 32,
    gpu_id: int = 0,
    pooling: str = "cls",
    output_dim: Optional[int] = None,
    normalize: bool = True,
    stats_path: Optional[str] = None,
    fit: bool = True,
    return_tensor: bool = False,
):
    """Return BERT-Whitening sentence embeddings for a text batch.

    If stats_path exists, the saved whitening transform is loaded. Otherwise,
    when fit=True, whitening stats are fit on `texts` and saved to stats_path
    if provided. Callers can fit on train text and reuse stats_path for valid/test
    to avoid evaluation leakage.
    """
    pooling = pooling.lower()
    tokenizer, model, device = _load_language_model(model_name, gpu_id)
    raw_batches: list[torch.Tensor] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for start in tqdm(
        range(0, len(texts), batch_size),
        total=total_batches,
        desc="Getting BERT-Whitening Embedding",
        unit="batch",
    ):
        batch = list(texts[start : start + batch_size])
        encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        encoded_input = {key: value.to(device) for key, value in encoded_input.items()}
        with torch.no_grad():
            model_output = model(**encoded_input)
        pooled = _pool_bert_output(model_output, encoded_input["attention_mask"], pooling)
        raw_batches.append(pooled.detach().cpu())

    if not raw_batches:
        empty_dim = int(output_dim or 0)
        empty = torch.empty((0, empty_dim), dtype=torch.float32)
        return empty if return_tensor else empty.tolist()

    raw_embeddings = torch.cat(raw_batches, dim=0).float()
    kernel: Optional[torch.Tensor] = None
    bias: Optional[torch.Tensor] = None
    if stats_path and os.path.exists(stats_path):
        kernel, bias = _load_whitening_stats(stats_path)
    elif fit:
        kernel, bias = _compute_whitening_stats(raw_embeddings, output_dim)
        if stats_path:
            _save_whitening_stats(stats_path, kernel, bias)

    if kernel is not None and bias is not None:
        embeddings = (raw_embeddings + bias) @ kernel
    else:
        embeddings = raw_embeddings[:, : int(output_dim)] if output_dim is not None else raw_embeddings
    if normalize:
        embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings if return_tensor else embeddings.tolist()
