
import os
import random
from typing import Any, Optional, cast
import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd
import pickle

from data import DATASET_DICT
from data.process.embeddings import get_bert_whitening_embedding_batch, get_embedding_batch
from data.process.sentiment_anlysis import predict_sentiments, map_rating_to_sentiment, check_consistency
from sklearn.model_selection import train_test_split


def _build_sentiment_features(inter_df, review_sentiments, review_scores):
    inter_df["review_score"] = review_scores
    inter_df["review_sentiment"] = review_sentiments
    inter_df["rating_sentiment"] = inter_df["rating"].apply(map_rating_to_sentiment)
    inter_df["is_consistent"] = inter_df.apply(
        lambda row: check_consistency(row["review_sentiment"], row["rating_sentiment"], row["rating"]),
        axis=1,
    )


def _load_cached_review_embeddings(path):
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".pt":
        try:
            loaded = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            loaded = torch.load(path, map_location="cpu")
        if isinstance(loaded, torch.Tensor):
            return [tensor.clone().detach() for tensor in loaded]
        return loaded
    if suffix == ".npy":
        loaded = np.load(path, allow_pickle=False)
        return [row for row in loaded]
    if suffix == ".pkl":
        raise ValueError(
            "Unsafe review embedding format '.pkl' is not supported. "
            "Use a trusted .pt tensor file or numeric .npy array instead."
        )
    raise ValueError(f"Unsupported review embedding format: {path}")


def _embedding_cache_path(configs, dataset: str, backend: str, model_name: str) -> str:
    safe_model = model_name.split("/")[-1]
    if backend == "sentence_transformer":
        return f"{configs['embedding_path']}/{dataset}/{safe_model}.pt"
    if backend == "bert_whitening":
        dim = int(configs.get("bert_whitening_dim", configs.get("review_dim", 64)))
        pooling = configs.get("bert_whitening_pooling", "cls")
        return f"{configs['embedding_path']}/{dataset}/{backend}_{safe_model}_{pooling}_{dim}.pt"
    return f"{configs['embedding_path']}/{dataset}/{backend}_{safe_model}.pt"


def _bert_whitening_full_stats_path(embedding_path: str) -> str:
    root, ext = os.path.splitext(embedding_path)
    return f"{root}_stats{ext or '.pt'}"


def _infer_embedding_dim(review_embeddings) -> Optional[int]:
    for review_embedding in review_embeddings:
        if review_embedding is None:
            continue
        if isinstance(review_embedding, torch.Tensor):
            return int(review_embedding.numel())
        return int(torch.tensor(review_embedding, dtype=torch.float32).numel())
    return None


def _validate_review_dim(review_embeddings, configs) -> None:
    actual_dim = _infer_embedding_dim(review_embeddings)
    if actual_dim is None:
        return
    configured_dim = int(configs.get("review_dim", actual_dim))
    if actual_dim != configured_dim:
        raise ValueError(
            f"Review embedding dim mismatch: generated/cached dim={actual_dim}, "
            f"but configs['review_dim']={configured_dim}. Set review_dim to match the backend "
            "(for BERT-Whitening usually --bert_whitening_dim) or provide matching cached embeddings."
        )


def _fill_missing_embeddings(frame: pd.DataFrame, review_embeddings: list[object], configs) -> list[object]:
    dim = int(configs.get("review_dim", configs.get("bert_whitening_dim", 64)))
    fallback = [0.0] * dim
    return [embedding if embedding is not None else fallback for embedding in review_embeddings]


def _compute_bert_whitening_for_frame(frame: pd.DataFrame, configs, stats_path: str, fit: bool) -> list[object]:
    gpu_id = configs.get('gpu', 0)
    review_embeddings: list[object] = [None] * len(frame)
    non_empty_idx = [i for i, text in enumerate(frame["reviewText"].tolist()) if text]
    if non_empty_idx:
        non_empty_texts = [frame.iloc[i]["reviewText"] for i in non_empty_idx]
        predicted_embeddings = get_bert_whitening_embedding_batch(
            str(configs.get("bert_whitening_model", "bert-base-uncased")),
            non_empty_texts,
            batch_size=8,
            gpu_id=gpu_id,
            pooling=str(configs.get("bert_whitening_pooling", "cls")),
            output_dim=int(configs.get("bert_whitening_dim", configs.get("review_dim", 64))),
            normalize=bool(configs.get("bert_whitening_normalize", True)),
            stats_path=stats_path,
            fit=fit,
        )
        for idx, emb in zip(non_empty_idx, predicted_embeddings):
            review_embeddings[idx] = emb
    review_embeddings = _fill_missing_embeddings(frame, review_embeddings, configs)
    _validate_review_dim(review_embeddings, configs)
    return review_embeddings


def _bert_whitening_cache_paths(configs) -> tuple[str, str]:
    dataset = str(configs["dataset"])
    model_name = str(configs.get("bert_whitening_model", "bert-base-uncased")).split("/")[-1]
    pooling = str(configs.get("bert_whitening_pooling", "cls"))
    dim = int(configs.get("bert_whitening_dim", configs.get("review_dim", 64)))
    split_protocol = str(configs.get("split_protocol", "default"))
    seed = int(configs.get("seed", 42))
    cache_name = f"bert_whitening_{model_name}_{pooling}_{dim}_{split_protocol}_seed{seed}"
    cache_dir = os.path.join(str(configs["embedding_path"]), dataset)
    return os.path.join(cache_dir, f"{cache_name}.pt"), os.path.join(cache_dir, f"{cache_name}_stats.pt")


def _split_signature(frame: pd.DataFrame):
    if len(frame) == 0:
        return {"n": 0, "user_sum": 0, "item_sum": 0, "rating_sum": 0.0}
    return {
        "n": int(len(frame)),
        "user_sum": int(frame["user_id"].sum()),
        "item_sum": int(frame["item_id"].sum()),
        "rating_sum": float(frame["rating"].sum()),
    }


def _bert_whitening_cache_signatures(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame):
    return {
        "train": _split_signature(train_df),
        "valid": _split_signature(valid_df),
        "test": _split_signature(test_df),
    }


def attach_bert_whitening_review_features(train_df, valid_df, test_df, configs):
    if str(configs.get("review_feature_backend", "sentence_transformer")) != "bert_whitening":
        return train_df, valid_df, test_df
    if str(configs.get("bert_whitening_cache_scope", "full")) == "full":
        return train_df, valid_df, test_df

    configs["review_dim"] = int(configs.get("bert_whitening_dim", configs.get("review_dim", 64)))
    cache_path, default_stats_path = _bert_whitening_cache_paths(configs)
    signatures = _bert_whitening_cache_signatures(train_df, valid_df, test_df)
    if os.path.exists(cache_path):
        try:
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        except TypeError:
            cached = torch.load(cache_path, map_location="cpu")
        if isinstance(cached, dict) and cached.get("signatures") == signatures:
            print(f"Load cached BERT-Whitening review embeddings from {cache_path}")
            train_df = train_df.copy()
            valid_df = valid_df.copy()
            test_df = test_df.copy()
            train_df["review_embedding"] = cached["train"]
            valid_df["review_embedding"] = cached["valid"]
            test_df["review_embedding"] = cached["test"]
            return train_df, valid_df, test_df
        print(f"Cached BERT-Whitening embeddings ignored due to split signature mismatch: {cache_path}")

    stats_path = configs.get("bert_whitening_stats_path")
    if not stats_path:
        stats_path = default_stats_path
        configs["bert_whitening_stats_path"] = stats_path

    print("Fit BERT-Whitening stats on train reviews and apply to splits. . . ")
    train_df = train_df.copy()
    valid_df = valid_df.copy()
    test_df = test_df.copy()
    train_df["review_embedding"] = _compute_bert_whitening_for_frame(train_df, configs, str(stats_path), fit=True)
    valid_df["review_embedding"] = _compute_bert_whitening_for_frame(valid_df, configs, str(stats_path), fit=False)
    test_df["review_embedding"] = _compute_bert_whitening_for_frame(test_df, configs, str(stats_path), fit=False)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(
        {
            "signatures": signatures,
            "train": train_df["review_embedding"].tolist(),
            "valid": valid_df["review_embedding"].tolist(),
            "test": test_df["review_embedding"].tolist(),
        },
        cache_path,
    )
    print(f"Saved BERT-Whitening review embeddings to {cache_path}")
    return train_df, valid_df, test_df

# dataset column 
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_interaction_data(configs):
    print(f"load interaction data from {configs['data_path']}")
    path = configs["data_path"]
    gpu_id = configs.get('gpu', 0)
    model_name = configs['language_model']
    sentiment_model = configs['sentiment_model']
    dataset = configs['dataset']

    use_review_text = bool(configs.get("use_review_text", False))
    use_review_embedding = bool(configs.get("use_review_embedding", False))
    compute_review_embedding = bool(configs.get("compute_review_embedding", False))
    use_sentiment = bool(configs.get("use_sentiment", False))
    use_bert_whitening = bool(configs.get("use_bert_whitening", False))
    review_feature_backend = str(configs.get("review_feature_backend", "sentence_transformer"))

    if review_feature_backend not in {"sentence_transformer", "bert_whitening", "cached"}:
        raise ValueError("review_feature_backend must be sentence_transformer, bert_whitening, or cached.")

    if (use_review_embedding or compute_review_embedding or use_bert_whitening) and not use_review_text:
        print(f"[WARN] use_review_text=False but review embedding/BERT-Whitening is requested. Auto-enabling use_review_text=True.")
        use_review_text = True

    if use_bert_whitening:
        configs["review_dim"] = int(configs.get("bert_whitening_dim", configs.get("review_dim", 64)))

    inter_path = f"{path}/{dataset}/{dataset}.inter"
    inter_df = pd.read_csv(inter_path, sep="\t")

    if use_review_text:
        review_path = f"{path}/{dataset}/{dataset}.review"
        review_df = pd.read_csv(review_path, sep="\t")
        inter_df = pd.merge(
            inter_df,
            review_df,
            on=["user_id:token", "item_id:token"],
            how="inner"
        )
        inter_df.columns = inter_df.columns.str.split(':').str[0]
        inter_df["reviewText"] = inter_df["reviewText"].apply(normalize_review_text)
        print("Merged interaction and review data")
    else:
        inter_df.columns = inter_df.columns.str.split(':').str[0]
        print("Loaded interaction data only (no review text)")

    if use_review_embedding or use_bert_whitening:
        print("Get embedding from review...")
        configured_review_emb_path = configs.get("review_emb_path")
        if review_feature_backend == "cached" and not configured_review_emb_path:
            raise ValueError("review_feature_backend='cached' requires review_emb_path.")
        backend_model_name = str(configs.get("bert_whitening_model", "bert-base-uncased")) if use_bert_whitening else model_name
        embedding_path = configured_review_emb_path if review_feature_backend == "cached" else _embedding_cache_path(configs, dataset, review_feature_backend, backend_model_name)

        if use_bert_whitening:
            cache_scope = str(configs.get("bert_whitening_cache_scope", "full"))
            if cache_scope == "full":
                stats_path = configs.get("bert_whitening_stats_path") or _bert_whitening_full_stats_path(embedding_path)
                configs["bert_whitening_stats_path"] = stats_path
                if os.path.exists(embedding_path):
                    print(f"Load cached full BERT-Whitening review embeddings from {embedding_path}")
                    review_embeddings = _load_cached_review_embeddings(embedding_path)
                    if len(review_embeddings) != len(inter_df):
                        raise ValueError(
                            f"Cached embeddings length {len(review_embeddings)} != inter_df length {len(inter_df)}"
                        )
                else:
                    print("Fit BERT-Whitening stats on all reviews and cache full-dataset embeddings...")
                    review_embeddings = _compute_bert_whitening_for_frame(inter_df, configs, str(stats_path), fit=True)
                    os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
                    torch.save(review_embeddings, embedding_path)
                    print(f"Saved full BERT-Whitening review embeddings to {embedding_path}")
                _validate_review_dim(review_embeddings, configs)
                inter_df["review_embedding"] = review_embeddings
            else:
                print("BERT-Whitening embeddings are fit after train/valid/test split to avoid whitening-stat leakage.")
                inter_df["review_embedding"] = [None] * len(inter_df)
        elif review_feature_backend == "cached":
            if not os.path.exists(embedding_path):
                raise FileNotFoundError(f"Review embedding file not found: {embedding_path}")
            review_embeddings = _load_cached_review_embeddings(embedding_path)
            if len(review_embeddings) != len(inter_df):
                raise ValueError(
                    f"Cached embeddings length {len(review_embeddings)} != inter_df length {len(inter_df)}"
                )
            actual_dim = _infer_embedding_dim(review_embeddings)
            if actual_dim is not None:
                configs["review_dim"] = actual_dim
            _validate_review_dim(review_embeddings, configs)
            inter_df["review_embedding"] = review_embeddings
        elif compute_review_embedding:
            review_embeddings = [None] * len(inter_df)
            non_empty_idx = [i for i, text in enumerate(inter_df["reviewText"].tolist()) if text]
            if non_empty_idx:
                non_empty_texts = [inter_df.iloc[i]["reviewText"] for i in non_empty_idx]
                predicted_embeddings = get_embedding_batch(model_name, non_empty_texts, batch_size=8, gpu_id=gpu_id)
                for idx, emb in zip(non_empty_idx, predicted_embeddings):
                    review_embeddings[idx] = emb
            os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
            torch.save(review_embeddings, embedding_path)
            actual_dim = _infer_embedding_dim(review_embeddings)
            if actual_dim is not None:
                configs["review_dim"] = actual_dim
            _validate_review_dim(review_embeddings, configs)
            inter_df["review_embedding"] = review_embeddings
        else:
            print("use_review_embedding=True but compute_review_embedding=False and backend is not cached/bert_whitening. Skipping embedding computation.")
            inter_df["review_embedding"] = [None] * len(inter_df)
    else:
        print("Skipping review embeddings (use_review_embedding=False, use_bert_whitening=False)")

    if use_sentiment:
        print("Get sentiment from review...")
        sentiment_path = f"{configs['sentiment_path']}/{dataset}/{sentiment_model.split('/')[-1]}.pt"
        if os.path.exists(sentiment_path):
            cached_sentiment = torch.load(sentiment_path, map_location="cpu")
            review_scores: Any
            if isinstance(cached_sentiment, dict):
                review_sentiments = cached_sentiment.get("review_sentiment")
                review_scores = cast(Any, cached_sentiment.get("review_score"))
            else:
                review_sentiments = cached_sentiment
                review_scores = cast(Any, None)

            if review_sentiments is None or len(review_sentiments) != len(inter_df):
                raise ValueError(
                    f"Cached sentiments length {len(review_sentiments) if review_sentiments is not None else 'None'} != inter_df length {len(inter_df)}"
                )

            if review_scores is None:
                review_scores = [[0.0, 0.0, 0.0] for _ in range(len(inter_df))]
            elif len(review_scores) != len(inter_df):
                raise ValueError(
                    f"Cached sentiment scores length {len(review_scores)} != inter_df length {len(inter_df)}"
                )
        else:
            review_sentiments = ["neutral"] * len(inter_df)
            review_scores: list[list[object]] = [[0.0, 0.0, 0.0] for _ in range(len(inter_df))]

            non_empty_idx = [i for i, text in enumerate(inter_df["reviewText"].tolist()) if text]
            if non_empty_idx:
                non_empty_texts = [inter_df.iloc[i]["reviewText"] for i in non_empty_idx]
                predicted, scores = predict_sentiments(sentiment_model, non_empty_texts, batch_size=32, gpu_id=gpu_id)
                for idx, sentiment, score in zip(non_empty_idx, predicted, scores):
                    review_sentiments[idx] = sentiment
                    review_scores[idx] = list(score)
            os.makedirs(os.path.dirname(sentiment_path), exist_ok=True)
            torch.save(
                {
                    "review_sentiment": review_sentiments,
                    "review_score": review_scores,
                },
                sentiment_path,
            )

        _build_sentiment_features(inter_df, review_sentiments, review_scores)
    else:
        print("Skipping sentiment features (use_sentiment=False)")

    print("Completed Get Embedding and Sentiment Data")
    print()
    print("Apply id mapping...")
    inter_df = apply_id_mapping(inter_df, dataset, configs)
    print("Completed Apply Id Mapping")
    print()
    return inter_df
    
def normalize_review_text(text: object) -> str:
    """Normalize review text; missing/blank values become empty string."""
    if text is None:
        return ""
    if isinstance(text, float) and pd.isna(text):
        return ""
    normalized = str(text).strip()
    return normalized
    
def split_by_reviewgraph(df, seed=42):
    """ReviewGraph-style split: shuffle, first 10% valid, next 10% test, remaining 80% train.

    Migrate valid/test rows with unseen user_id/item_id back to train.
    """
    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    n_valid = max(1, int(n * 0.1))
    n_test = max(1, int(n * 0.1))

    valid_df = shuffled.iloc[:n_valid].reset_index(drop=True)
    test_df = shuffled.iloc[n_valid:n_valid + n_test].reset_index(drop=True)
    train_df = shuffled.iloc[n_valid + n_test:].reset_index(drop=True)

    def _log_sizes(name, tr, va, te):
        def _stats(frame, label):
            if len(frame) == 0:
                return f"{label}: n=0"
            return (f"{label}: n={len(frame)}, "
                    f"rating_mean={frame['rating'].mean():.4f}, "
                    f"rating_std={frame['rating'].std():.4f}, "
                    f"distribution={dict(frame['rating'].value_counts().sort_index())}")
        print(f"[ReviewGraph Split] {name}:")
        print(f"  {_stats(tr, 'train')}")
        print(f"  {_stats(va, 'valid')}")
        print(f"  {_stats(te, 'test')}")

    _log_sizes("before migration", train_df, valid_df, test_df)

    train_users = set(train_df["user_id"].tolist())
    train_items = set(train_df["item_id"].tolist())

    def _migrate(split_df, split_name):
        nonlocal train_df
        moved = []
        kept = []
        for _, row in split_df.iterrows():
            if row["user_id"] not in train_users or row["item_id"] not in train_items:
                moved.append(row)
            else:
                kept.append(row)
        if moved:
            moved_df = pd.DataFrame(moved).reset_index(drop=True)
            train_df = pd.concat([train_df, moved_df], ignore_index=True)
            train_users.update(moved_df["user_id"].tolist())
            train_items.update(moved_df["item_id"].tolist())
            print(f"[ReviewGraph Split] Moved {len(moved_df)} rows from {split_name} -> train "
                  f"(unseen user/item)")
        return pd.DataFrame(kept).reset_index(drop=True) if kept else pd.DataFrame(columns=split_df.columns)

    # Iterative migration: repeat until no more migrations
    max_iter = 20
    for iteration in range(max_iter):
        prev_train_len = len(train_df)
        valid_df = _migrate(valid_df, "valid")
        test_df = _migrate(test_df, "test")
        if len(train_df) == prev_train_len:
            break

    _log_sizes("after migration", train_df, valid_df, test_df)
    return train_df, valid_df, test_df


def split_by_ratio(df, train_ratio=0.8, valid_ratio=0.1, random_state=42):
    test_ratio = 1 - train_ratio - valid_ratio

    train_df, temp_df = train_test_split(
        df, train_size=train_ratio, random_state=random_state, shuffle=True
    )

    valid_df, test_df = train_test_split(
        temp_df,
        test_size=test_ratio / (valid_ratio + test_ratio),
        random_state=random_state,
        shuffle=True
    )

    train_df = cast(pd.DataFrame, train_df)
    valid_df = cast(pd.DataFrame, valid_df)
    test_df = cast(pd.DataFrame, test_df)

    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True), test_df.reset_index(drop=True)

def build_id_mappings(train_df):
    user_ids = sorted(train_df["user_id"].unique())
    item_ids = sorted(train_df["item_id"].unique())

    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {it: i for i, it in enumerate(item_ids)}
    
    return user2idx, item2idx

def apply_id_mapping(inter_df, dataset_name, configs):
    mapping_dir = os.path.join(configs["result_path"], "mappings")
    os.makedirs(mapping_dir, exist_ok=True)
    
    mapping_path = os.path.join(mapping_dir, "id_mappings.pkl")
    
    if os.path.exists(mapping_path):
        with open(mapping_path, "rb") as f:
            mappings = pickle.load(f)
        user2idx = mappings["user2idx"]
        item2idx = mappings["item2idx"]
    else:
        user2idx, item2idx = build_id_mappings(inter_df)
        mappings = {
            "user2idx": user2idx,
            "item2idx": item2idx,
            "idx2user": {v: k for k, v in user2idx.items()},
            "idx2item": {v: k for k, v in item2idx.items()},
        }
        with open(mapping_path, "wb") as f:
            pickle.dump(mappings, f)
    
    inter_df["user_id"] = inter_df["user_id"].map(user2idx)
    inter_df["item_id"] = inter_df["item_id"].map(item2idx)
    
    return inter_df


def get_dataloader(train_df, valid_df, test_df, configs):
    model_name = configs.get('basemodel') or configs.get('model', {}).get('name', 'neumf')
    dataset_cls = DATASET_DICT[model_name]

    def max_id_across_splits(column: str) -> int:
        max_values = [frame[column].max() for frame in (train_df, valid_df, test_df) if len(frame) > 0]
        if not max_values:
            return -1
        return int(max(max_values))

    configs["num_users"] = max_id_across_splits("user_id") + 1
    configs["num_items"] = max_id_across_splits("item_id") + 1

    if model_name == "narre":
        from data.narre_dataset import NARREDataset

        train_dataset = NARREDataset(train_df, configs, split="train", fit_df=valid_df)
        valid_dataset = NARREDataset(valid_df, configs, split="valid", train_dataset=train_dataset)
        test_dataset = NARREDataset(test_df, configs, split="test", train_dataset=train_dataset)
    elif model_name == "ssg":
        from data.ssg_dataset import SSGDataset

        train_dataset = SSGDataset(train_df, configs, split="train")
        valid_dataset = SSGDataset(valid_df, configs, split="valid", train_dataset=train_dataset)
        test_dataset = SSGDataset(test_df, configs, split="test", train_dataset=train_dataset)
    elif model_name == "daml":
        from data.daml_dataset import DAMLDataset

        train_dataset = DAMLDataset(train_df, configs, split="train")
        valid_dataset = DAMLDataset(valid_df, configs, split="valid", train_dataset=train_dataset)
        test_dataset = DAMLDataset(test_df, configs, split="test", train_dataset=train_dataset)
    elif model_name == "sgdn":
        from data.sgdn_dataset import SGDNDataset

        train_dataset = SGDNDataset(train_df, configs, split="train")
        valid_dataset = SGDNDataset(valid_df, configs, split="valid", train_dataset=train_dataset)
        test_dataset = SGDNDataset(test_df, configs, split="test", train_dataset=train_dataset)
    else:
        train_dataset = dataset_cls(train_df, configs, split="train")
        valid_dataset = dataset_cls(valid_df, configs, split="valid")
        test_dataset = dataset_cls(test_df, configs, split="test")

    if hasattr(valid_dataset, "_setup_evaluation"):
        valid_dataset._setup_evaluation(train_df, valid_df, test_df)
    if hasattr(test_dataset, "_setup_evaluation"):
        test_dataset._setup_evaluation(train_df, valid_df, test_df)

    del train_df, valid_df, test_df

    batch_size = configs.get('batch', 256)
    eval_batch_size = configs.get('eval_batch', 4096)
    
    # Use custom collate_fn for full-graph models so DataLoader does not stack
    # graph dictionaries and accidentally add an extra batch dimension.
    if model_name == 'rgcl':
        from data.rgcl_dataset import rgcl_collate_fn
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=rgcl_collate_fn)
        valid_dataloader = DataLoader(valid_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=rgcl_collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=rgcl_collate_fn)
    elif model_name == 'sgdn':
        from data.sgdn_dataset import sgdn_collate_fn
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=sgdn_collate_fn)
        valid_dataloader = DataLoader(valid_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=sgdn_collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=sgdn_collate_fn)
    elif model_name == 'ssg':
        from data.ssg_dataset import ssg_collate_fn
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=ssg_collate_fn)
        valid_dataloader = DataLoader(valid_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=ssg_collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=ssg_collate_fn)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=eval_batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)

    return train_dataloader, valid_dataloader, test_dataloader
