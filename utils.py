
import os
import random
from typing import Any, cast
import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd
import pickle

from data import DATASET_DICT
from data.process.embeddings import get_embedding_batch
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
    gpu_id = configs.get('gpu', 3)
    model_name = configs['language_model']
    sentiment_model = configs['sentiment_model']
    dataset = configs['dataset']

    inter_path = f"{path}/{dataset}/{dataset}.inter"
    review_path = f"{path}/{dataset}/{dataset}.review"
    configured_review_emb_path = configs.get("review_emb_path")
    embedding_path = configured_review_emb_path or f"{configs['embedding_path']}/{dataset}/{model_name.split('/')[-1]}.pt"
    sentiment_path = f"{configs['sentiment_path']}/{dataset}/{sentiment_model.split('/')[-1]}.pt"

    inter_df = pd.read_csv(inter_path, sep="\t")
    review_df = pd.read_csv(review_path, sep="\t")
    
    # merge inter_df and review_df
    inter_df = pd.merge(
        inter_df,
        review_df,
        on=["user_id:token", "item_id:token"],
        how="inner"
    )
    inter_df.columns = inter_df.columns.str.split(':').str[0]
    inter_df["reviewText"] = inter_df["reviewText"].apply(normalize_review_text)

    print("Completed Loading Interaction Data")

    print()
    print("Get embedding from review. . . ")
    if configured_review_emb_path is not None and not os.path.exists(embedding_path):
        raise FileNotFoundError(f"Review embedding file not found: {embedding_path}")

    if os.path.exists(embedding_path):
        review_embeddings = _load_cached_review_embeddings(embedding_path)
        if len(review_embeddings) != len(inter_df):
            raise ValueError(
                f"Cached embeddings length {len(review_embeddings)} != inter_df length {len(inter_df)}"
            )
    else:
        review_embeddings = [None] * len(inter_df)
        non_empty_idx = [i for i, text in enumerate(inter_df["reviewText"].tolist()) if text]
        if non_empty_idx:
            non_empty_texts = [inter_df.iloc[i]["reviewText"] for i in non_empty_idx]
            predicted_embeddings = get_embedding_batch(model_name, non_empty_texts, batch_size=8, gpu_id=gpu_id)

            for idx, emb in zip(non_empty_idx, predicted_embeddings):
                review_embeddings[idx] = emb
        os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
        torch.save(review_embeddings, embedding_path)
    inter_df["review_embedding"] = review_embeddings


    if configs['sentiment']:
        print("Get sentiment from review. . . ")
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
    
    print("Completed Get Embedding and Sentiment Data")
    print()
    print("Apply id mapping. . . ")
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

    configs["num_users"] = int(
        max(train_df["user_id"].max(), valid_df["user_id"].max(), test_df["user_id"].max())
    ) + 1
    configs["num_items"] = int(
        max(train_df["item_id"].max(), valid_df["item_id"].max(), test_df["item_id"].max())
    ) + 1

    train_dataset = dataset_cls(train_df, configs, split="train")
    if model_name == "narre":
        valid_dataset = dataset_cls(valid_df, configs, split="valid", train_dataset=train_dataset)
        test_dataset = dataset_cls(test_df, configs, split="test", train_dataset=train_dataset)
    else:
        valid_dataset = dataset_cls(valid_df, configs, split="valid")
        test_dataset = dataset_cls(test_df, configs, split="test")

    if hasattr(valid_dataset, "_setup_evaluation"):
        valid_dataset._setup_evaluation(train_df, valid_df, test_df)
    if hasattr(test_dataset, "_setup_evaluation"):
        test_dataset._setup_evaluation(train_df, valid_df, test_df)

    del train_df, valid_df, test_df

    batch_size = configs.get('batch', 256)
    eval_batch_size = configs.get('eval_batch', 4096)
    
    # Use custom collate_fn for RGCL to handle sparse tensors
    if model_name == 'rgcl':
        from data.rgcl_dataset import rgcl_collate_fn
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=rgcl_collate_fn)
        valid_dataloader = DataLoader(valid_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=rgcl_collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=rgcl_collate_fn)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=eval_batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)

    return train_dataloader, valid_dataloader, test_dataloader
