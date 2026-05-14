
from collections.abc import Sequence

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def map_rating_to_sentiment(rating: float, sentiment_mode: int = 3) -> str:
    """Map a numerical rating to a sentiment category.

    3-class mapping rule:
    - 1-2 -> ``negative``
    - 3   -> ``neutral``
    - 4-5 -> ``positive``

    5-class mapping rule:
    - 1 -> ``1``
    - 2 -> ``2``
    - 3 -> ``3``
    - 4 -> ``4``
    - 5 -> ``5``

    Missing or invalid ratings default to ``neutral``.

    Args:
        rating: Numerical rating value.

    Returns:
        One of ``negative``, ``neutral``, or ``positive``.
    """
    if sentiment_mode == 5:
        if pd.isna(rating):
            return "3"
        try:
            rating_value = float(rating)
        except (TypeError, ValueError):
            return "3"
        return str(int(round(rating_value)))

    if pd.isna(rating):
        return "neutral"

    try:
        rating_value = float(rating)
    except (TypeError, ValueError):
        return "neutral"

    if rating_value <= 2.0:
        return "negative"
    if rating_value == 3.0:
        return "neutral"
    return "positive"

def check_consistency(review_sentiment: str, rating_sentiment: str, rating: float, sentiment_mode: int = 3) -> bool:
    """Return whether review sentiment satisfies the rating consistency rule."""
    if sentiment_mode == 5:
        try:
            review_int = int(str(review_sentiment).strip())
            rating_int = int(str(rating_sentiment).strip())
        except (TypeError, ValueError):
            return False

        if review_int in {1, 2, 3}:
            return rating_int in {1, 2, 3}
        if review_int in {4, 5}:
            return rating_int in {4, 5}
        return False

    normalized_review = str(review_sentiment).lower()
    normalized_rating = str(rating_sentiment).lower()

    try:
        rating_value = float(rating) if rating is not None and not pd.isna(rating) else None
    except (TypeError, ValueError):
        rating_value = None

    if rating_value == 3.0:
        return normalized_review in {"neutral", "negative"}

    return normalized_review == normalized_rating

def load_sentiment_model(model_name, gpu_id):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

def normalize_label(label: str, num_classes: int = 3) -> str:
    """Normalize model labels to 3-class or 5-class labels."""
    normalized = str(label).strip().lower()
    if num_classes == 5:
        if "strongly negative" in normalized or normalized.endswith("_0"):
            return "1"
        if "negative or neutral" in normalized or normalized.endswith("_2"):
            return "3"
        if "strongly positive" in normalized or normalized.endswith("_4"):
            return "5"
        if "negative" in normalized or normalized.endswith("_1"):
            return "2"
        if "positive" in normalized or normalized.endswith("_3"):
            return "4"
        return "3"

    if "negative" in normalized or normalized.endswith("_0"):
        return "negative"
    if "neutral" in normalized or normalized.endswith("_1"):
        return "neutral"
    if "positive" in normalized or normalized.endswith("_2"):
        return "positive"
    return "neutral"
    
def predict_sentiments(model_name:str, texts: Sequence[str], batch_size: int = 32, gpu_id: int = 0, num_classes: int = 3) -> list[str]:
    """Predict sentiments for non-empty texts in batches.

    Reviews are tokenized with ``truncation=True`` and ``max_length=512`` to
    satisfy RoBERTa's input length limit.

    Args:
        model_name: Name of the sentiment analysis model.
        texts: List of review texts to analyze.
        batch_size: Number of reviews per batch.
        gpu_id: GPU device ID for inference.

    Returns:
        List of sentiment labels.
    """
    tokenizer, model, device = load_sentiment_model(model_name, gpu_id)
    id2label = model.config.id2label
    predictions: list[str] = []
    scores: list[list[float]] = []

    # 최종 score 순서를 고정: 3-class=[neg, neu, pos], 5-class=[1,2,3,4,5]
    target_order = ["negative", "neutral", "positive"] if num_classes == 3 else ["1", "2", "3", "4", "5"]
    normalized_id2label = {
        i: normalize_label(label, num_classes=num_classes) for i, label in id2label.items()
    }
    label_to_idx = {}
    for i, label in normalized_id2label.items():
        if label not in label_to_idx:
            label_to_idx[label] = i
    reorder_idx = [label_to_idx.get(label, 0) for label in target_order]
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for start in tqdm(
        range(0, len(texts), batch_size), 
        total=total_batches, 
        desc="Sentiment analysis", 
        unit="batch"
    ):

        batch = list(texts[start : start + batch_size])
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).cpu()
            pred_ids = torch.argmax(logits, dim=-1).cpu().tolist()

        batch_labels = [normalize_label(id2label[pred_id], num_classes=num_classes) for pred_id in pred_ids]
        predictions.extend(batch_labels)
        
        # 텐서에서 바로 순서 재배열: [neg, neu, pos]
        batch_scores = probs[:, reorder_idx].cpu().tolist()
        scores.extend([[round(x, 4) for x in row] for row in batch_scores])

    return predictions, scores


def load_pyabsa_model():
    from pyabsa import AspectPolarityClassification as APC
    from pyabsa import available_checkpoints

    ckpts = available_checkpoints()
    # find a suitable checkpoint and use the name:
    sentiment_classifier = APC.SentimentClassifier(
        checkpoint="english"
    )

    return sentiment_classifier

def predict_sentiments_pyabsa(sentiment_classifier, texts):
    """
    Predict sentiments for non-empty texts in batches.

    Args:
        texts: List of review texts to analyze.
        batch_size: Number of reviews per batch.
        gpu_id: GPU device ID for inference.

    Returns:
        List of sentiment labels.
    """
    sentiment_classifier = load_pyabsa_model()
    predictions = []
    for text in texts:
        result = sentiment_classifier.predict(text, print_result=False, save_result=False, ignore_error=True)
        predictions.append(result.sentiment[0].lower())
    return predictions
