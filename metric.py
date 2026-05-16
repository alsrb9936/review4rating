import numpy as np
from collections import defaultdict


def rmse(predictions, ground_truth):
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    return np.sqrt(np.mean((predictions - ground_truth) ** 2))

def mse(predictions, ground_truth):
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    return np.mean((predictions - ground_truth) ** 2)

def mae(predictions, ground_truth):
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    return np.mean(np.abs(predictions - ground_truth))


def dcg_at_k(relevances, k):
    relevances = np.array(relevances)[:k]
    if len(relevances) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    return np.sum(relevances / discounts)


def ndcg_at_k(predictions, ground_truth, user_ids, k=10, rating_threshold=4.0):
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    user_ids = np.array(user_ids)
    
    user_data = defaultdict(lambda: {'preds': [], 'ratings': []})
    for i in range(len(user_ids)):
        uid = user_ids[i]
        user_data[uid]['preds'].append(predictions[i])
        user_data[uid]['ratings'].append(ground_truth[i])
    
    ndcg_scores = []
    
    for uid, data in user_data.items():
        preds = np.array(data['preds'])
        ratings = np.array(data['ratings'])
        
        relevances = (ratings >= rating_threshold).astype(float)
        
        if np.sum(relevances) == 0:
            continue
        
        sorted_indices = np.argsort(-preds)
        sorted_relevances = relevances[sorted_indices]
        
        dcg = dcg_at_k(sorted_relevances, k)
        
        ideal_relevances = np.sort(relevances)[::-1]
        idcg = dcg_at_k(ideal_relevances, k)
        
        if idcg > 0:
            ndcg_scores.append(dcg / idcg)
    
    return np.mean(ndcg_scores) if ndcg_scores else 0.0


def hitrate_at_k(predictions, ground_truth, user_ids, k=10, rating_threshold=4.0):
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    user_ids = np.array(user_ids)
    
    user_data = defaultdict(lambda: {'preds': [], 'ratings': []})
    for i in range(len(user_ids)):
        uid = user_ids[i]
        user_data[uid]['preds'].append(predictions[i])
        user_data[uid]['ratings'].append(ground_truth[i])
    
    hit_scores = []
    
    for uid, data in user_data.items():
        preds = np.array(data['preds'])
        ratings = np.array(data['ratings'])
        
        relevances = (ratings >= rating_threshold).astype(float)
        
        if np.sum(relevances) == 0:
            continue
        
        sorted_indices = np.argsort(-preds)
        top_k_relevances = relevances[sorted_indices[:k]]
        
        hit = 1.0 if np.sum(top_k_relevances) > 0 else 0.0
        hit_scores.append(hit)
    
    return np.mean(hit_scores) if hit_scores else 0.0


def recall_at_k(predictions, ground_truth, user_ids, k=10, rating_threshold=4.0):
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    user_ids = np.array(user_ids)
    
    user_data = defaultdict(lambda: {'preds': [], 'ratings': []})
    for i in range(len(user_ids)):
        uid = user_ids[i]
        user_data[uid]['preds'].append(predictions[i])
        user_data[uid]['ratings'].append(ground_truth[i])
    
    recall_scores = []
    
    for uid, data in user_data.items():
        preds = np.array(data['preds'])
        ratings = np.array(data['ratings'])
        
        relevances = (ratings >= rating_threshold).astype(float)
        
        total_relevant = np.sum(relevances)
        if total_relevant == 0:
            continue
        
        sorted_indices = np.argsort(-preds)
        top_k_relevances = relevances[sorted_indices[:k]]
        
        recall = np.sum(top_k_relevances) / total_relevant
        recall_scores.append(recall)
    
    return np.mean(recall_scores) if recall_scores else 0.0


def print_results(results):
    for key in ["mse", "rmse", "mae"]:
        if key in results:
            value = results[key]
            if isinstance(value, (int, float, np.integer, np.floating)):
                print(f"  {key}: {float(value):.4f}")
            else:
                print(f"  {key}: {value}")
    
    ranking_keys = [k for k in results.keys() if any(k.startswith(m) for m in ["ndcg@", "hitrate@", "recall@"])]
    for key in sorted(ranking_keys):
        value = results[key]
        if isinstance(value, (int, float, np.integer, np.floating)):
            print(f"  {key}: {float(value):.4f}")
        else:
            print(f"  {key}: {value}")
