import numpy as np


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


def print_results(results):
    for key, value in sorted(results.items()):
        print(f"  {key}: {value:.4f}")
