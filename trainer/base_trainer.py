import os
import json
from tqdm import tqdm
import torch
import numpy as np
import importlib
from typing import Any

_metric = importlib.import_module("metric")
rmse = _metric.rmse
mse = _metric.mse
mae = _metric.mae
print_results = _metric.print_results


class BaseTrainer:
    def __init__(self, model, train_dataloader, valid_dataloader, test_dataloader,
                 configs):
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.configs = configs
        
        self.device = torch.device(f"cuda:{configs.get('gpu', 0)}" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.optimizer = self._create_optimizer()
        
        self.early_stop_patience = configs.get('early_stop_patience', 5)
        self.best_valid_metric = float('inf')
        self.patience_counter = 0
        
        self.result_path = configs['result_path']
        os.makedirs(self.result_path, exist_ok=True)
        
        self.train_log = []
        
        self.best_metric_name = configs.get('best_metric_name', 'rmse')
        self.min_rating = float(configs.get('min_rating', 1.0))
        self.max_rating = float(configs.get('max_rating', 5.0))
        self.eval_clip = self._get_bool_config('eval_clip', True)
        
        self.epoch = configs.get('epoch', 100)
        self.eval_step = configs.get('eval_step', 1)
        self.model_name = configs.get('basemodel') or configs.get('model', {}).get('name', 'unknown')
        self.dataset_name = configs.get('dataset', 'unknown')
    
    def _create_optimizer(self):
        return torch.optim.Adam(
            self.model.parameters(), 
            lr=self.configs.get('lr', 0.001),
            weight_decay=self.configs.get('weight_decay', 0)
        )
    
    def train_epoch(self, epoch_idx):
        raise NotImplementedError("Subclasses must implement train_epoch()")

    def _build_metrics(self, predictions: np.ndarray[Any, Any], ratings: np.ndarray[Any, Any]):
        return {
            'mse': float(mse(predictions, ratings)),
            'rmse': float(rmse(predictions, ratings)),
            'mae': float(mae(predictions, ratings))
        }

    def _get_bool_config(self, key, default=False):
        value = self.configs.get(key, default)
        if isinstance(value, str):
            return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
        return bool(value)

    def _range_value(self, values: np.ndarray[Any, Any], fn):
        if values.size == 0:
            return float('nan')
        return float(fn(values))

    def _build_eval_metrics(self, predictions: np.ndarray[Any, Any], ratings: np.ndarray[Any, Any], phase='valid'):
        raw_predictions = predictions.astype(np.float64, copy=False)
        eval_predictions = (
            np.clip(raw_predictions, self.min_rating, self.max_rating)
            if self.eval_clip
            else raw_predictions
        )

        metrics = self._build_metrics(eval_predictions, ratings)
        abs_errors = np.abs(eval_predictions - ratings)

        rating_dist = {}
        if ratings.size > 0:
            unique, counts = np.unique(ratings, return_counts=True)
            rating_dist = {str(float(u)): int(c) for u, c in zip(unique, counts)}

        metrics.update({
            'num_samples': int(ratings.size),
            'eval_clip': bool(self.eval_clip),
            'min_rating': float(self.min_rating),
            'max_rating': float(self.max_rating),
            'unclipped_prediction_min': self._range_value(raw_predictions, np.min),
            'unclipped_prediction_max': self._range_value(raw_predictions, np.max),
            'clipped_prediction_min': self._range_value(eval_predictions, np.min),
            'clipped_prediction_max': self._range_value(eval_predictions, np.max),
            'label_min': self._range_value(ratings, np.min),
            'label_max': self._range_value(ratings, np.max),
            'rating_mean': self._range_value(ratings, np.mean),
            'rating_std': self._range_value(ratings, np.std),
            'rating_distribution': rating_dist,
            'pred_mean': self._range_value(eval_predictions, np.mean),
            'pred_std': self._range_value(eval_predictions, np.std),
            'pred_min': self._range_value(eval_predictions, np.min),
            'pred_max': self._range_value(eval_predictions, np.max),
            'abs_error_mean': self._range_value(abs_errors, np.mean),
            'abs_error_p90': self._range_value(abs_errors, lambda x: float(np.percentile(x, 90))),
            'abs_error_p95': self._range_value(abs_errors, lambda x: float(np.percentile(x, 95))),
            'abs_error_p99': self._range_value(abs_errors, lambda x: float(np.percentile(x, 99))),
            'abs_error_max': self._range_value(abs_errors, np.max),
        })
        return metrics
    
    def train(self):
        print(f"Starting training for {self.epoch} epochs")
        print(f"Model: {self.model_name}")
        print(f"Dataset: {self.dataset_name}")
        print(f"Device: {self.device}")
        print(f"Best metric: {self.best_metric_name}")

        for epoch in range(self.epoch):
            train_loss_dict = self.train_epoch(epoch)

            log_entry = {
                'epoch': epoch + 1,
                'train': train_loss_dict
            }

            print(f"Epoch {epoch+1}/{self.epoch} - "
                  f"Total Loss: {train_loss_dict.get('total_loss', 0):.4f}")

            if (epoch + 1) % self.eval_step == 0:
                valid_metrics = self.evaluate(self.valid_dataloader, phase='valid')
                log_entry['valid'] = valid_metrics
                
                current_metric = valid_metrics.get(self.best_metric_name, float('inf'))
                
                print("  Validation Metrics:")
                print_results(valid_metrics)
                
                if current_metric < self.best_valid_metric:
                    self.best_valid_metric = current_metric
                    self.patience_counter = 0
                    self.save_checkpoint('best_model.pt')
                    print(f"  New Best! {self.best_metric_name}: {current_metric:.4f}")
                else:
                    self.patience_counter += 1
                    print(f"  Patience: {self.patience_counter}/{self.early_stop_patience}")
                
                if self.patience_counter >= self.early_stop_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            self.train_log.append(log_entry)
        
        print("Training completed!")
        self.save_checkpoint('final_model.pt')
        self.save_logs()
        
        return self.best_valid_metric
    
    @torch.no_grad()
    def evaluate(self, dataloader, phase='valid'):
        self.model.eval()
        
        all_predictions = []
        all_ratings = []
        
        for batch in dataloader:
            ratings = batch["rating"].to(self.device)
            
            predictions = self._predict_batch(batch).view(-1)
            all_predictions.append(predictions.cpu().numpy())
            all_ratings.append(ratings.cpu().numpy())

        if not all_ratings:
            empty = np.array([], dtype=np.float64)
            return self._build_eval_metrics(empty, empty, phase=phase)
         
        predictions = np.concatenate(all_predictions)
        ratings = np.concatenate(all_ratings)
        
        return self._build_eval_metrics(predictions, ratings, phase=phase)
    
    def _predict_batch(self, batch):
        raise NotImplementedError("Subclasses must implement _predict_batch()")
    
    def save_checkpoint(self, filename):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(self.result_path, filename))
    
    def save_logs(self):
        with open(os.path.join(self.result_path, 'training_log.json'), 'w') as f:
            json.dump(self.train_log, f, indent=2)
    
    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
