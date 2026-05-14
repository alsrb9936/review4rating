"""
YAML-based configuration system.
Provides dictionary-like access with attribute access support.
"""

import yaml
import os
from typing import Any, Union, List, Dict, Optional, Iterator


class Config(dict[str, Any]):
    SECTION_KEY_MAP = {
        "training": {
            "batch",
            "eval_batch",
            "epoch",
            "lr",
            "weight_decay",
            "lr_decay",
            "grad_clip",
            "batch_size",
            "train_max_iter",
            "train_lr",
            "train_min_lr",
            "train_lr_decay_factor",
            "train_decay_patience",
            "train_early_stopping_patience",
            "train_grad_clip",
            "train_log_interval",
            "train_valid_interval",
            "train_optimizer",
        },
        "evaluation": {
            "eval_step",
            "early_stop_patience",
            "best_metric_name",
            "metrics",
            "clamp_eval_pred",
            "eval_clip",
            "drop_cold_start_eval",
            "min_rating",
            "max_rating",
        },
        "model": {
            "name",
            "embedding_size",
            "dropout_prob",
            "mf_embedding_size",
            "mlp_embedding_size",
            "mlp_hidden_size",
            "dropout",
            "gcn_dropout",
            "factor_dropout",
            "pred_dropout",
            "node_dropout",
            "review_dim",
            "review_input_mode",
            "use_review",
            "symm",
            "use_contrastive",
            "nd_weight",
            "ed_weight",
            "classification",
            "init_pred_bias_with_rating_mean",
            "match_original_sgdn_dims",
            "num_layers",
            "d_id",
            "d_text",
            "review_emb_dim",
            "review_context_mode",
            "history_aggregation",
            "history_temporal",
            "d_model",
            "num_intents",
            "eta",
            "tau_p",
            "gate_alpha",
            "shared_fusion_scale",
            "residual_fusion_scale",
            "fixed_gate_value",
            "ssg_preset",
        },
        "loss": {
            "loss_preset",
            "lambda_rating",
            "lambda_align",
            "lambda_sep",
            "lambda_recon",
            "lambda_gate",
            "lambda_proto",
            "tau_c",
            "eps",
            "detach_gate_for_align",
            "detach_zx_for_recon",
        },
        "data": {
            "data_path",
            "other_path",
            "embedding_path",
            "sentiment_path",
            "sentiment_model",
            "language_model",
            "use_review_text",
            "use_review_embedding",
            "compute_review_embedding",
            "use_sentiment",
            "review_feature_backend",
            "review_emb_path",
            "use_bert_whitening",
            "bert_whitening_model",
            "bert_whitening_dim",
            "bert_whitening_pooling",
            "bert_whitening_normalize",
            "bert_whitening_cache_scope",
            "bert_whitening_stats_path",
        },
    }
    """
    Configuration class that loads from YAML files.
    
    Supports:
    - Dictionary-style access: config['lr']
    - Attribute-style access: config.lr
    - Nested access: config.training.batch
    - Multiple YAML file merging
    - Runtime overrides
    
    Example:
        config = Config.from_yaml(['default.yaml', 'neumf.yaml'])
        config.merge({'lr': 0.0005})
        print(config.lr)
        print(config['training']['batch'])
    """
    
    @classmethod
    def from_yaml(cls, paths: Union[str, List[str]], model_name: Optional[str] = None) -> 'Config':
        """Load configuration from YAML file(s)."""
        config = cls()
        
        if isinstance(paths, str):
            paths = [paths]
        
        if model_name:
            model_config_path = os.path.join(
                os.path.dirname(__file__), 'yaml', f'{model_name}.yaml'
            )
            if os.path.exists(model_config_path):
                paths.append(model_config_path)
        
        for path in paths:
            if not os.path.exists(path):
                alt_path = os.path.join(os.path.dirname(__file__), 'yaml', path)
                if os.path.exists(alt_path):
                    path = alt_path
                else:
                    raise FileNotFoundError(f"Config file not found: {path}")
            
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                if data:
                    config._merge_nested(data)
        
        return config
    
    def _iter_leaf_items(self) -> Iterator[tuple[str, Any]]:
        for key, value in self.items():
            if isinstance(value, Config):
                yield from value._iter_leaf_items()
            else:
                yield key, value

    def _sync_flat_key_to_section(self, key: str, value: Any) -> None:
        for section_name, section_keys in self.SECTION_KEY_MAP.items():
            if key not in section_keys:
                continue

            section = self.get(section_name)
            if not isinstance(section, Config):
                section = Config()
                super().__setitem__(section_name, section)
            section[key] = value
            return

    def _merge_nested(self, data: Dict[str, Any], prefix: str = "") -> None:
        """Recursively merge nested dictionaries while preserving flat access."""
        for key, value in data.items():
            if isinstance(value, dict):
                existing = self.get(key)
                if not isinstance(existing, Config):
                    existing = Config()
                    super().__setitem__(key, existing)
                existing._merge_nested(value, prefix)
                for nested_key, nested_value in existing._iter_leaf_items():
                    super().__setitem__(nested_key, nested_value)
            else:
                super().__setitem__(key, value)
                self._sync_flat_key_to_section(key, value)
    
    def merge(self, other: Union[Dict[str, Any], 'Config']) -> 'Config':
        """Merge another dictionary or Config into this one."""
        if isinstance(other, Config):
            other = dict(other)
        
        for key, value in other.items():
            if isinstance(value, dict) and key in self and isinstance(self[key], dict):
                if not isinstance(self[key], Config):
                    self[key] = Config(self[key])
                self[key].merge(value)
                for nested_key, nested_value in self[key]._iter_leaf_items():
                    super().__setitem__(nested_key, nested_value)
            else:
                super().__setitem__(key, value)
                self._sync_flat_key_to_section(key, value)
                if '.' in key:
                    parts = key.split('.')
                    target = self
                    for part in parts[:-1]:
                        if part not in target:
                            target[part] = Config()
                        target = target[part]
                    target[parts[-1]] = value
        
        return self
    
    def __getattr__(self, key: str) -> Any:
        """Allow attribute-style access."""
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")
    
    def __setattr__(self, key: str, value: Any) -> None:
        """Allow attribute-style setting."""
        super().__setitem__(key, value)
        self._sync_flat_key_to_section(key, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value with default, supports nested keys with dots."""
        if '.' in key:
            parts = key.split('.')
            target = self
            for part in parts:
                if isinstance(target, dict) and part in target:
                    target = target[part]
                else:
                    return default
            return target
        return super().get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Config to regular dictionary."""
        result = {}
        for key, value in self.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def save_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = self._to_nested_dict()
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    def _to_nested_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}

        for section_name in self.SECTION_KEY_MAP:
            section = self.get(section_name)
            if isinstance(section, Config):
                result[section_name] = section.to_dict()

        for key, value in self.items():
            if isinstance(value, Config):
                if key not in result:
                    result[key] = value.to_dict()
                continue

            was_mapped = False
            for section_name, section_keys in self.SECTION_KEY_MAP.items():
                if key in section_keys:
                    result.setdefault(section_name, {})[key] = value
                    was_mapped = True
                    break

            if not was_mapped:
                result[key] = value
        return result
    
    def sync_args(self, args) -> Any:
        args.batch = self.get('batch', 256)
        args.eval_batch = self.get('eval_batch', 4096)
        args.epoch = self.get('epoch', 100)
        args.eval_step = self.get('eval_step', 1)
        return args

    def __repr__(self) -> str:
        """String representation."""
        items = [f"{k}={v!r}" for k, v in list(self.items())[:10]]
        if len(self) > 10:
            items.append(f"... and {len(self) - 10} more items")
        return f"Config({', '.join(items)})"
