import argparse
import os
from datetime import datetime

import torch
import json

from config import Config
from model import MODEL_DICT
from utils import attach_bert_whitening_review_features, set_seed, load_interaction_data, split_by_ratio, split_by_reviewgraph, get_dataloader
from trainer import MODEL_TRAINER_DICT
from metric import print_results


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def apply_iard_loss_preset(configs):
    model_name = configs.get('basemodel') or configs.get('model', {}).get('name')
    if model_name != 'iard_rm':
        return

    preset = configs.get('loss_preset', 'full_iard')
    full_defaults = {
        'eta': 0.1,
        'shared_fusion_scale': 1.0,
        'residual_fusion_scale': 1.0,
        'fixed_gate_value': None,
        'lambda_align': 0.1,
        'lambda_sep': 0.05,
        'lambda_recon': 0.1,
        'lambda_gate': 0.001,
        'lambda_proto': 0.01,
    }
    preset_overrides = {
        'rating_only': {
            **full_defaults,
            'shared_fusion_scale': 0.0,
            'residual_fusion_scale': 0.0,
            'lambda_align': 0.0,
            'lambda_sep': 0.0,
            'lambda_recon': 0.0,
            'lambda_gate': 0.0,
            'lambda_proto': 0.0,
        },
        'review_fusion': {
            **full_defaults,
            'shared_fusion_scale': 1.0,
            'residual_fusion_scale': 1.0,
            'lambda_align': 0.0,
            'lambda_sep': 0.0,
            'lambda_recon': 0.0,
            'lambda_gate': 0.0,
            'lambda_proto': 0.0,
        },
        'align_only': {
            **full_defaults,
            'shared_fusion_scale': 1.0,
            'residual_fusion_scale': 0.0,
            'lambda_align': 0.1,
            'lambda_sep': 0.0,
            'lambda_recon': 0.0,
            'lambda_gate': 0.0,
            'lambda_proto': 0.0,
        },
        'full_iard': {
            **full_defaults,
        },
        'full_no_sep': {
            **full_defaults,
            'lambda_sep': 0.0,
        },
        'full_no_residual_pred': {
            **full_defaults,
            'residual_fusion_scale': 0.0,
        },
        'full_fixed_gate': {
            **full_defaults,
            'fixed_gate_value': 0.5,
        },
        'full_eta_0_3': {
            **full_defaults,
            'eta': 0.3,
        },
        'full_low_align': {
            **full_defaults,
            'lambda_align': 0.05,
        },
    }
    if preset not in preset_overrides:
        raise ValueError(f"Unsupported IARD loss preset: {preset}")
    configs.merge(preset_overrides[preset])


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Model name")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name")
    parser.add_argument("--data_path", type=str, default="/home/infolab/mnt/mingyu/review_rec/dataset", help="Data path")
    parser.add_argument("--gpu", type=int, default=3, help="GPU ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mode", type=str, default="train", help="train or eval")
    
    parser.add_argument("--result_path", type=str, default=None, help="Path to results folder (for eval mode)")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--eval_batch", type=int, default=None)
    parser.add_argument("--eval_step", type=int, default=None, help="Evaluation every N epochs")
    parser.add_argument("--sentiment", type=bool, default=False, help="Only evaluate the model")
    parser.add_argument("--review_emb_path", type=str, default=None, help="Path to precomputed review embeddings")
    parser.add_argument("--review_feature_backend", type=str, default=None, choices=["sentence_transformer", "bert_whitening", "cached"], help="Review feature backend")
    parser.add_argument("--review_dim", type=int, default=None, help="Review embedding dimension")
    parser.add_argument("--bert_whitening_dim", type=int, default=None, help="Output dimension for BERT-Whitening review features")
    parser.add_argument("--review_input_mode", type=str, default=None, choices=["token", "embedding"], help="SSG review input mode")
    parser.add_argument("--ssg_preset", type=str, default=None, choices=["custom", "set_only", "set_sequence", "set_graph", "full", "no_decov"], help="SSG ablation preset")
    parser.add_argument("--overfit_n", type=int, default=None, help="Train on only N interactions for overfit debugging")
    parser.add_argument("--split_protocol", type=str, default=None, choices=["default", "reviewgraph"], help="Data split protocol")
    parser.add_argument("--loss_preset", type=str, default=None, help="IARD loss preset: rating_only, review_fusion, align_only, full_iard, full_no_sep, full_no_residual_pred, full_fixed_gate, full_eta_0_3, full_low_align")
    parser.add_argument("--eval_clip", type=str_to_bool, default=None, help="Clip validation/test predictions to [min_rating, max_rating]")
    parser.add_argument("--drop_cold_start_eval", type=str_to_bool, default=None, help="Drop valid/test rows with users/items unseen in train")
    
    return parser.parse_args()


def setup_environment(configs):
    set_seed(configs.get('seed', 42))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    model_name = configs.get('basemodel') or configs.get('model', {}).get('name', 'unknown')
    result_path = os.path.join("results", f"{model_name}_{configs.get('seed', 42)}_{configs.get('dataset', 'unknown')}_{timestamp}")
    os.makedirs(result_path, exist_ok=True)
    configs["result_path"] = result_path
    return result_path


def _config_bool(configs, key, default=False):
    value = configs.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def diagnose_and_filter_cold_start(train_df, valid_df, test_df, configs):
    train_users = set(train_df["user_id"].tolist())
    train_items = set(train_df["item_id"].tolist())
    drop_cold = _config_bool(configs, "drop_cold_start_eval", True)
    summary = {}

    def process_split(name, frame):
        if len(frame) == 0:
            stats = {
                "num_samples_before": 0,
                "cold_user_count": 0,
                "cold_item_count": 0,
                "cold_interaction_count": 0,
                "removed_count": 0,
                "num_samples_after": 0,
            }
            print(f"Cold-start {name}: empty split; drop_cold_start_eval={drop_cold}")
            return frame.reset_index(drop=True), stats

        user_seen = frame["user_id"].isin(train_users)
        item_seen = frame["item_id"].isin(train_items)
        keep_mask = user_seen & item_seen
        cold_user_count = int((~user_seen).sum())
        cold_item_count = int((~item_seen).sum())
        cold_interaction_count = int((~keep_mask).sum())
        filtered = frame.loc[keep_mask].reset_index(drop=True) if drop_cold else frame.reset_index(drop=True)
        removed_count = cold_interaction_count if drop_cold else 0
        stats = {
            "num_samples_before": int(len(frame)),
            "cold_user_count": cold_user_count,
            "cold_item_count": cold_item_count,
            "cold_interaction_count": cold_interaction_count,
            "removed_count": int(removed_count),
            "num_samples_after": int(len(filtered)),
        }
        print(
            f"Cold-start {name}: before={stats['num_samples_before']}, "
            f"cold_user_rows={cold_user_count}, cold_item_rows={cold_item_count}, "
            f"cold_either_rows={cold_interaction_count}, removed={removed_count}, "
            f"after={stats['num_samples_after']}, drop_cold_start_eval={drop_cold}"
        )
        return filtered, stats

    valid_df, summary["valid"] = process_split("valid", valid_df)
    test_df, summary["test"] = process_split("test", test_df)
    configs["cold_start_eval_summary"] = summary
    return valid_df, test_df


def prepare_data(configs):
    print("prepare data")
    inter_df = load_interaction_data(configs)
    split_protocol = configs.get('split_protocol', 'default')
    if split_protocol == 'reviewgraph':
        train_df, valid_df, test_df = split_by_reviewgraph(inter_df, seed=configs.get('seed', 42))
    else:
        train_df, valid_df, test_df = split_by_ratio(inter_df, random_state=configs.get('seed', 42))
    overfit_n = int(configs.get('overfit_n', 0) or 0)
    if overfit_n > 0:
        overfit_n = min(overfit_n, len(train_df))
        train_df = train_df.iloc[:overfit_n].reset_index(drop=True)
        valid_df = train_df.copy()
        test_df = train_df.copy()
        print(
            f"overfit mode enabled: using the same {overfit_n} train interactions for train/valid/test. "
            "These metrics are debug-only and intentionally leak train data into validation/test."
        )
    valid_df, test_df = diagnose_and_filter_cold_start(train_df, valid_df, test_df, configs)
    train_df, valid_df, test_df = attach_bert_whitening_review_features(train_df, valid_df, test_df, configs)
    train_loader, valid_loader, test_loader = get_dataloader(train_df, valid_df, test_df, configs)
    return {
        'train_df': train_df,
        'valid_df': valid_df,
        'test_df': test_df,
        'train_loader': train_loader,
        'valid_loader': valid_loader,
        'test_loader': test_loader
    }


def create_trainer(model_name, data_dict, configs):
    device = torch.device(f"cuda:{configs.get('gpu', 0)}" if torch.cuda.is_available() else "cpu")

    train_dataset = data_dict['train_loader'].dataset

    model_cls = MODEL_DICT[model_name]
    model = model_cls(configs, train_dataset)

    trainer_cls = MODEL_TRAINER_DICT[model_name]
    trainer = trainer_cls(
        model=model,
        train_dataloader=data_dict['train_loader'],
        valid_dataloader=data_dict['valid_loader'],
        test_dataloader=data_dict['test_loader'],
        configs=configs
    )

    return trainer


def train_mode(configs, data_dict):
    result_path = configs["result_path"]

    device = torch.device(f"cuda:{configs.get('gpu', 0)}" if torch.cuda.is_available() else "cpu")

    print(f"Creating model and trainer...")
    model_name = configs.get('basemodel') or configs.get('model', {}).get('name', 'unknown')
    trainer = create_trainer(model_name, data_dict, configs)

    print(f"Starting training...")
    best_valid_metric = trainer.train()

    print("\n" + "="*50)
    print("Loading best model and evaluating on test set...")
    print("="*50)
    if int(configs.get('overfit_n', 0) or 0) > 0:
        print("WARNING: overfit mode is active. Reported validation/test metrics are not generalization metrics.")

    best_model_path = os.path.join(result_path, 'best_model.pt')
    trainer.load_checkpoint(best_model_path)

    test_metrics = trainer.evaluate(data_dict['test_loader'], phase='test')
    test_metrics['best_valid_metric'] = float(best_valid_metric)
    inspect_batch_fn = getattr(trainer, 'inspect_batch', None)
    if int(configs.get('overfit_n', 0) or 0) > 0 and callable(inspect_batch_fn):
        print("debug train batch inspect:", inspect_batch_fn(dataloader=data_dict['train_loader'], backward=False))
    print("\nTest Results:")
    print_results(test_metrics)

    with open(os.path.join(result_path, 'test_results.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\nResults saved to {result_path}/test_results.json")


def eval_mode(args):
    if args.result_path is None:
        raise ValueError("--result_path is required for eval mode")

    config_path = os.path.join(args.result_path, 'config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    print(f"Loading config from {config_path}")
    configs = Config.from_yaml(config_path)
    configs['data_path'] = args.data_path
    configs['result_path'] = args.result_path
    configs['gpu'] = args.gpu
    if args.review_emb_path is not None:
        configs['review_emb_path'] = args.review_emb_path
    if args.review_feature_backend is not None:
        configs['review_feature_backend'] = args.review_feature_backend
    if args.review_dim is not None:
        configs['review_dim'] = args.review_dim
    if args.bert_whitening_dim is not None:
        configs['bert_whitening_dim'] = args.bert_whitening_dim
        configs['review_dim'] = args.bert_whitening_dim
    if args.review_input_mode is not None:
        configs['review_input_mode'] = args.review_input_mode
    if args.ssg_preset is not None:
        configs['ssg_preset'] = args.ssg_preset
    if args.split_protocol is not None:
        configs['split_protocol'] = args.split_protocol
    if args.overfit_n is not None:
        configs['overfit_n'] = args.overfit_n
    if args.loss_preset is not None:
        configs['loss_preset'] = args.loss_preset
    if args.eval_clip is not None:
        configs['eval_clip'] = args.eval_clip
    if args.drop_cold_start_eval is not None:
        configs['drop_cold_start_eval'] = args.drop_cold_start_eval
    apply_iard_loss_preset(configs)

    model_name = configs.get('basemodel') or configs.get('model', {}).get('name')
    dataset = configs.get('dataset')

    if not model_name:
        raise ValueError("Model name not found in config")
    if not dataset:
        raise ValueError("Dataset not found in config")

    checkpoint_path = os.path.join(args.result_path, 'best_model.pt')
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(args.result_path, 'final_model.pt')

    print(f"Loading checkpoint from {checkpoint_path}")

    print(f"Model: {model_name}")
    print(f"Dataset: {dataset}")

    print(f"Preparing data...")
    data_dict = prepare_data(configs)

    print(f"Creating model and trainer...")
    trainer = create_trainer(model_name, data_dict, configs)

    trainer.load_checkpoint(checkpoint_path)

    print("\n" + "="*50)
    print("Evaluating on test set...")
    print("="*50)

    test_metrics = trainer.evaluate(data_dict['test_loader'], phase='test')
    print("\nTest Results:")
    print_results(test_metrics)

    eval_result_path = os.path.join(args.result_path, 'eval_results.json')
    with open(eval_result_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\nResults saved to {eval_result_path}")


def main():
    args = args_parser()
    
    if args.mode == "eval":
        eval_mode(args)
        return
    
    configs = Config.from_yaml(
        paths=['default.yaml'],
        model_name=args.model
    )
    
    cli_overrides = {
        'batch': args.batch,
        'eval_batch': args.eval_batch,
        'epoch': args.epoch,
        'eval_step': args.eval_step,
        'gpu': args.gpu,
        'seed': args.seed,
        'review_emb_path': args.review_emb_path,
        'review_feature_backend': args.review_feature_backend,
        'review_dim': args.review_dim,
        'bert_whitening_dim': args.bert_whitening_dim,
        'review_input_mode': args.review_input_mode,
        'ssg_preset': args.ssg_preset,
        'split_protocol': args.split_protocol,
        'overfit_n': args.overfit_n,
        'loss_preset': args.loss_preset,
        'eval_clip': args.eval_clip,
        'drop_cold_start_eval': args.drop_cold_start_eval,
    }
    configs.merge({k: v for k, v in cli_overrides.items() if v is not None})
    
    configs['dataset'] = args.dataset
    if not configs.get('basemodel'):
        configs['basemodel'] = args.model
    configs['sentiment'] = args.sentiment
    if args.review_dim is not None:
        configs['review_dim'] = args.review_dim
    if args.bert_whitening_dim is not None:
        configs['review_dim'] = args.bert_whitening_dim
    apply_iard_loss_preset(configs)
    
    setup_environment(configs)
    data_dict = prepare_data(configs)

    config_save_path = os.path.join(configs.result_path, 'config.yaml')
    configs.save_yaml(config_save_path)
    print(f"Config saved to {config_save_path}")

    train_mode(configs, data_dict)


if __name__ == "__main__":
    main()
