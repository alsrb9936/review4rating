#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml


REPO_ROOT = Path(__file__).resolve().parent
MA_YAML_PATH = REPO_ROOT / "config" / "yaml" / "ma_rgcl.yaml"


SEARCH_SPACE = {
    "training.lr": [0.003, 0.005, 0.01, 0.02],
    "training.weight_decay": [0.0, 1e-6, 1e-5, 1e-4],
    "training.grad_clip": [0.5, 1.0, 2.0, 5.0],
    "model.embedding_size": [32, 64, 128],
    "model.num_layers": [1, 2],
    "model.dropout": [0.3, 0.5, 0.7],
    "model.node_dropout": [0.5, 0.7, 0.8],
    "model.nd_weight": [0.1, 0.3, 0.5],
    "model.ed_weight": [0.3, 0.7, 1.0],
    "model.align_weight": [0.01, 0.02, 0.05],
    "model.residual_weight": [0.0005, 0.001, 0.002],
    "model.gate_hidden_dim": [32, 64, 128],
    "model.epsilon": [0.01, 0.05, 0.1],
    "model.lambda_residual": [0.1, 0.25, 0.5],
    "model.review_generation_weight": [0.01, 0.05, 0.1],
}


def get_nested(config: dict, dotted_key: str):
    cur = config
    for key in dotted_key.split("."):
        cur = cur[key]
    return cur


def set_nested(config: dict, dotted_key: str, value):
    cur = config
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        cur = cur[key]
    cur[keys[-1]] = value


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, config: dict):
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)


def patch_ma_yaml(base_config: dict, params: dict):
    config = deepcopy(base_config)
    for key, value in params.items():
        set_nested(config, key, value)
    save_yaml(MA_YAML_PATH, config)


def extract_result_json(stdout: str) -> Optional[Path]:
    match = re.search(r"Results saved to\s+(.+?test_results\.json)", stdout)
    if match:
        return REPO_ROOT / match.group(1).strip()

    candidates = list((REPO_ROOT / "results").glob("**/test_results.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_train(args, trial_params: dict, trial_name: str) -> dict:
    cmd = [
        sys.executable,
        "main.py",
        "--model", "ma_rgcl",
        "--dataset", args.dataset,
        "--mode", "train",
        "--split_protocol", args.split_protocol,
        "--gpu", str(args.gpu),
        "--seed", str(args.seed),
    ]

    if args.epoch is not None:
        cmd += ["--epoch", str(args.epoch)]

    print("\n" + "=" * 90)
    print(f"[TRIAL] {trial_name}")
    print(f"[PARAMS] {trial_params}")
    print("[CMD]", " ".join(cmd))
    print("=" * 90)

    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    print(proc.stdout)

    result_json = extract_result_json(proc.stdout)

    result = {
        "trial_name": trial_name,
        "params": json.dumps(trial_params, ensure_ascii=False),
        "returncode": proc.returncode,
        "result_json": str(result_json) if result_json else "",
        args.metric: None,
        "status": "failed",
    }

    if proc.returncode != 0:
        return result

    if result_json is None or not result_json.exists():
        return result

    with result_json.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    if args.metric not in metrics:
        result["status"] = f"metric_not_found:{args.metric}"
        result["available_metrics"] = json.dumps(list(metrics.keys()))
        return result

    result[args.metric] = float(metrics[args.metric])
    result["all_metrics"] = json.dumps(metrics, ensure_ascii=False)
    result["status"] = "ok"
    return result


def is_better(new_score, best_score, mode: str) -> bool:
    if new_score is None:
        return False
    if best_score is None:
        return True
    if mode == "min":
        return new_score < best_score
    return new_score > best_score


def write_csv_row(csv_path: Path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "trial_name",
        "status",
        "returncode",
        "params",
        "result_json",
        "rmse",
        "mse",
        "mae",
        "all_metrics",
        "available_metrics",
    ]

    normalized = {k: row.get(k, "") for k in fieldnames}
    exists = csv_path.exists()

    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(normalized)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Amazon_Office_Products_14")
    parser.add_argument("--split_protocol", type=str, default="reviewgraph")
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric", type=str, default="rmse")
    parser.add_argument("--mode", type=str, choices=["min", "max"], default="min")
    parser.add_argument("--max_rounds", type=int, default=3)
    parser.add_argument("--epoch", type=int, default=None)
    args = parser.parse_args()

    if not MA_YAML_PATH.exists():
        raise FileNotFoundError(f"Cannot find {MA_YAML_PATH}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "results" / "greedy_search_ma_rgcl" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "search_log.csv"
    best_path = out_dir / "best_config.json"

    original_yaml_text = MA_YAML_PATH.read_text(encoding="utf-8")
    base_config = load_yaml(MA_YAML_PATH)

    current_params = {}
    for key in SEARCH_SPACE:
        current_params[key] = get_nested(base_config, key)

    best_params = deepcopy(current_params)
    best_score = None
    best_result = None

    try:
        patch_ma_yaml(base_config, best_params)
        baseline = run_train(args, best_params, "baseline")
        write_csv_row(csv_path, baseline)

        if baseline["status"] == "ok":
            best_score = baseline[args.metric]
            best_result = baseline
        for round_idx in range(1, args.max_rounds + 1):
            print("\n" + "#" * 90)
            print(f"[ROUND {round_idx}] current best {args.metric}={best_score}, params={best_params}")
            print("#" * 90)

            improved = False

            for param_name, candidates in SEARCH_SPACE.items():
                local_best_value = best_params[param_name]
                local_best_score = best_score
                local_best_result = best_result

                for value in candidates:
                    if value == best_params[param_name]:
                        continue

                    trial_params = deepcopy(best_params)
                    trial_params[param_name] = value

                    patch_ma_yaml(base_config, trial_params)

                    safe_value = str(value).replace("/", "_")
                    trial_name = f"round{round_idx}_{param_name}_{safe_value}"

                    result = run_train(args, trial_params, trial_name)
                    write_csv_row(csv_path, result)

                    score = result.get(args.metric)
                    if result["status"] == "ok" and is_better(score, local_best_score, args.mode):
                        local_best_score = score
                        local_best_value = value
                        local_best_result = result

                if local_best_value != best_params[param_name]:
                    print(
                        f"[UPDATE] {param_name}: "
                        f"{best_params[param_name]} -> {local_best_value}, "
                        f"{args.metric}: {best_score} -> {local_best_score}"
                    )
                    best_params[param_name] = local_best_value
                    best_score = local_best_score
                    best_result = local_best_result
                    improved = True

                    with best_path.open("w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "metric": args.metric,
                                "mode": args.mode,
                                "best_score": best_score,
                                "best_params": best_params,
                                "best_result": best_result,
                            },
                            f,
                            indent=2,
                            ensure_ascii=False,
                        )

            if not improved:
                print(f"[STOP] No improvement at round {round_idx}.")
                break

        patch_ma_yaml(base_config, best_params)

        print("\n" + "=" * 90)
        print("[DONE] Greedy search finished.")
        print(f"[BEST {args.metric}] {best_score}")
        print(f"[BEST PARAMS] {json.dumps(best_params, indent=2, ensure_ascii=False)}")
        print(f"[LOG] {csv_path}")
        print(f"[BEST JSON] {best_path}")
        print("=" * 90)

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Restoring original ma_rgcl.yaml")
        MA_YAML_PATH.write_text(original_yaml_text, encoding="utf-8")
        raise

    except Exception:
        print("\n[ERROR] Restoring original ma_rgcl.yaml")
        MA_YAML_PATH.write_text(original_yaml_text, encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
