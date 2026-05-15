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
IARD_YAML_PATH = REPO_ROOT / "config" / "yaml" / "iard_rm.yaml"


SEARCH_SPACE = {
    "training.lr": [0.0001, 0.0002, 0.0005, 0.001],
    "training.weight_decay": [0.0, 1e-6, 1e-5, 1e-4],
    "training.grad_clip": [0.5, 1.0, 2.0, 5.0],
    "model.d_id": [32, 64, 128],
    "model.d_model": [64, 128, 256],
    "model.num_layers": [1, 2],
    "model.num_intents": [3, 5, 8],
    "model.eta": [0.3, 0.5, 0.6, 0.8],
    "model.dropout": [0.1, 0.2, 0.3, 0.5],
    "model.tau_p": [0.1, 0.2, 0.3],
    "model.gate_alpha": [1.0, 5.0, 10.0],
    "model.history_top_k": [10, 30, 50],
    "loss.lambda_rating": [0.5, 1.0, 2.0],
    "loss.lambda_align": [0.05, 0.1, 0.2],
    "loss.lambda_sep": [0.01, 0.05, 0.1],
    "loss.lambda_recon": [0.05, 0.1, 0.2],
}


def get_nested(config: dict, dotted_key: str):
    cur = config
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        cur = cur[key]
    return cur[keys[-1]]


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


def patch_iard_yaml(base_config: dict, params: dict):
    config = deepcopy(base_config)
    for key, value in params.items():
        set_nested(config, key, value)
    save_yaml(IARD_YAML_PATH, config)


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
        "--model", "iard_rm",
        "--dataset", args.dataset,
        "--mode", "train",
        "--split_protocol", args.split_protocol,
        "--gpu", str(args.gpu),
        "--seed", str(args.seed),
    ]

    if args.epoch is not None:
        cmd += ["--epoch", str(args.epoch)]

    if args.loss_preset is not None:
        cmd += ["--loss_preset", args.loss_preset]

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
    parser.add_argument("--loss_preset", type=str, default=None, help="Loss preset for IARD-RM (e.g., rating_only, full_iard)")
    args = parser.parse_args()

    if not IARD_YAML_PATH.exists():
        raise FileNotFoundError(f"Cannot find {IARD_YAML_PATH}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "results" / "greedy_search_iard_rm" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "search_log.csv"
    best_path = out_dir / "best_config.json"

    original_yaml_text = IARD_YAML_PATH.read_text(encoding="utf-8")
    base_config = load_yaml(IARD_YAML_PATH)

    current_params = {}
    for key in SEARCH_SPACE:
        try:
            current_params[key] = get_nested(base_config, key)
        except (KeyError, TypeError):
            pass

    best_params = deepcopy(current_params)
    best_score = None
    best_result = None

    try:
        patch_iard_yaml(base_config, best_params)
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
                if param_name not in best_params:
                    continue
                    
                local_best_value = best_params[param_name]
                local_best_score = best_score
                local_best_result = best_result

                for value in candidates:
                    if value == best_params[param_name]:
                        continue

                    trial_params = deepcopy(best_params)
                    trial_params[param_name] = value

                    patch_iard_yaml(base_config, trial_params)

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

        patch_iard_yaml(base_config, best_params)

        print("\n" + "=" * 90)
        print("[DONE] Greedy search finished.")
        print(f"[BEST {args.metric}] {best_score}")
        print(f"[BEST PARAMS] {json.dumps(best_params, indent=2, ensure_ascii=False)}")
        print(f"[LOG] {csv_path}")
        print(f"[BEST JSON] {best_path}")
        print("=" * 90)

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Restoring original iard_rm.yaml")
        IARD_YAML_PATH.write_text(original_yaml_text, encoding="utf-8")
        raise

    except Exception:
        print("\n[ERROR] Restoring original iard_rm.yaml")
        IARD_YAML_PATH.write_text(original_yaml_text, encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
