#!/usr/bin/env bash

set -uo pipefail

SEEDS=(42 2024 2025 3407 9999)

MODEL="rgcl"
DATASET="Amazon_Office_Products_14"
MODE="train"
SPLIT_PROTOCOL="reviewgraph"
GPU=3

LOG_DIR="logs/${MODEL}_${DATASET}_reviewgraph"
mkdir -p "$LOG_DIR"

for SEED in "${SEEDS[@]}"; do
    echo "=========================================="
    echo "Running seed=${SEED}"
    echo "=========================================="

    python main.py \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --mode "$MODE" \
        --split_protocol "$SPLIT_PROTOCOL" \
        --gpu "$GPU" \
        --seed "$SEED" \
        2>&1 | tee "${LOG_DIR}/seed_${SEED}.log"

    STATUS=${PIPESTATUS[0]}

    if [ "$STATUS" -ne 0 ]; then
        echo "[FAILED] seed=${SEED}, exit code=${STATUS}" | tee -a "${LOG_DIR}/failed_seeds.txt"
    else
        echo "[DONE] seed=${SEED}" | tee -a "${LOG_DIR}/completed_seeds.txt"
    fi
done

for SEED in "${SEEDS[@]}"; do
    echo "=========================================="
    echo "Running seed=${SEED}"
    echo "=========================================="

    python main.py \
        --model "$MODEL" \
        --dataset "Amazon_Musical_Instruments_14" \
        --mode "$MODE" \
        --split_protocol "$SPLIT_PROTOCOL" \
        --gpu "$GPU" \
        --seed "$SEED" \
        2>&1 | tee "${LOG_DIR}/seed_${SEED}.log"

    STATUS=${PIPESTATUS[0]}

    if [ "$STATUS" -ne 0 ]; then
        echo "[FAILED] seed=${SEED}, exit code=${STATUS}" | tee -a "${LOG_DIR}/failed_seeds.txt"
    else
        echo "[DONE] seed=${SEED}" | tee -a "${LOG_DIR}/completed_seeds.txt"
    fi
done
echo "All seed runs finished."