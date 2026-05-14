#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-Amazon_Musical_Instruments_14}"
GPU="${GPU:-0}"
EPOCH="${EPOCH:-200}"
SEED="${SEED:-42}"

cd "$(dirname "$0")/.."

run_iard() {
  python main.py \
    --model iard_rm \
    --dataset "$DATASET" \
    --mode train \
    --gpu "$GPU" \
    --seed "$SEED" \
    --epoch "$EPOCH" \
    "$@"
}

echo "[IARD-RM sweep] residual strength grid on $DATASET"
for eta in 0.3 0.5 1.0; do
  for gate_alpha in 1.0 2.0 5.0; do
    for residual_fusion_scale in 1.0 2.0; do
      echo "eta=$eta gate_alpha=$gate_alpha residual_fusion_scale=$residual_fusion_scale"
      run_iard \
        --loss_preset full_iard \
        --eta "$eta" \
        --gate_alpha "$gate_alpha" \
        --residual_fusion_scale "$residual_fusion_scale"
    done
  done
done

echo "[IARD-RM sweep] loss preset ablations on $DATASET"

echo "loss_preset=review_fusion"
run_iard \
  --loss_preset review_fusion \
  --lambda_align 0.0 \
  --lambda_sep 0.0 \
  --lambda_recon 0.0 \
  --lambda_gate 0.0 \
  --lambda_proto 0.0

echo "loss_preset=full_low_align"
run_iard \
  --loss_preset full_low_align \
  --lambda_align 0.05

echo "loss_preset=full_no_sep"
run_iard \
  --loss_preset full_no_sep \
  --lambda_sep 0.0

echo "loss_preset=full_iard"
run_iard --loss_preset full_iard
