# Research
---
- Review Based Recommendation for Rating Prediction Task

### Models
---
- NeuMF
- RGCL
- DeepCoNN
- SSG
- IARD-RM

## SSG Usage

SSG combines set, temporal sequence, and full-graph review views for rating prediction. The dataset must include `user_id`, `item_id`, `rating`, `review`, and `timestamp` columns after the `.inter` and `.review` files are merged.

### 1. Full SSG Training

```bash
python main.py --model ssg --dataset Amazon_Musical_Instruments_14 --mode train --ssg_preset full
```

### 2. Small Overfit / Smoke Test

```bash
python main.py --model ssg --dataset Amazon_Musical_Instruments_14 --mode train --overfit_n 100 --epoch 20 --batch 8 --ssg_preset full
```

`--overfit_n` intentionally reuses the same interactions for train/valid/test, so those metrics are debug-only.

### 3. Ablation Presets

```bash
python main.py --model ssg --dataset Amazon_Musical_Instruments_14 --mode train --ssg_preset set_only
python main.py --model ssg --dataset Amazon_Musical_Instruments_14 --mode train --ssg_preset set_sequence
python main.py --model ssg --dataset Amazon_Musical_Instruments_14 --mode train --ssg_preset set_graph
python main.py --model ssg --dataset Amazon_Musical_Instruments_14 --mode train --ssg_preset full
python main.py --model ssg --dataset Amazon_Musical_Instruments_14 --mode train --ssg_preset no_decov
```

### 4. Review Feature Modes

Token mode uses the configured GloVe file and is the default:

```bash
python main.py --model ssg --dataset Amazon_Musical_Instruments_14 --mode train --review_input_mode token
```

Embedding mode uses cached or generated review embeddings, including BERT-Whitening:

```bash
python main.py --model ssg --dataset Amazon_Musical_Instruments_14 --mode train --review_feature_backend bert_whitening --review_input_mode embedding
```

Validation and test SSG datasets reuse the train vocabulary, embedding matrix, historical interactions, and full train graph. Graph presets use a full train interaction graph, not a target-local subgraph.

## IARD-RM Usage

### 1. Sanity Check

```bash
python -c "from model.iard_rm import run_iard_rm_sanity_check; print(run_iard_rm_sanity_check())"
python -c "from trainer.iard_rm_trainer import run_iard_trainer_sanity_check; print(run_iard_trainer_sanity_check())"
```

### 2. Small Overfit

```bash
python main.py --model iard_rm --dataset Amazon_Musical_Instruments_14 --mode train --overfit_n 100 --epoch 200 --batch 32
```

### 3. Rating-Only Training

```bash
python main.py --model iard_rm --dataset Amazon_Musical_Instruments_14 --mode train --loss_preset rating_only
```

### 4. Full IARD Training

```bash
python main.py --model iard_rm --dataset Amazon_Musical_Instruments_14 --mode train --loss_preset full_iard
```

### 5. Ablation Training

```bash
python main.py --model iard_rm --dataset Amazon_Musical_Instruments_14 --mode train --loss_preset review_fusion
python main.py --model iard_rm --dataset Amazon_Musical_Instruments_14 --mode train --loss_preset align_only
python main.py --model iard_rm --dataset Amazon_Musical_Instruments_14 --mode train --loss_preset full_iard
python main.py --model iard_rm --dataset Amazon_Musical_Instruments_14 --mode train --loss_preset full_no_sep
python main.py --model iard_rm --dataset Amazon_Musical_Instruments_14 --mode train --loss_preset full_no_residual_pred
python main.py --model iard_rm --dataset Amazon_Musical_Instruments_14 --mode train --loss_preset full_fixed_gate
python main.py --model iard_rm --dataset Amazon_Musical_Instruments_14 --mode train --loss_preset full_eta_0_3
python main.py --model iard_rm --dataset Amazon_Musical_Instruments_14 --mode train --loss_preset full_low_align
```
