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

## Data Split Protocol

The `--split_protocol` flag controls how data is split into train/valid/test:

### Default (80/10/10 Random)

```bash
python main.py --model neumf --dataset Amazon_Musical_Instruments_14 --mode train
```

Standard sklearn `train_test_split` with shuffle: 80% train, 10% valid, 10% test.

### ReviewGraph Protocol

```bash
python main.py --model ssg --dataset Amazon_Musical_Instruments_14 --mode train --split_protocol reviewgraph
```

Shuffle → first 10% valid, next 10% test, remaining 80% train. Rows in valid/test with user_id or item_id not seen in train are migrated to train (iteratively until stable). Split sizes and rating distributions are logged before/after migration.

Use this protocol for SSG/RGCL/SGDN baseline reproduction.

## Baseline Configs

### SSG (Token + Word2Vec/CNN)

SSG uses token/word2vec mode by default (GloVe embeddings + CNN). Config defaults:
- `review_input_mode: "token"` (GloVe-based)
- `ssg_preset: "full"`, `use_set_view: true`, `use_sequence_view: true`, `use_graph_view: true`
- `train_clip: false`, `test_clip: true` (clipping only during evaluation)
- `batch: 100`, `epoch: 6`, `lr: 0.002`, `word_dim: 300`, `latent_dim: 8`

```bash
python main.py --model ssg --dataset Amazon_Musical_Instruments_14 --mode train --split_protocol reviewgraph
```

### SGDN (Graph Disentangled)

Config defaults:
- `num_factors: 2`, `dropout: 0.8`, `lr: 0.01`, `epoch: 2000`
- `cl_weight: 0.005`, `num_pos: 10`, `num_neg: 2048`
- `classification: false`, `disentangle_weight: 0.0`
- `review_dim: 64`, `temperature: 0.2`, `edge_temperature: 0.5`, `debug_shapes: false`
- `eval_clip: false` (no clipping for baseline comparison)
- Dependencies: `dgl`; optional `faiss-cpu` for prototype KMeans, otherwise sklearn `KMeans` is used.

```bash
python main.py --model sgdn --dataset Amazon_Musical_Instruments_14 --mode train --split_protocol reviewgraph --eval_clip false
python main.py --model sgdn --dataset Amazon_Musical_Instruments_14 --mode train --overfit_n 512 --epoch 20 --batch 1
```

### RGCL (Graph Contrastive Learning)

Config defaults:
- `classification: true` (expected-rating from softmax probabilities)
- `dropout: 0.7`, `lr: 0.01`, `epoch: 400`
- `nd_weight: 0.3`, `ed_weight: 1.0`, `review_dim: 64`
- Classification mode produces expected ratings in [1,5] range; eval clipping not needed.

```bash
python main.py --model rgcl --dataset Amazon_Musical_Instruments_14 --mode train --split_protocol reviewgraph
```

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

SSG uses token/word2vec mode by default (configured via GloVe). No additional flags needed:

```bash
python main.py --model ssg --dataset Amazon_Musical_Instruments_14 --mode train
```

To explicitly set token mode:

```bash
python main.py --model ssg --dataset Amazon_Musical_Instruments_14 --mode train --review_input_mode token
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
