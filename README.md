# Research
---
- Review Based Recommendation for Rating Prediction Task

### Models
---
- NeuMF
- RGCL
- DeepCoNN
- IARD-RM

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
