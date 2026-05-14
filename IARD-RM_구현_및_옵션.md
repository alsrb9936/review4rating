# IARD-RM 구현 및 Ablation 옵션 정리

이 문서는 현재 코드베이스에 구현된 `iard_rm` 모델의 구조, 데이터 입력 방식, 손실 함수, ablation 옵션, 실행 예시를 정리한다. 기준 코드는 `model/iard_rm.py`, `data/iard_rm_dataset.py`, `trainer/iard_rm_trainer.py`, `main.py`, `config/yaml/iard_rm.yaml`이다.

## 1. 모델 개요

IARD-RM은 rating graph 기반 표현과 review 기반 표현을 함께 사용해 평점을 예측하는 review-aware rating prediction 모델이다. 핵심 아이디어는 다음과 같다.

- user-item rating graph에서 rating-side 표현 `zY`를 만든다.
- target review를 직접 쓰는 대신, 기본 설정에서는 train history review embedding으로 review context를 만든다.
- review context에서 multi-intent 표현 `zX`를 만들고, 이를 shared component `zS`와 residual component `zR`로 분해한다.
- `zY`와 `zS`의 정렬 정도를 기반으로 inconsistency gate를 계산한다.
- 최종 평점은 rating head, shared review head, residual review head를 gate로 조합하고 user/item/global bias를 더해 예측한다.

## 2. 데이터 및 Review Context 구성

구현 위치: `data/iard_rm_dataset.py`

### 2.1 기본 입력 필드

각 sample은 다음 필드를 반환한다.

| 필드 | 의미 |
|---|---|
| `user_ids` | user index |
| `item_ids` | item index |
| `ratings` | target rating |
| `review_emb` | mean history 방식에서 사용하는 fused review context |
| `user_history_emb` | user history review embedding sequence, shape `[K, D]` |
| `user_history_mask` | user history valid mask, shape `[K]` |
| `item_history_emb` | item history review embedding sequence, shape `[K, D]` |
| `item_history_mask` | item history valid mask, shape `[K]` |
| `review_source_used` | history 사용 여부 표시. history mode에서는 1 |
| `empty_history_mask` | user/item history 중 비어 있는 context가 있는지 표시 |

### 2.2 Leakage 방지 정책

현재 기본 설정은 `review_context_mode: "history"`이다. 이 모드에서는 valid/test의 target review embedding을 입력으로 사용하지 않는다.

- train split:
  - 기본값 `retain_rui: false`이면 target interaction 자신의 review를 user/item history에서 제외한다.
  - `retain_rui: true`이면 reproduction-style 설정으로 train target review를 history에 유지한다.
- valid/test split:
  - target review embedding을 로드하지 않는다.
  - `_setup_evaluation()`에서 train interactions만 history source로 사용한다.
  - valid/test에서 `review_context_mode="target"`은 금지되어 있다.

로그 예시는 다음과 같다.

```text
IARD-RM valid history context: source=train_only, encoder=attention, top_k=10, target_review_used=False, empty_ratio=0.0000
IARD-RM test leakage check: history_source_ratio=1.0000, target_review_used=False, empty_history_ratio=0.0000
```

## 3. 모델 아키텍처

구현 위치: `model/iard_rm.py`

### 3.1 RatingGraphEncoder

`RatingGraphEncoder`는 user/item embedding을 bipartite rating graph 위에서 전파한 뒤 target `(user_id, item_id)`에 대한 rating-side 표현 `zY`를 만든다.

입력:

- `user_ids`
- `item_ids`
- `edge_index`
- optional `edge_weight`

출력:

- `zY`, shape `[B, d_model]`

관련 옵션:

| 옵션 | 기본값 | 설명 |
|---|---:|---|
| `d_id` | 64 | user/item ID embedding 차원 |
| `d_model` | 128 | 내부 표현 차원 |
| `num_layers` | 1 | graph propagation layer 수 |

### 3.2 Review Context Encoder

Review context는 두 방식 중 하나로 구성된다.

#### `history_encoder: "mean"`

기존 방식이다. user history 평균과 item history 평균을 만든 뒤 다음과 같이 결합한다.

```text
[user_hist, item_hist, user_hist * item_hist, abs(user_hist - item_hist)]
```

이 fused vector가 `ReviewProjectionEncoder`로 들어가 `hX`가 된다.

#### `history_encoder: "attention"`

새로 추가된 interaction-conditioned history attention 방식이다.

- query: rating graph 표현 `zY`
- key: user/item history review embedding을 `d_model`로 projection한 값
- value: raw user/item history review embedding
- mask: padding history는 attention에서 제외
- all-padding history는 zero context로 처리해 NaN을 방지

user context와 item context를 각각 attention으로 만든 뒤 mean 방식과 동일하게 결합한다.

```text
[user_ctx, item_ctx, user_ctx * item_ctx, abs(user_ctx - item_ctx)]
```

관련 옵션:

| 옵션 | 기본값 | 설명 |
|---|---:|---|
| `history_encoder` | `mean` | `mean` 또는 `attention` |
| `history_top_k` | 10 | attention/sequence history에서 사용할 최대 history 개수 |
| `history_aggregation` | `mean` | 현재는 mean aggregation만 지원 |
| `history_temporal` | false | timestamp 기반 과거 history만 사용하는 옵션 |
| `retain_rui` | false | train에서 target interaction review를 history에 유지할지 여부 |

### 3.3 PrototypeIntentExtractor

`PrototypeIntentExtractor`는 review-side representation `hX`를 prototype 기반 multi-intent space로 변환한다.

출력:

- `zX`: review intent representation
- `intent_weights`: prototype attention weight

관련 옵션:

| 옵션 | 기본값 | 설명 |
|---|---:|---|
| `num_intents` | 5 | prototype 개수 |
| `tau_p` | 0.2 | prototype softmax temperature |

### 3.4 SharedResidualDisentangler

`SharedResidualDisentangler`는 `zX`를 shared representation `zS`와 residual representation `zR`로 분해한다.

관련 옵션:

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `disentangler_mode` | `conditioned` | disentangler 입력 방식 |

두 모드가 있다.

- `conditioned`: 현재 기본 방식. `[zX, zY, zX * zY, abs(zX - zY)]`를 입력으로 사용한다.
- `independent`: 기존 독립 방식. `zX`만 사용한다.

### 3.5 InconsistencyGate

`InconsistencyGate`는 `zY`와 `zS`의 cosine alignment를 계산하고, raw gate와 alignment gate를 섞어 최종 gate를 만든다.

출력:

- `alignment`
- `p_inc = 1 - gate`
- `gate`

관련 옵션:

| 옵션 | 기본값 | 설명 |
|---|---:|---|
| `gate_alpha` | 5.0 | alignment gate의 sharpness |
| `fixed_gate_value` | null | 지정 시 gate를 고정값으로 사용 |

### 3.6 RatingPredictor 및 Bias Term

`RatingPredictor`는 세 개의 head를 사용한다.

- `fY(zY)`: rating graph 기반 예측
- `fS(zS)`: shared review signal 예측
- `fR(zR)`: residual review signal 예측

기본 예측식은 다음과 같다.

```text
pred = yY + shared_fusion_scale * gate * yS
           + residual_fusion_scale * (1 - gate) * eta * yR
```

이후 bias term을 추가한다.

```text
pred = pred + user_bias[user] + item_bias[item] + global_bias
```

Bias 초기화는 다음과 같다.

- `user_bias`: 0 초기화
- `item_bias`: 0 초기화
- `global_bias`: `train_dataset.ratings.float().mean()`으로 초기화, 없으면 0.0

관련 옵션:

| 옵션 | 기본값 | 설명 |
|---|---:|---|
| `eta` | preset 적용 후 보통 0.3 | residual prediction scale |
| `shared_fusion_scale` | 1.0 | shared review signal 반영 강도 |
| `residual_fusion_scale` | 1.0 | residual signal 반영 강도 |

## 4. 손실 함수

구현 위치: `IARDLossComputer`

최종 loss는 다음 항들의 weighted sum이다.

```text
loss = lambda_rating * rating_loss
     + lambda_align  * align_loss
     + lambda_sep    * sep_loss
     + lambda_recon  * recon_loss
     + lambda_gate   * gate_loss
     + lambda_proto  * proto_loss
```

각 항의 의미는 다음과 같다.

| Loss | 구현 의미 |
|---|---|
| `rating_loss` | `pred`와 rating 간 MSE |
| `align_loss` | `zY`와 `zS`의 batch contrastive alignment loss |
| `sep_loss` | `zR`이 `zY`, `zS`와 과도하게 겹치지 않도록 하는 cosine penalty |
| `recon_loss` | `[zS, zR]`로 `zX`를 재구성하는 MSE |
| `gate_loss` | gate entropy 기반 regularization |
| `proto_loss` | prototype 간 orthogonality regularization |

관련 옵션:

| 옵션 | 기본값 |
|---|---:|
| `lambda_rating` | 1.0 |
| `lambda_align` | 0.1 |
| `lambda_sep` | 0.05 |
| `lambda_recon` | 0.1 |
| `lambda_gate` | 0.001 |
| `lambda_proto` | 0.01 |
| `tau_c` | 0.2 |
| `eps` | 1e-8 |
| `detach_gate_for_align` | true |
| `detach_zx_for_recon` | false |

## 5. Loss Preset 및 Ablation 옵션

구현 위치: `main.py`의 `apply_iard_loss_preset()`

Preset은 기본값을 먼저 적용한 뒤, CLI에서 명시적으로 지정한 값은 다시 복원한다. 따라서 예를 들어 `--loss_preset full_iard --lambda_align 0.05`처럼 실행하면 preset 적용 후 CLI override가 유지된다.

### 5.1 공통 full defaults

대부분 preset은 다음 full default에서 시작한다.

| 옵션 | 값 |
|---|---:|
| `eta` | 0.3 |
| `shared_fusion_scale` | 1.0 |
| `residual_fusion_scale` | 1.0 |
| `fixed_gate_value` | null |
| `lambda_align` | 0.1 |
| `lambda_sep` | 0.05 |
| `lambda_recon` | 0.1 |
| `lambda_gate` | 0.001 |
| `lambda_proto` | 0.01 |

### 5.2 Preset 목록

| Preset | 의미 | 주요 변경 |
|---|---|---|
| `rating_only` | rating graph + bias 중심 baseline | `shared_fusion_scale=0`, `residual_fusion_scale=0`, auxiliary loss 전부 0 |
| `review_fusion` | review signal은 fusion에 쓰지만 auxiliary loss는 끔 | fusion scale 유지, `lambda_align/sep/recon/gate/proto=0` |
| `full_iard` | 전체 IARD-RM objective | full defaults 그대로 사용 |
| `full_no_sep` | separation loss 제거 ablation | `lambda_sep=0` |
| `full_low_align` | alignment 약화 ablation | `lambda_align=0.05` |
| `full_no_residual_pred` | residual prediction contribution 제거 | `residual_fusion_scale=0` |
| `full_fixed_gate` | gate를 학습하지 않고 0.5로 고정 | `fixed_gate_value=0.5` |

> README에는 `full_eta_0_3`도 사용 예시로 언급되어 있으나, 현재 `apply_iard_loss_preset()`의 지원 preset 목록에는 없다. 현재 코드 기준으로는 위 7개 preset이 유효하다.

## 6. 기타 Ablation 축

Loss preset 외에도 다음 옵션으로 구조적 ablation을 수행할 수 있다.

### 6.1 Review history encoder

```bash
--history_encoder mean
--history_encoder attention
```

- `mean`: user/item history 평균 기반 fused context
- `attention`: `(user_id, item_id)` pair의 rating graph 표현 `zY`로 history를 attention하는 interaction-conditioned context

### 6.2 History top-k

```bash
--history_top_k 5
--history_top_k 10
--history_top_k 20
```

`history_encoder=attention`일 때 주요 ablation 축이다. 데이터셋은 각 user/item history를 `[K, D]`로 padding하고 mask를 함께 제공한다.

### 6.3 Target review 유지 여부

```bash
--retain_rui false
--retain_rui true
```

- `false`: fair evaluation 기본값. train에서도 target interaction review를 history에서 제외한다.
- `true`: reproduction-style 설정. train history에 target interaction review를 유지한다.
- valid/test에서는 항상 train history만 사용하므로 target review leakage는 차단된다.

### 6.4 Disentangler mode

```bash
--disentangler_mode conditioned
--disentangler_mode independent
```

- `conditioned`: `zY`에 조건화된 disentanglement
- `independent`: `zX`만으로 disentanglement

### 6.5 Fusion scale / gate 관련 옵션

```bash
--eta 0.3
--shared_fusion_scale 1.0
--residual_fusion_scale 1.0
--gate_alpha 5.0
```

- `eta`: residual head의 contribution 크기
- `shared_fusion_scale`: shared review prediction 반영 정도
- `residual_fusion_scale`: residual prediction 반영 정도
- `gate_alpha`: alignment 기반 gate의 민감도

## 7. 학습 및 평가 로그

구현 위치: `trainer/iard_rm_trainer.py`

학습/평가 중 다음 metric이 기록된다.

| Metric | 의미 |
|---|---|
| `rmse`, `mse`, `mae` | rating prediction metric |
| `mean_gate`, `std_gate` | gate 분포 |
| `mean_p_inc`, `std_p_inc` | inconsistency probability 분포 |
| `mean_alignment`, `std_alignment` | `zY`와 `zS` cosine alignment 분포 |
| `mean_intent_entropy` | prototype intent weight entropy |
| `mean_empty_history_ratio` | empty history 비율 |
| `history_source_ratio` | valid/test에서 history source가 사용된 비율 |

Test 단계에서는 `test_sample_analysis.csv`도 저장된다. 이 파일에는 sample별 prediction, error, gate, p_inc, alignment, intent entropy, `yY/yS/yR`, `review_source_used` 등이 포함된다.

또한 alignment quantile 기준으로 다음 group metric도 생성된다.

- `low_alignment_rmse`, `low_alignment_mae`, `low_alignment_count`
- `mid_alignment_rmse`, `mid_alignment_mae`, `mid_alignment_count`
- `high_alignment_rmse`, `high_alignment_mae`, `high_alignment_count`

## 8. 기본 설정 요약

`config/yaml/iard_rm.yaml` 기준 주요 기본값은 다음과 같다.

```yaml
model:
  d_id: 64
  d_text: 384
  review_context_mode: "history"
  history_aggregation: "mean"
  history_encoder: "mean"
  history_top_k: 10
  history_temporal: false
  retain_rui: false
  d_model: 128
  num_layers: 1
  num_intents: 5
  eta: 0.6
  dropout: 0.2
  tau_p: 0.2
  gate_alpha: 5.0
  shared_fusion_scale: 1.0
  residual_fusion_scale: 1.0
  fixed_gate_value: null
  disentangler_mode: "conditioned"

loss:
  loss_preset: full_iard
  lambda_rating: 1.0
  lambda_align: 0.1
  lambda_sep: 0.05
  lambda_recon: 0.1
  lambda_gate: 0.001
  lambda_proto: 0.01
```

주의할 점은 `apply_iard_loss_preset()`이 실행되면 일부 값은 preset default로 덮인다. 예를 들어 `full_iard` 계열 preset에서는 `eta`가 0.3으로 적용된다. CLI에서 명시한 값은 preset 적용 후 복원된다.

## 9. 실행 예시

### 9.1 Rating-only baseline

```bash
python main.py \
  --model iard_rm \
  --dataset Amazon_Musical_Instruments_14 \
  --mode train \
  --loss_preset rating_only
```

### 9.2 Full IARD-RM

```bash
python main.py \
  --model iard_rm \
  --dataset Amazon_Musical_Instruments_14 \
  --mode train \
  --loss_preset full_iard
```

### 9.3 Attention history encoder

```bash
python main.py \
  --model iard_rm \
  --dataset Amazon_Musical_Instruments_14 \
  --mode train \
  --loss_preset full_iard \
  --history_encoder attention \
  --history_top_k 10
```

### 9.4 Disentangler 비교

```bash
python main.py --model iard_rm --dataset Amazon_Musical_Instruments_14 --mode train \
  --loss_preset full_iard --disentangler_mode conditioned

python main.py --model iard_rm --dataset Amazon_Musical_Instruments_14 --mode train \
  --loss_preset full_iard --disentangler_mode independent
```

### 9.5 History top-k ablation

```bash
python main.py --model iard_rm --dataset Amazon_Musical_Instruments_14 --mode train \
  --loss_preset full_iard --history_encoder attention --history_top_k 5

python main.py --model iard_rm --dataset Amazon_Musical_Instruments_14 --mode train \
  --loss_preset full_iard --history_encoder attention --history_top_k 10

python main.py --model iard_rm --dataset Amazon_Musical_Instruments_14 --mode train \
  --loss_preset full_iard --history_encoder attention --history_top_k 20
```

### 9.6 Sanity check

```bash
python -c "from model.iard_rm import run_iard_rm_sanity_check; print(run_iard_rm_sanity_check())"
python -c "from trainer.iard_rm_trainer import run_iard_trainer_sanity_check; print(run_iard_trainer_sanity_check())"
```

## 10. 추천 Ablation Matrix

우선순위가 높은 비교는 다음과 같다.

| 목적 | 비교 |
|---|---|
| bias 추가 후 baseline 확인 | `rating_only` |
| review fusion 자체 효과 | `rating_only` vs `review_fusion` |
| auxiliary loss 효과 | `review_fusion` vs `full_iard` |
| alignment 강도 | `full_iard` vs `full_low_align` |
| residual branch 효과 | `full_iard` vs `full_no_residual_pred` |
| disentangler 조건화 효과 | `conditioned` vs `independent` |
| interaction-conditioned history 효과 | `history_encoder=mean` vs `attention` |
| history 길이 민감도 | `history_top_k=5/10/20` |
| train target review 포함 영향 | `retain_rui=false` vs `true` |

## 11. 해석 시 주의사항

1. `retain_rui=true`는 train context에 target review를 포함하므로 fair setting과 구분해서 기록해야 한다.
2. valid/test에서는 항상 train history만 사용하도록 구현되어 있으며, 로그의 `target_review_used=False`를 확인해야 한다.
3. `history_encoder=attention`은 속도가 mean보다 느릴 수 있다. 실제 smoke run에서는 attention path가 정상 동작하고 leakage check도 통과했다.
4. `loss_preset`은 여러 coefficient를 한 번에 바꾸므로, 개별 `lambda_*` ablation을 할 때는 CLI override가 적용되었는지 저장된 `config.yaml`을 확인하는 것이 좋다.
5. 비교 시에는 `test_results.json`, `training_log.json`, `test_sample_analysis.csv`를 함께 확인하는 것이 좋다. 특히 RMSE/MAE 외에 `mean_gate`, `mean_p_inc`, `mean_alignment`를 같이 봐야 IARD-RM의 내부 동작을 해석할 수 있다.
