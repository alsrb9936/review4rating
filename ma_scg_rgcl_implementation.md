# `scg_rgcl` / `ma_rgcl` 구현 설명

## 1. 전체 위치 요약

두 모델은 기존 `rgcl`을 기반으로 확장된 full-graph review-aware GNN 계열 모델입니다.

| 구분 | `scg_rgcl` | `ma_rgcl` |
| --- | --- | --- |
| 모델 | `model/scg_rgcl.py` | `model/ma_rgcl.py` |
| 데이터셋 | `data/scg_rgcl_dataset.py` | `data/ma_rgcl_dataset.py` |
| 트레이너 | `trainer/scg_rgcl_trainer.py` | `trainer/ma_rgcl_trainer.py` |
| 설정 | `config/yaml/scg_rgcl.yaml` | `config/yaml/ma_rgcl.yaml` |
| 등록 | `model/__init__.py`, `data/__init__.py`, `trainer/__init__.py` | 동일 |

공통적으로 `__len__() == 1`인 full-graph 방식입니다. 즉, 일반 mini-batch가 아니라 train graph 전체를 한 번에 encoder에 넣고, decoder에서 해당 split의 모든 `(user, item)` rating을 예측합니다.

---

## 2. 기본 RGCL 구조

둘 다 기존 `model/rgcl.py`의 구조를 계승합니다.

기본 RGCL 흐름은 다음과 같습니다.

```text
interaction dataframe
→ rating별 edge graph 구성
→ user/item full graph message passing
→ decoder에서 user-item pair 예측
→ rating loss + contrastive loss 학습
```

기본 batch 구조는 대략 이렇습니다.

```python
{
  "encoder_data": {
    "1": {...},
    "2": {...},
    ...
    "5": {...}
  },
  "norm_factors": ...,
  "decoder_user_ids": ...,
  "decoder_item_ids": ...,
  "decoder_review_feat": ...,
  "labels": ...,
  "ratings": ...
}
```

`encoder_data`는 rating 값별로 분리된 edge list입니다. 예를 들어 rating 5인 interaction만 `"5"` key 아래 들어갑니다.

---

# `scg_rgcl`

## 3. `scg_rgcl` 핵심 아이디어

`SCG_RGCL`은 **Sentiment-Calibrated Gated RGCL**에 가깝습니다.

기존 RGCL은 모든 edge의 message를 거의 동일한 구조로 전파합니다. 반면 `scg_rgcl`은 review sentiment와 rating이 얼마나 어긋나는지를 계산하고, 그 어긋남이 큰 edge의 graph message를 gate로 약화합니다.

즉 핵심은 다음입니다.

```text
rating과 review sentiment가 잘 맞음 → gate 큼 → message 강하게 전달
rating과 review sentiment가 어긋남 → gate 작음 → message 약하게 전달
```

---

## 4. `scg_rgcl` 데이터셋 구현

파일: `data/scg_rgcl_dataset.py`

`SCGRGCLDataset`은 `RecDataset`을 상속합니다.

기본적으로 decoder용 tensor를 먼저 만들고, train split이면 encoder graph를 구성합니다.

```python
self._build_decoder_tensors(self.df)
if self.split == "train":
    self._build_encoder_graph(self.df)
```

valid/test에서는 `_setup_evaluation()`에서 train graph를 encoder graph로 다시 만듭니다.

```python
def _setup_evaluation(self, train_df, valid_df, test_df):
    if self.split in {"valid", "test"}:
        self._build_encoder_graph(train_df)
```

이 구조 때문에 valid/test 예측 시에도 message passing graph는 train interaction만 사용합니다. 평가 target 자체가 encoder에 들어가지 않도록 하려는 구조입니다.

### sentiment feature 탐지

`SCGRGCLDataset`은 sentiment probability 또는 sentiment label column을 자동으로 찾습니다.

탐지 대상 예시는 다음과 같습니다.

- probability matrix:
  - `sent_prob_0` ~ `sent_prob_4`
  - `sentiment_prob_0` ~ `sentiment_prob_4`
  - `review_sent_prob_0` ~ `review_sent_prob_4`
  - `review_score`
  - `sentiment_score`
  - `sentiment_probs`
  - `sent_probs`
- label:
  - `sent_label`
  - `sentiment_label`
  - `review_sentiment`
  - `sentiment`
  - `sentiment_class`

### sentiment calibration

핵심 함수는 `_compute_sentiment_features()`입니다.

sentiment probability가 있으면 각 sentiment class가 실제 rating 평균과 어떻게 대응되는지 table을 만듭니다.

```text
sentiment class 0 → 평균 rating
sentiment class 1 → 평균 rating
...
sentiment class 4 → 평균 rating
```

그 후 review sentiment score를 rating scale로 변환합니다.

```python
sent_scores = probs @ table
```

그 다음 rating과 sentiment score를 `[0, 1]`로 normalize합니다.

```python
y_norm = (rating - min_rating) / (max_rating - min_rating)
s_norm = (sent_score - min_rating) / (max_rating - min_rating)
```

최종 misalignment score는 다음처럼 계산됩니다.

```python
rating_sent_distance = abs(y_norm - s_norm)
edge_misalign_score = confidence * rating_sent_distance
```

즉 sentiment classifier가 확신이 높고 rating과 sentiment가 많이 다르면 misalignment가 커집니다.

### encoder_data 추가 필드

기본 RGCL은 rating별 edge마다 다음 정도만 가집니다.

```python
"user_ids"
"item_ids"
"review_feat"
```

`scg_rgcl`은 여기에 다음을 추가합니다.

```python
"edge_misalign_score"
"sent_calibrated_score"
"sent_confidence"
"rating_sent_distance"
```

모델에서는 특히 `edge_misalign_score`를 gate 계산에 씁니다.

---

## 5. `scg_rgcl` 모델 구현

파일: `model/scg_rgcl.py`

주요 class는 다음입니다.

```text
SCGContrastLoss
SentimentPropagationGate
SCGGCMCGraphConv
SCGGCMCLayer
SCGMLPPredictorMI
SCG_RGCL
```

### SentimentPropagationGate

가장 중요한 부분은 `SentimentPropagationGate`입니다.

```python
raw_gate = sigmoid(beta0 - beta1 * edge_misalign_score)
gate = min_gate + (1 - min_gate) * raw_gate
```

의미는 간단합니다.

- `edge_misalign_score`가 작으면 `raw_gate`가 커짐
- `edge_misalign_score`가 크면 `raw_gate`가 작아짐
- 그래도 완전히 0이 되지 않도록 `min_gate`를 둠

설정 파일에서는 다음 값들이 들어갑니다.

```yaml
use_sentiment_gate: true
min_gate: 0.2
gate_beta0_init: 3.0
gate_beta1_init: 2.0
learnable_gate: true
```

`learnable_gate: true`이면 `beta0`, `beta1` 계열 파라미터가 학습됩니다.

### SCGGCMCGraphConv

기존 RGCL message는 대략 다음 형태입니다.

```python
msg = src_feat * pa + review_proj(review_feat) * ra
```

`scg_rgcl`에서는 여기에 edge gate가 곱해집니다.

```python
msg = msg * edge_gate.view(-1, 1)
```

즉 review sentiment와 rating이 어긋난 edge는 user/item embedding 업데이트에 덜 반영됩니다.

### SCGGCMCLayer

rating별 graph convolution을 수행합니다.

```python
for rating in self.rating_vals:
    edge_data = encoder_data.get(str(rating))
    edge_gate = sentiment_gate(edge_misalign_score)
    user_out += user_conv(...)
    item_out += item_conv(...)
```

각 rating별로 user 방향, item 방향 message passing을 따로 수행합니다.

### decoder

`SCGMLPPredictorMI`는 user embedding과 item embedding을 concat해서 rating logits를 만듭니다.

```python
interaction = MLP([user_emb, item_emb])
scores = predictor(interaction)
```

contrastive 학습 시에는 decoder review feature를 hidden dimension으로 projection해서 edge-level MI loss도 계산합니다.

---

## 6. `scg_rgcl` loss

`SCG_RGCL.cal_loss()`는 기본 RGCL과 loss 식이 같습니다.

```text
total_loss = rating_loss
           + nd_weight * nd_loss
           + ed_weight * ed_loss
```

다만 encoder에서 sentiment gate가 적용되므로 같은 loss라도 학습되는 representation이 달라집니다.

contrastive가 켜져 있으면 forward를 두 번 수행합니다.

```python
pred1, ed1, user1, item1 = forward_once(...)
pred2, ed2, user2, item2 = forward_once(...)
```

그 뒤 node contrastive loss를 계산합니다.

```python
nd_loss = contrast(user1, user2) + contrast(item1, item2)
```

classification mode에서는 label이 0~4 class이고, 예측 시에는 softmax 기대값으로 rating을 만듭니다.

```python
probs = softmax(logits)
rating = sum(probs * [1, 2, 3, 4, 5])
```

---

# `ma_rgcl`

## 7. `ma_rgcl` 핵심 아이디어

`MA_RGCL`은 **Misalignment-Aware RGCL**입니다.

`scg_rgcl`이 misalignment score로 단순히 edge message를 줄이는 방식이라면, `ma_rgcl`은 더 적극적으로 다음을 합니다.

1. user/item/review/rating/sentiment/misalign을 모두 보고 edge gate를 학습
2. message를 shared path와 residual path로 분리
3. misalignment가 크면 residual path를 더 활용
4. decoder에서 target review를 직접 쓰거나, history 기반으로 generated review를 생성
5. 추가 loss로 gate alignment, residual energy, review generation을 함께 학습

즉 `ma_rgcl`은 `scg_rgcl`보다 훨씬 강한 확장형입니다.

---

## 8. `ma_rgcl` 데이터셋 구현

파일: `data/ma_rgcl_dataset.py`

`MARGCLDataset`은 `SCGRGCLDataset`을 상속합니다.

```python
class MARGCLDataset(SCGRGCLDataset):
```

즉 sentiment detection, normalization, graph construction 일부 로직은 `scg_rgcl` 쪽을 재사용합니다.

### encoder graph

`_build_encoder_graph()`에서는 rating별 edge마다 다음을 저장합니다.

```python
"user_ids"
"item_ids"
"review_feat"
"normalized_rating"
"sentiment_score"
"edge_misalign_score"
```

`scg_rgcl`과 달리 `normalized_rating`, `sentiment_score`가 모델에 직접 들어갑니다. `MAEdgeGate`가 이 값들을 입력으로 쓰기 때문입니다.

### decoder history feature

`ma_rgcl`만의 중요한 추가 기능은 history review feature입니다.

```python
decoder_user_history_feat
decoder_item_history_feat
decoder_history_empty_mask
```

`_build_history_feature_tensors()`가 user별, item별 review embedding 평균을 만듭니다.

train에서는 자기 자신의 review를 제외합니다.

```python
exclude_self=True
```

valid/test에서는 train_df만 history로 사용합니다.

```python
self._build_history_feature_tensors(self.df, train_df, exclude_self=False)
```

이 구조는 평가 target의 review를 history에 섞지 않으려는 목적입니다.

### decoder sentiment score

`decoder_sentiment_score`도 추가됩니다.

다만 주석상 decoder rating은 노출하지 않고, review-derived sentiment만 사용합니다.

```python
# This is review-derived only; decoder ratings are never exposed.
```

---

## 9. `ma_rgcl` 모델 구현

파일: `model/ma_rgcl.py`

주요 class는 다음입니다.

```text
MAEdgeGate
MAGCMCGraphConv
MAGCMCLayer
MAMLPPredictorMI
MA_RGCL
```

### MAEdgeGate

`MAEdgeGate`는 `ma_rgcl`의 핵심입니다.

입력은 다음입니다.

```python
user_emb
item_emb
review_feat
normalized_rating
sentiment_score
misalign_score
```

review feature는 hidden dimension으로 projection됩니다.

```python
review_shared = review_proj(review_feat)
```

그 다음 모두 concat해서 gate를 계산합니다.

```python
gate_input = [
  user_emb,
  item_emb,
  review_shared,
  normalized_rating,
  sentiment_score,
  misalign_score
]

gate = sigmoid(MLP(gate_input))
```

즉 `scg_rgcl`의 gate는 misalignment scalar 중심이지만, `ma_rgcl`의 gate는 user/item/review/rating/sentiment까지 모두 보는 learnable gate입니다.

---

## 10. `ma_rgcl` graph convolution

`MAGCMCGraphConv`는 message를 두 갈래로 나눕니다.

### shared message

기본 collaborative signal + shared review signal입니다.

```python
shared_message = src_feat * pa + review_shared * shared_ra
```

### residual message

review residual projection입니다.

```python
residual_message = residual_review * residual_ra
```

### 최종 message

```python
gamma = epsilon + lambda_residual * (1 - gate)

message = gate * shared_message + gamma * residual_message
```

의미는 다음입니다.

- gate가 크다  
  → shared message를 강하게 사용
- gate가 작다  
  → residual message 비중이 상대적으로 증가
- `epsilon` 덕분에 residual path가 완전히 사라지지 않음
- `lambda_residual`이 residual 보정 강도를 조절

설정 파일에서는 다음처럼 들어갑니다.

```yaml
epsilon: 0.1
lambda_residual: 0.25
```

### align loss

gate가 misalignment와 반대로 움직이도록 alignment loss를 둡니다.

```python
align_loss = mse(gate, 1 - misalign_score)
```

즉 misalignment가 작으면 gate가 커지고, misalignment가 크면 gate가 작아지도록 유도합니다.

---

## 11. `ma_rgcl` decoder

`MAMLPPredictorMI`는 `scg_rgcl` decoder보다 복잡합니다.

설정:

```yaml
decoder_review_mode: generated
```

가능한 값은 두 개입니다.

```python
"target"
"generated"
```

### target mode

target review embedding을 그대로 decoder에 넣습니다.

```python
review_feat = batch["decoder_review_feat"]
```

### generated mode

현재 설정에서는 `generated`입니다.

이 경우 user/item embedding과 history review feature를 이용해 review representation을 생성합니다.

입력은 다음입니다.

```python
edge_user
edge_item
user_history_feat
item_history_feat
user_history_feat * item_history_feat
abs(user_history_feat - item_history_feat)
```

이를 `review_generator`에 넣어 decoder용 review feature를 만듭니다.

```python
review_feat = review_generator(generator_input)
```

학습 중에는 generated review와 실제 target review embedding 사이 MSE를 `generation_loss`로 둡니다.

```python
generation_loss = mse(generated_review, target_review_feat)
```

최종 rating prediction은 다음 입력을 사용합니다.

```python
[edge_user, edge_item, projected_review, gate]
```

즉 decoder 단계에서도 gate가 예측에 직접 들어갑니다.

---

## 12. `ma_rgcl` loss

`MA_RGCL.cal_loss()`의 total loss는 다음입니다.

```text
total_loss =
    rating_loss
  + nd_weight * nd_loss
  + ed_weight * ed_loss
  + align_weight * align_loss
  + residual_weight * residual_loss
  + review_generation_weight * generation_loss
```

설정 파일 기준:

```yaml
nd_weight: 0.1
ed_weight: 1.0
align_weight: 0.05
residual_weight: 0.002
review_generation_weight: 0.01
```

추가 loss 의미는 다음입니다.

| Loss | 의미 |
| --- | --- |
| `rating_loss` | rating classification/regression loss |
| `nd_loss` | user/item node-level contrastive loss |
| `ed_loss` | interaction-review edge-level contrastive loss |
| `align_loss` | gate가 `1 - misalign`에 맞도록 유도 |
| `residual_loss` | residual message energy regularization |
| `generation_loss` | generated review embedding과 target review embedding 정렬 |

---

## 13. 트레이너 구현 차이

### `scg_rgcl`

파일: `trainer/scg_rgcl_trainer.py`

`SCGRGCLTrainer`는 full-graph batch 하나를 받아 학습합니다.

```python
loss, loss_dict = self.model.cal_loss(batch)
loss.backward()
clip_grad_norm_(...)
optimizer.step()
```

평가에서는 다음을 호출합니다.

```python
predictions = self.model.predict_ratings(batch)
```

그리고 모델의 `current_gate_stats`가 있으면 metric에 추가합니다.

test phase에서는 다음 통계를 출력합니다.

```text
raw prediction min/max/mean
clipped prediction min/max/mean
mean_edge_misalign
```

### `ma_rgcl`

파일: `trainer/ma_rgcl_trainer.py`

`MARGCLTrainer`는 `SCGRGCLTrainer`를 상속합니다.

```python
class MARGCLTrainer(SCGRGCLTrainer):
```

평가 로직은 거의 같지만 test 출력에 `mean_gamma`가 추가됩니다.

```text
mean_gate
std_gate
min_gate
max_gate
mean_gamma
mean_misalign
```

---

## 14. 설정 차이

### `scg_rgcl.yaml`

```yaml
embedding_size: 32
num_layers: 2
dropout: 0.7
use_contrastive: true
nd_weight: 0.1
ed_weight: 0.7
classification: true

use_sentiment_gate: true
min_gate: 0.2
gate_beta0_init: 3.0
gate_beta1_init: 2.0
learnable_gate: true

use_sentiment_confidence: true
calibrate_sentiment_to_rating: true
use_raw_sentiment_distance: false
```

`scg_rgcl`은 sentiment를 rating scale로 calibration해서 gate를 계산하는 쪽에 초점이 있습니다.

### `ma_rgcl.yaml`

```yaml
embedding_size: 64
num_layers: 1
dropout: 0.3
use_contrastive: true
classification: true

nd_weight: 0.1
ed_weight: 1.0
align_weight: 0.05
residual_weight: 0.002

gate_hidden_dim: 32
epsilon: 0.1
lambda_residual: 0.25

decoder_review_mode: generated
review_generation_weight: 0.01

calibrate_sentiment_to_rating: false
use_raw_sentiment_distance: true
```

`ma_rgcl`은 calibration gate보다, misalignment-aware gate와 residual/generation 구조에 초점이 있습니다.

---

## 15. 한눈에 보는 차이

| 항목 | `scg_rgcl` | `ma_rgcl` |
| --- | --- | --- |
| 기본 방향 | sentiment-calibrated gate | misalignment-aware shared/residual |
| Dataset 상속 | `RecDataset` | `SCGRGCLDataset` |
| Gate 입력 | `edge_misalign_score` | user, item, review, rating, sentiment, misalign |
| Message 처리 | 기존 RGCL message × gate | shared message + residual message |
| Residual path | 없음 | 있음 |
| History feature | 없음 | 있음 |
| Review generation | 없음 | 있음 |
| 추가 loss | 없음 | align/residual/generation loss |
| Gate 통계 | mean/std/min/max/misalign | mean/std/min/max/gamma/misalign |
| Decoder 입력 | user + item | user + item + projected review + gate |

---

## 16. 최종 요약

`scg_rgcl`은 기존 RGCL에 sentiment 기반 gate를 추가한 모델입니다. review sentiment와 rating이 잘 맞는 interaction은 강하게 전파하고, 어긋나는 interaction은 graph message를 약하게 만들어 noisy edge의 영향을 줄이려는 구현입니다.

`ma_rgcl`은 그보다 더 확장된 버전입니다. misalignment를 단순히 message 감쇠에만 쓰지 않고, user/item/review/rating/sentiment를 모두 보는 learnable gate를 만들며, message를 shared path와 residual path로 나눕니다. 또한 decoder에서는 history 기반 review representation을 생성할 수 있고, rating loss 외에도 alignment, residual, generation loss를 함께 최적화합니다.
