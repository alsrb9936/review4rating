# IARD-RM 방법론 상세 정리

이 문서는 현재 코드베이스의 `iard_rm` 구현을 기준으로 IARD-RM의 방법론을 단계별로 설명한다. 기준 파일은 `model/iard_rm.py`, `data/iard_rm_dataset.py`, `trainer/iard_rm_trainer.py`, `config/yaml/iard_rm.yaml`, `main.py`이다.

## 1. 문제 설정과 전체 아이디어

IARD-RM은 사용자 `u`와 아이템 `i`가 주어졌을 때 평점 `r_ui`를 예측하는 모델이다. 단순히 user/item ID만 사용하는 협업 필터링 모델과 달리, 이 모델은 두 종류의 정보를 함께 사용한다.

1. **Rating-side 정보**: user-item rating graph에서 얻는 협업적 상호작용 표현
2. **Review-side 정보**: 사용자의 과거 리뷰와 아이템의 과거 리뷰에서 얻는 텍스트 의미 표현

모델의 핵심 가정은 다음과 같다.

> 리뷰에는 평점 예측에 도움이 되는 의미 정보가 있지만, 모든 리뷰 신호가 rating graph와 항상 일치하는 것은 아니다. 따라서 review representation을 rating representation과 공유되는 부분과 residual 부분으로 나누고, 두 표현의 정렬 정도에 따라 review 신호를 다르게 반영한다.

전체 흐름은 다음과 같다.

```text
(u, i, history reviews)
  -> RatingGraphEncoder로 zY 생성
  -> Review context encoder로 hX 생성
  -> PrototypeIntentExtractor로 zX 생성
  -> SharedResidualDisentangler로 zS, zR 분해
  -> InconsistencyGate로 gate 계산
  -> RatingPredictor + bias로 최종 평점 예측
  -> rating/alignment/separation/reconstruction/gate/prototype loss 학습
```

기호는 다음과 같이 둔다.

| 기호 | 의미 |
|---|---|
| `u` | 사용자 ID |
| `i` | 아이템 ID |
| `r_ui` | 실제 평점 |
| `\hat r_ui` | 예측 평점 |
| `d_id` | user/item ID embedding 차원 |
| `d_model` | 모델 내부 표현 차원 |
| `e_u`, `e_i` | user/item 초기 embedding |
| `zY` | rating graph 기반 표현 |
| `hX` | review context를 projection한 표현 |
| `zX` | prototype intent가 반영된 review 표현 |
| `zS` | rating-side와 공유되는 review 표현 |
| `zR` | shared로 설명되지 않는 residual review 표현 |
| `g` | gate 값. shared review 신호 반영 정도 |
| `p_inc` | inconsistency probability. `1 - g` |

## 2. 데이터 구성: target review 대신 history review 사용

기본 설정은 `review_context_mode: "history"`이다. 즉, 예측 대상 `(u, i)`의 target review를 직접 입력으로 쓰지 않고 user/item의 과거 리뷰 embedding을 사용한다. 이는 valid/test에서 target review를 쓰면 정답에 가까운 정보를 미리 보는 leakage가 생길 수 있기 때문이다.

### 2.1 User history와 item history

각 interaction `(u, i)`에 대해 user history와 item history를 만든다.

```text
H_u = {x_k | user가 u인 과거 interaction의 review embedding}
H_i = {x_k | item이 i인 과거 interaction의 review embedding}
```

현재 구현에서 train split은 기본적으로 `retain_rui: false`이므로 target interaction 자신의 review를 history에서 제외한다.

```text
H_u <- H_u \ {x_ui}
H_i <- H_i \ {x_ui}
```

valid/test에서는 `_setup_evaluation()`이 train interactions만 history source로 사용한다. 따라서 valid/test target review embedding은 입력으로 쓰이지 않는다.

### 2.2 Mean history context

`history_encoder="mean"`일 때는 user history와 item history를 각각 평균낸다.

```math
\bar x_u = \frac{1}{|H_u|}\sum_{x \in H_u} x
```

```math
\bar x_i = \frac{1}{|H_i|}\sum_{x \in H_i} x
```

history가 비어 있으면 해당 평균 벡터는 0 벡터로 처리된다.

두 history 평균은 다음과 같이 결합된다.

```math
x_{hist} = [\bar x_u ; \bar x_i ; \bar x_u \odot \bar x_i ; |\bar x_u - \bar x_i|]
```

각 항의 의미는 다음과 같다.

| 항 | 의미 |
|---|---|
| `\bar x_u` | 사용자가 과거에 남긴 리뷰들의 평균 의미 |
| `\bar x_i` | 아이템에 달린 과거 리뷰들의 평균 의미 |
| `\bar x_u \odot \bar x_i` | user review 성향과 item review 특성의 상호작용 |
| `|\bar x_u - \bar x_i|` | user와 item review context의 차이 |

### 2.3 Attention history context

`history_encoder="attention"`일 때는 rating graph 표현 `zY`를 query로 사용해 user/item history에서 현재 `(u, i)`에 더 관련 있는 리뷰를 가중합한다.

먼저 raw review embedding `x_k`를 key space로 projection한다.

```math
k_t = W_K x_t
```

rating-side 표현 `zY`와 key의 dot product로 attention score를 계산한다.

```math
s_t = \frac{zY^\top k_t}{\sqrt{d_{model}}}
```

padding 위치는 mask로 제외하고 softmax를 적용한다.

```math
\alpha_t = \frac{\exp(s_t)}{\sum_{j \in H}\exp(s_j)}
```

최종 context는 raw review embedding의 weighted sum이다.

```math
c_u = \sum_{t \in H_u}\alpha_t x_t, \qquad c_i = \sum_{t \in H_i}\alpha_t x_t
```

user context와 item context도 mean 방식과 같은 형태로 결합된다.

```math
x_{hist} = [c_u ; c_i ; c_u \odot c_i ; |c_u - c_i|]
```

이 방식의 의미는 “현재 user-item pair의 rating-side 신호를 기준으로, history 중 어떤 리뷰를 더 볼지 선택한다”는 것이다.

## 3. RatingGraphEncoder: rating-side 표현 `zY`

RatingGraphEncoder는 사용자 노드와 아이템 노드로 이루어진 bipartite graph를 사용한다. 각 interaction은 양방향 edge로 들어간다.

```text
u -> i
i -> u
```

초기 노드 embedding은 다음과 같다.

```math
h_u^{(0)} = e_u, \qquad h_i^{(0)} = e_i
```

한 번의 graph propagation은 정규화된 이웃 합으로 계산된다.

```math
h_v^{(l+1)} = \sum_{a \in \mathcal{N}(v)} \frac{w_{av}}{\sqrt{d_a d_v}} h_a^{(l)}
```

여기서 `w_av`는 edge weight이며 현재 dataset에서는 기본적으로 `None`이므로 1로 처리된다. `d_a`, `d_v`는 edge weight를 반영한 degree이다.

여러 layer의 출력을 평균내 최종 node embedding을 만든다.

```math
h_v = \frac{1}{L+1}\sum_{l=0}^{L} h_v^{(l)}
```

target user/item의 embedding을 뽑아 interaction representation을 만든다.

```math
m_{ui} = [h_u ; h_i ; h_u \odot h_i]
```

이를 linear projection하여 rating-side 표현 `zY`를 얻는다.

```math
zY = W_Y m_{ui} + b_Y
```

`zY`는 “리뷰를 보지 않고 user-item rating graph만으로 얻은 협업적 선호 표현”이다.

## 4. ReviewProjectionEncoder: review context 표현 `hX`

history에서 만든 `x_hist`는 MLP와 layer normalization을 거쳐 `hX`가 된다.

```math
hX = \mathrm{LayerNorm}(W_2 \; \mathrm{GELU}(W_1 x_{hist} + b_1) + b_2)
```

여기서 `hX`는 “현재 user-item pair 주변의 리뷰 의미를 `d_model` 차원으로 정리한 표현”이다.

## 5. PrototypeIntentExtractor: multi-intent review 표현 `zX`

리뷰는 하나의 단일 의미만 담는 것이 아니라 여러 intent를 가질 수 있다. 예를 들어 음악 상품 리뷰라면 “음질”, “배송”, “가격”, “브랜드 선호” 등이 섞일 수 있다. IARD-RM은 learnable prototype을 사용해 review-side 표현을 multi-intent space로 보낸다.

prototype matrix를 다음과 같이 둔다.

```math
P = [p_1, p_2, \dots, p_K]^\top \in \mathbb{R}^{K \times d_{model}}
```

`hX`와 각 prototype을 cosine 기반으로 비교한다.

```math
s_k = \frac{\mathrm{cos}(hX, p_k)}{\tau_p}
```

`\tau_p`는 prototype softmax temperature이다. 작을수록 특정 prototype에 더 날카롭게 집중한다.

prototype weight는 softmax로 계산한다.

```math
a_k = \frac{\exp(s_k)}{\sum_{j=1}^{K}\exp(s_j)}
```

prototype 기반 review intent 표현은 다음과 같다.

```math
z_{proto} = \sum_{k=1}^{K} a_k p_k
```

최종 `zX`는 원래 `hX`와 prototype representation을 residual 방식으로 더한 뒤 정규화한다.

```math
zX = \mathrm{LayerNorm}(hX + z_{proto})
```

`zX`의 의미는 “history review context를 여러 intent prototype으로 해석한 review-side 표현”이다.

## 6. SharedResidualDisentangler: `zX`를 `zS`와 `zR`로 분해

IARD-RM은 review-side 표현 `zX`를 두 부분으로 나눈다.

| 표현 | 의미 |
|---|---|
| `zS` | rating-side 표현 `zY`와 공유되는 review signal |
| `zR` | rating-side와 바로 정렬되지 않는 residual review signal |

기본 설정은 `disentangler_mode="conditioned"`이다. 이때 disentangler 입력은 `zX`와 `zY`를 함께 사용한다.

```math
q = [zX ; zY ; zX \odot zY ; |zX - zY|]
```

두 개의 MLP가 각각 shared와 residual 표현을 만든다.

```math
zS = \mathrm{LayerNorm}(f_S(q))
```

```math
zR = \mathrm{LayerNorm}(f_R(q))
```

또한 reconstruction branch는 `[zS; zR]`로부터 원래 review intent 표현 `zX`를 복원하도록 학습된다.

```math
\tilde zX = \mathrm{LayerNorm}(f_{rec}([zS ; zR]))
```

이 구조의 의미는 다음과 같다.

1. `zS`는 rating graph와 잘 맞는 리뷰 의미를 담도록 유도된다.
2. `zR`은 `zY`나 `zS`와 겹치지 않는 나머지 리뷰 의미를 담도록 유도된다.
3. `zS`와 `zR`을 합치면 원래 `zX`를 복원할 수 있어야 하므로, 분해 과정에서 review 정보 전체가 사라지지 않도록 한다.

`disentangler_mode="independent"`를 쓰면 입력 `q` 대신 `zX`만 사용한다.

```math
q = zX
```

## 7. InconsistencyGate: 정렬 정도에 따른 review fusion

gate는 `zY`와 `zS`가 얼마나 잘 정렬되어 있는지를 보고 shared review signal과 residual review signal의 반영 비중을 조절한다.

먼저 cosine alignment를 계산한다.

```math
a = \mathrm{cos}(zY, zS)
```

`a`가 크면 rating-side 표현과 shared review 표현이 비슷하다는 뜻이고, 작거나 음수이면 둘이 잘 맞지 않는다는 뜻이다.

학습 가능한 raw gate는 다음 입력으로 계산된다.

```math
v_g = [zY ; zS ; zR ; zY \odot zS ; |zY - zS|]
```

```math
g_{raw} = \sigma(f_g(v_g))
```

alignment 기반 gate는 다음과 같다.

```math
g_{align} = \sigma(\alpha a)
```

여기서 `\alpha`는 `gate_alpha`이며 alignment에 대한 gate 민감도를 조절한다.

최종 gate는 raw gate와 alignment gate의 평균이다.

```math
g = \frac{1}{2}g_{raw} + \frac{1}{2}g_{align}
```

inconsistency probability는 다음과 같이 정의된다.

```math
p_{inc} = 1 - g
```

직관적으로는 다음과 같다.

| 값 | 해석 |
|---|---|
| `g`가 큼 | rating graph와 shared review가 잘 맞으므로 shared review signal을 더 믿음 |
| `g`가 작음 | 두 표현이 덜 맞으므로 residual review signal의 상대적 비중이 커짐 |
| `p_inc`가 큼 | rating-side와 review-side 사이의 불일치 가능성이 큼 |

`fixed_gate_value`가 설정되면 gate를 학습하지 않고 해당 상수로 고정한다. 예를 들어 `full_fixed_gate` preset은 `g=0.5`를 사용한다.

## 8. RatingPredictor: 최종 평점 예측

RatingPredictor는 세 개의 head를 가진다.

```math
yY = f_Y(zY)
```

```math
yS = f_S(zS)
```

```math
yR = f_R(zR)
```

각 항의 의미는 다음과 같다.

| 항 | 의미 |
|---|---|
| `yY` | rating graph만 보고 예측한 기본 평점 신호 |
| `yS` | rating-side와 정렬되는 shared review 신호 |
| `yR` | rating-side와 덜 정렬되는 residual review 신호 |

fusion 전 예측식은 다음과 같다.

```math
\hat r_{ui}^{raw}
= yY
+ s_{shared} \cdot g \cdot yS
+ s_{residual} \cdot (1-g) \cdot \eta \cdot yR
```

여기서 `s_shared`는 `shared_fusion_scale`, `s_residual`은 `residual_fusion_scale`, `\eta`는 residual branch의 크기를 조절하는 계수이다.

최종 예측은 user bias, item bias, global bias를 더한다.

```math
\hat r_{ui}
= \hat r_{ui}^{raw} + b_u + b_i + b_0
```

현재 구현에서 `b_u`, `b_i`는 0으로 초기화되고, `b_0`는 train rating 평균으로 초기화된다.

이 식의 의미는 다음과 같다.

1. `yY`는 항상 기본 예측의 중심이 된다.
2. `g`가 크면 `yS`가 강하게 반영된다.
3. `g`가 작으면 `(1-g)`가 커지므로 residual branch `yR`의 반영 비중이 커진다.
4. bias term은 특정 user/item의 평균적 평점 성향과 전체 평균 평점을 보정한다.

## 9. 손실 함수 전체 구조

최종 objective는 여러 loss의 weighted sum이다.

```math
\mathcal{L}
= \lambda_{rating}\mathcal{L}_{rating}
+ \lambda_{align}\mathcal{L}_{align}
+ \lambda_{sep}\mathcal{L}_{sep}
+ \lambda_{recon}\mathcal{L}_{recon}
+ \lambda_{gate}\mathcal{L}_{gate}
+ \lambda_{proto}\mathcal{L}_{proto}
```

각 loss의 구현과 의미는 아래와 같다.

## 10. Rating loss

평점 예측 오차는 MSE로 계산한다.

```math
\mathcal{L}_{rating}
= \frac{1}{B}\sum_{n=1}^{B}(\hat r_n - r_n)^2
```

의미는 단순하다. 최종 예측 평점 `\hat r`이 실제 평점 `r`에 가까워지도록 학습한다. 이 항이 모델의 주된 supervised objective이다.

## 11. Alignment loss

alignment loss는 batch 안에서 `zY_n`과 같은 sample의 `zS_n`이 서로 가까워지도록 하는 contrastive loss이다.

먼저 정규화된 similarity logit을 만든다.

```math
\ell_{nm}
= \frac{\mathrm{cos}(zY_n, zS_m)}{\tau_c}
```

여기서 `\tau_c`는 contrastive temperature이다.

각 `zY_n`에 대해 정답 class는 같은 sample의 `zS_n`이다.

```math
\mathcal{L}_{nce,n}
= -\log \frac{\exp(\ell_{nn})}{\sum_{m=1}^{B}\exp(\ell_{nm})}
```

구현에서는 gate로 sample별 NCE loss를 weighting한다.

```math
\mathcal{L}_{align}
= \frac{1}{B}\sum_{n=1}^{B} g_n \mathcal{L}_{nce,n}
```

기본 설정 `detach_gate_for_align: true`에서는 `g_n`을 detach하여 alignment loss가 gate 자체를 직접 흔들지 않도록 한다.

이 loss의 의미는 다음과 같다.

> shared representation `zS`는 residual이 아니라 rating-side와 공유되는 리뷰 정보여야 하므로, 같은 sample의 `zY`와 가깝고 다른 sample의 `zS`와는 구분되도록 만든다.

## 12. Separation loss

separation loss는 residual representation `zR`이 `zY`, `zS`와 너무 겹치지 않도록 하는 항이다.

```math
\mathcal{L}_{sep}
= \frac{1}{B}\sum_{n=1}^{B}\mathrm{cos}(zY_n, zR_n)^2
+ \frac{1}{B}\sum_{n=1}^{B}\mathrm{cos}(zS_n, zR_n)^2
```

cosine 값을 제곱하기 때문에 양의 상관과 음의 상관을 모두 penalty로 본다. 최소화되려면 `zR`은 `zY`, `zS`와 직교에 가까워져야 한다.

의미는 다음과 같다.

> residual branch가 shared/rating branch와 같은 정보를 반복해서 담지 않도록 막고, rating-side와 다른 review-specific 정보를 담당하게 만든다.

## 13. Reconstruction loss

분해된 `zS`, `zR`이 원래 review intent 표현 `zX`를 복원할 수 있도록 한다.

```math
\tilde zX = f_{rec}([zS ; zR])
```

```math
\mathcal{L}_{recon}
= \frac{1}{B}\sum_{n=1}^{B}\|\tilde zX_n - zX_n\|_2^2
```

`detach_zx_for_recon`이 true이면 target `zX`를 detach하지만, 기본 설정은 false이다.

이 loss의 의미는 다음과 같다.

> `zX`를 `zS`와 `zR`로 나누더라도 review-side 정보가 사라지지 않게 한다. 즉, disentanglement가 정보 손실이 아니라 정보 분해가 되도록 유도한다.

## 14. Gate loss

gate loss는 gate의 binary entropy에 음수를 붙인 형태이다.

```math
H(g_n) = -g_n\log(g_n + \epsilon) - (1-g_n)\log(1-g_n + \epsilon)
```

```math
\mathcal{L}_{gate}
= -\frac{1}{B}\sum_{n=1}^{B}H(g_n)
```

전체 loss는 이 값을 최소화하므로, `\lambda_gate`가 양수이면 entropy를 크게 만드는 방향으로 작동한다. 즉 gate가 지나치게 0이나 1로 빨리 포화되는 것을 완화하고 더 부드러운 fusion을 유도한다.

의미는 다음과 같다.

> 학습 초기에 gate가 한쪽 branch만 선택하는 것을 막고, shared/residual review signal을 모두 탐색할 여지를 준다.

## 15. Prototype orthogonality loss

prototype들이 서로 같은 의미로 붕괴하지 않도록 정규화된 prototype의 Gram matrix를 identity matrix에 가깝게 만든다.

정규화 prototype을 `\bar p_k`라고 하면,

```math
G_{kl} = \bar p_k^\top \bar p_l
```

prototype loss는 다음과 같다.

```math
\mathcal{L}_{proto}
= \frac{1}{K^2}\sum_{k=1}^{K}\sum_{l=1}^{K}(G_{kl} - I_{kl})^2
```

의미는 다음과 같다.

> 서로 다른 prototype이 서로 다른 intent를 담당하도록 하여 multi-intent representation의 다양성을 유지한다.

## 16. Loss preset의 방법론적 의미

`main.py`의 `apply_iard_loss_preset()`은 IARD-RM의 ablation을 쉽게 수행하기 위해 여러 계수를 한 번에 바꾼다.

| Preset | 방법론적 의미 |
|---|---|
| `rating_only` | review fusion과 auxiliary loss를 모두 끄고 rating graph + bias만 보는 baseline |
| `review_fusion` | review branch는 예측에 사용하지만 alignment/separation/reconstruction/gate/prototype 정규화는 끈 설정 |
| `full_iard` | shared/residual 분해와 모든 auxiliary loss를 사용하는 전체 모델 |
| `full_no_sep` | residual을 분리시키는 separation loss의 효과를 제거 |
| `full_low_align` | rating-side와 shared review의 alignment 강도를 낮춤 |
| `full_no_residual_pred` | residual branch가 prediction에 직접 기여하지 못하게 함 |
| `full_fixed_gate` | gate를 학습하지 않고 0.5로 고정해 adaptive gating 효과를 제거 |

현재 코드 기준으로 지원되는 preset은 위 목록이다. README에 언급된 `full_eta_0_3`은 현재 `apply_iard_loss_preset()`의 지원 목록에는 없다.

## 17. 평가와 해석 지표

기본 평가지표는 rating prediction metric이다.

| 지표 | 의미 |
|---|---|
| RMSE | 큰 오차에 더 민감한 평점 예측 오차 |
| MSE | squared error 평균 |
| MAE | 절대 오차 평균 |

IARD-RM trainer는 내부 해석을 위해 다음 값도 기록한다.

| 지표 | 의미 |
|---|---|
| `mean_gate`, `std_gate` | shared review signal이 얼마나 반영되는지 |
| `mean_p_inc`, `std_p_inc` | rating-review 불일치 가능성 |
| `mean_alignment`, `std_alignment` | `zY`와 `zS`의 cosine alignment |
| `mean_intent_entropy` | prototype intent weight가 얼마나 분산되어 있는지 |
| `mean_empty_history_ratio` | history가 비어 있는 sample 비율 |
| `history_source_ratio` | valid/test에서 train history context가 사용되었는지 확인 |

test 단계에서는 `test_sample_analysis.csv`가 저장되며, sample별 `pred`, `abs_error`, `gate`, `p_inc`, `alignment`, `intent_entropy`, `yY`, `yS`, `yR` 등을 볼 수 있다.

## 18. 구현상 중요한 주의점

1. **valid/test leakage 방지**: `review_context_mode="history"`에서는 valid/test target review를 사용하지 않고 train history만 사용한다.
2. **rating graph도 train graph 사용**: evaluation setup에서 valid/test dataset의 `edge_index`는 train dataframe으로부터 다시 구성된다.
3. **history가 비어 있는 경우**: mean context는 0 벡터, attention context는 mask 처리 후 0 context로 처리되어 NaN을 방지한다.
4. **prediction clipping**: evaluation 설정의 `clamp_eval_pred: true`에 따라 평가 시 rating 범위로 clipping할 수 있다. trainer 내부 metric 계산은 raw prediction 기반 metric과 clipped prediction 기반 group metric이 함께 쓰이는 부분이 있으므로 결과 해석 시 저장 로그를 확인하는 것이 좋다.
5. **preset override**: `loss_preset`이 먼저 기본 계수를 적용하고, CLI에서 명시한 IARD 관련 옵션은 다시 복원된다.

## 19. 한 문장 요약

IARD-RM은 rating graph에서 얻은 협업적 표현 `zY`와 history review에서 얻은 의미 표현 `zX`를 정렬 가능한 shared 성분 `zS`와 불일치/residual 성분 `zR`로 분해한 뒤, alignment 기반 gate로 두 review 신호를 조절해 평점을 예측하는 review-aware rating prediction 모델이다.
