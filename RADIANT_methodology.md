# RADIANT: Rating-Anchored Disentangled Multi-Intent Review Learning

## 0. 연구 방향 요약

본 연구의 핵심 문제의식은 review-based rating prediction에서 review를 단순히 rating prediction을 보조하는 추가 정보로 사용하는 기존 관점의 한계를 다시 보는 것이다. Rating과 review는 동일한 user-item experience에서 생성되지만, 두 feedback은 서로 다른 성격을 가진다.

- Rating은 사용자가 경험한 item에 대한 전체 만족도 또는 utility를 제한된 scale 위에 압축한 scalar judgment이다.
- Review는 사용자가 rating을 남긴 이유, 구매 이유, 선호 이유, 불만 요소 중 일부만 선택적으로 언어화한 partial textual expression이다.
- 따라서 review가 rating에 포함된 모든 preference signal을 드러내지는 않는다.
- Rating은 높지만 review는 부정적인 aspect를 언급할 수 있고, rating은 낮지만 review 안에는 긍정적인 aspect가 존재할 수 있다.
- 이러한 불일치는 단순 sentiment mismatch가 아니라 rating을 결정한 dominant aspect와 review에 명시된 expressed aspect의 중요도 차이에서 발생할 수 있다.

따라서 본 연구는 inconsistent interaction을 noise로 제거하지 않는다. 대신 review가 rating에서 낮은 가중치를 받은 aspect, minor aspect, residual preference signal을 드러낸 것으로 재해석한다.

본 문서에서 제안하는 모델 이름은 다음과 같다.

> **RADIANT: Rating-Anchored Disentangled Multi-Intent Review Learning**

핵심 아이디어는 다음과 같다.

1. Rating graph encoder는 user-item rating interaction으로부터 rating utility factor \(z_{ui}^{Y}\)를 학습한다.
2. Review encoder는 review text에서 multi-intent 또는 multi-factor representation \(z_{ui,k}^{X}\)를 추출한다.
3. Disentanglement module은 review representation을 rating-aligned shared factor \(z_{ui}^{S}\)와 review-specific residual factor \(z_{ui}^{R}\)로 분리한다.
4. \(z_{ui}^{S}\)는 rating utility \(z_{ui}^{Y}\)와 align되어야 한다.
5. \(z_{ui}^{R}\)은 rating utility와 직접 align되지 않지만, 사용자가 경험한 minor aspect, low-weight aspect, review-specific signal을 담는다.
6. Inconsistency-aware gate \(g_{ui}\)는 interaction마다 review 정보를 rating prediction에 얼마나 반영할지 결정한다.
7. 최종 rating prediction은 rating graph signal, shared review signal, residual review signal의 gated combination으로 수행한다.

---

## 1. Problem Formulation

사용자 집합을 \(\mathcal{U}\), 아이템 집합을 \(\mathcal{I}\)라 하자. 관측된 interaction 집합은 다음과 같이 정의한다.

\[
\mathcal{D}=\{(u,i,y_{ui},x_{ui}) \mid u\in\mathcal{U}, i\in\mathcal{I}\}
\]

여기서:

- \(u\): user
- \(i\): item
- \(y_{ui}\in\{1,2,3,4,5\}\) 또는 \(y_{ui}\in\mathbb{R}\): user \(u\)가 item \(i\)에 남긴 rating
- \(x_{ui}\): 동일한 interaction에서 생성된 review text

목표는 다음과 같다.

\[
\hat{y}_{ui}=f(u,i,x_{ui},\mathcal{G}_Y)
\]

여기서 \(\mathcal{G}_Y\)는 user-item rating graph이다.

### 1.1 User / Item Review Document

사용자와 아이템의 review document는 다음과 같이 정의할 수 있다.

\[
D_u = \{x_{ui} \mid (u,i)\in\mathcal{D}\}
\]

\[
D_i = \{x_{ui} \mid (u,i)\in\mathcal{D}\}
\]

- \(D_u\): user \(u\)가 과거에 작성한 review들의 집합
- \(D_i\): item \(i\)에 대해 작성된 review들의 집합

기본형 모델에서는 interaction-level review \(x_{ui}\)만 사용하고, 확장형에서는 \(D_u\), \(D_i\)를 함께 사용할 수 있다.

### 1.2 Rating-based Utility Aspect와 Review-expressed Aspect

본 연구에서는 두 가지 latent aspect 개념을 구분한다.

#### Rating-based aspect / utility

\[
A_{ui}^{Y}
\]

이는 rating \(y_{ui}\)에 반영된 dominant utility factor 또는 rating-relevant aspect를 의미한다. 예를 들어 사용자가 item에 5점을 준 이유는 품질, 가격, 배송, 브랜드 신뢰도, 사용 편의성 중 일부일 수 있다. 하지만 rating은 scalar이므로 여러 요인이 하나의 값으로 압축된다.

#### Review-expressed aspect

\[
A_{ui}^{X}
\]

이는 review \(x_{ui}\)에 명시적으로 표현된 aspect를 의미한다. 사용자는 전체 만족도를 결정한 모든 요인을 review에 쓰지 않는다. 일부 이유, 불만, 예외적 경험, minor aspect만 선택적으로 표현할 수 있다.

따라서 일반적으로 다음이 성립할 수 있다.

\[
A_{ui}^{Y} \neq A_{ui}^{X}
\]

기존 review-based rating prediction은 보통 review를 rating prediction을 위한 side information으로 사용한다.

\[
\hat{y}_{ui}=f(u,i,x_{ui},D_u,D_i)
\]

반면 본 연구는 review representation 전체를 rating에 align하지 않는다. Review representation을 다음 두 부분으로 분리한다.

- Rating과 공유되는 정보: \(z_{ui}^{S}\)
- Review에 존재하지만 rating utility와 직접 정렬되지 않는 residual 정보: \(z_{ui}^{R}\)

---

## 2. Core Assumption

### Assumption 1. Rating is an overall utility anchor

Rating은 사용자의 전체 만족도 또는 utility를 제한된 scale 위에 압축한 scalar judgment이다.

\[
y_{ui} \sim U_{ui}^{Y}
\]

여기서 \(U_{ui}^{Y}\)는 user-item experience의 overall utility이다. 따라서 rating graph에서 학습되는 representation은 rating prediction의 anchor 역할을 해야 한다.

### Assumption 2. Review is a partial textual expression

Review는 전체 utility의 완전한 설명이 아니라 선택적으로 언어화된 partial expression이다.

\[
x_{ui} \sim \text{PartialExpression}(U_{ui}^{Y}, A_{ui}^{X})
\]

즉 review는 rating의 이유 전체가 아니라 일부 aspect만 드러낸다.

### Assumption 3. Inconsistency is not necessarily noise

High rating + negative review, low rating + positive review는 noise가 아닐 수 있다.

예를 들어:

- 5점이지만 “배송은 늦었다”라고 쓰는 경우: overall utility는 높지만 review는 minor negative aspect를 표현한다.
- 2점이지만 “디자인은 예쁘다”라고 쓰는 경우: overall utility는 낮지만 review는 positive sub-aspect를 표현한다.

따라서 inconsistency는 다음처럼 해석하는 것이 적절하다.

\[
A_{ui}^{Y} \neq A_{ui}^{X}
\]

또는:

\[
\text{dominant rating factor} \neq \text{expressed review factor}
\]

### Why not sentiment mismatch?

단순히 다음 조건으로 inconsistency를 정의해서는 안 된다.

\[
\text{sentiment}(x_{ui}) \neq \text{sentiment}(y_{ui})
\]

그 이유는 다음과 같다.

1. Review sentiment는 전체 review의 평균 감정일 뿐 aspect별 중요도를 반영하지 않는다.
2. Rating은 user-specific scale bias를 가진다. 어떤 사용자의 3점은 부정적일 수 있고, 다른 사용자의 3점은 보통일 수 있다.
3. Item별 rating distribution bias가 존재한다. 인기 item은 minor complaint가 있어도 높은 rating을 받을 수 있다.
4. Review는 mixed sentiment일 수 있다.
5. 핵심은 sentiment mismatch가 아니라 utility direction과 expressed aspect direction의 mismatch이다.

---

## 3. Model Architecture

RADIANT는 다음 모듈로 구성된다.

1. Rating Graph Encoder
2. Review Encoder
3. Multi-Intent / Multi-Factor Extraction
4. Shared-Specific Disentanglement
5. Soft Inconsistency Estimation
6. Inconsistency-Aware Gating
7. Rating Prediction Head

전체 구조는 다음과 같다.

\[
z_{ui}^{Y}=\text{RatingGraphEncoder}(u,i,\mathcal{G}_Y)
\]

\[
h_{ui}^{X}=\text{ReviewEncoder}(x_{ui})
\]

\[
\{z_{ui,k}^{X}\}_{k=1}^{K}=\text{MultiIntentExtractor}(h_{ui}^{X})
\]

\[
z_{ui}^{S},z_{ui}^{R}=\text{Disentangle}(\{z_{ui,k}^{X}\}_{k=1}^{K},z_{ui}^{Y})
\]

\[
g_{ui}=\text{Gate}(z_{ui}^{Y},z_{ui}^{S},z_{ui}^{R},b_{ui})
\]

\[
\hat{y}_{ui}=f_Y(z_{ui}^{Y})+g_{ui}f_S(z_{ui}^{S})+(1-g_{ui})\eta f_R(z_{ui}^{R})
\]

### 3.1 Rating Graph Encoder

Rating graph는 user-item rating interaction으로 만든 bipartite graph이다.

\[
\mathcal{G}_Y=(\mathcal{U}\cup\mathcal{I},\mathcal{E}_Y)
\]

\[
(u,i,y_{ui})\in\mathcal{E}_Y
\]

Backbone은 minimal version에서는 LightGCN을 사용하는 것이 적절하다. LightGCN은 graph propagation의 핵심인 비슷한 이웃 노드의 preference 전파를 유지하면서도 모델이 과도하게 복잡해지는 것을 막는다.

초기 embedding은 다음과 같다.

\[
e_u^{(0)},e_i^{(0)}
\]

LightGCN propagation은 다음과 같다.

\[
e_u^{(l+1)}=\sum_{i\in\mathcal{N}(u)}\frac{1}{\sqrt{|\mathcal{N}(u)||\mathcal{N}(i)|}}e_i^{(l)}
\]

\[
e_i^{(l+1)}=\sum_{u\in\mathcal{N}(i)}\frac{1}{\sqrt{|\mathcal{N}(i)||\mathcal{N}(u)|}}e_u^{(l)}
\]

최종 embedding은 layer-wise aggregation으로 얻는다.

\[
e_u^Y=\frac{1}{L+1}\sum_{l=0}^{L}e_u^{(l)}
\]

\[
e_i^Y=\frac{1}{L+1}\sum_{l=0}^{L}e_i^{(l)}
\]

Interaction utility representation은 다음과 같이 정의한다.

\[
z_{ui}^{Y}=\phi_Y(e_u^Y,e_i^Y)
\]

예를 들어:

\[
z_{ui}^{Y}=[e_u^Y \Vert e_i^Y \Vert e_u^Y\odot e_i^Y]
\]

이 \(z_{ui}^{Y}\)가 rating prediction의 anchor이다.

### 3.2 Review Encoder

각 review \(x_{ui}\)를 textual representation으로 변환한다.

Minimal version에서는 cached sentence embedding을 사용할 수 있다.

\[
h_{ui}^{X}=\text{MLP}(r_{ui})
\]

여기서 \(r_{ui}\in\mathbb{R}^{384}\)는 Sentence-BERT 또는 sentence-transformer embedding이다.

Full version에서는 BERT 또는 Transformer encoder를 사용할 수 있다.

\[
h_{ui}^{X}=\text{BERT}([\text{CLS}],x_{ui})
\]

User/item document를 사용하는 확장형도 가능하다.

\[
h_u^X=\text{DocEncoder}(D_u)
\]

\[
h_i^X=\text{DocEncoder}(D_i)
\]

그리고 interaction review representation은 다음처럼 만들 수 있다.

\[
h_{ui}^{X}=\psi(h_u^X,h_i^X,h_{ui}^{review})
\]

그러나 기본형에서는 \(x_{ui}\) 단일 review embedding만 사용하는 것이 구현 안정성이 높다.

### 3.3 Multi-Intent / Multi-Factor Extraction

Review representation \(h_{ui}^{X}\)에서 \(K\)개의 latent intent/factor를 추출한다.

\[
\{z_{ui,1}^{X},z_{ui,2}^{X},\dots,z_{ui,K}^{X}\}
\]

가장 구현 가능한 방식은 prototype attention이다.

학습 가능한 prototype matrix를 다음과 같이 둔다.

\[
P=[p_1,p_2,\dots,p_K]\in\mathbb{R}^{K\times d}
\]

각 intent weight는 다음과 같다.

\[
\alpha_{ui,k}=\frac{\exp((W_qh_{ui}^{X})^\top p_k/\tau)}{\sum_{k'=1}^{K}\exp((W_qh_{ui}^{X})^\top p_{k'}/\tau)}
\]

각 intent representation은 다음과 같다.

\[
z_{ui,k}^{X}=\alpha_{ui,k}W_kh_{ui}^{X}
\]

각 factor는 명시적 aspect label 없이 latent aspect-level signal로 해석한다.

명시적 aspect extraction을 기본 가정으로 두지 않는 이유는 다음과 같다.

1. Aspect label이 없는 dataset이 많다.
2. Aspect extractor 오류가 downstream으로 전파될 수 있다.
3. Rating-review mismatch는 명시 aspect보다 latent preference factor에서 더 잘 드러날 수 있다.
4. Surface aspect보다 user-specific utility factor가 더 중요할 수 있다.

### 3.4 Shared-Specific Disentanglement

Review factors를 두 부분으로 분리한다.

- \(z_{ui}^{S}\): rating-aligned shared factor
- \(z_{ui}^{R}\): review-specific residual factor

먼저 multi-intent factors를 attention pooling한다.

\[
\bar{z}_{ui}^{X}=\sum_{k=1}^{K}\alpha_{ui,k}z_{ui,k}^{X}
\]

그 다음 두 projection head를 둔다.

\[
z_{ui}^{S}=f_S^{enc}(\bar{z}_{ui}^{X},z_{ui}^{Y})
\]

\[
z_{ui}^{R}=f_R^{enc}(\bar{z}_{ui}^{X},z_{ui}^{Y})
\]

목표는 다음이다.

\[
I(z^S;z^Y) \uparrow
\]

\[
I(z^R;z^Y) \downarrow
\]

\[
I(z^R;x) \text{ remains non-trivial}
\]

즉 \(z^S\)는 rating utility와 공유되는 review signal이고, \(z^R\)은 rating utility와 직접 align되지 않는 residual review signal이다.

Collapse를 막기 위해 reconstruction을 둔다.

\[
\hat{h}_{ui}^{X}=Dec(z_{ui}^{S},z_{ui}^{R})
\]

\[
\mathcal{L}_{rec}=||h_{ui}^{X}-\hat{h}_{ui}^{X}||_2^2
\]

또한 orthogonality regularization을 적용한다.

\[
\mathcal{L}_{orth}=||(Z^S)^\top Z^R||_F^2
\]

### 3.5 Inconsistency Estimation

Inconsistency는 hard sentiment mismatch가 아니라 soft alignment score로 정의한다.

Shared alignment score는 다음과 같다.

\[
a_{ui}=\text{sim}(z_{ui}^{Y},z_{ui}^{S})
\]

예를 들어 cosine similarity를 사용할 수 있다.

\[
a_{ui}=\cos(z_{ui}^{Y},z_{ui}^{S})
\]

또는 review-only rating predictor를 auxiliary로 둔다.

\[
\tilde{y}_{ui}^{X}=f_X(h_{ui}^{X})
\]

\[
d_{ui}=|y_{ui}-\tilde{y}_{ui}^{X}|
\]

이 \(d_{ui}\)는 sentiment mismatch가 아니라 review representation이 예측하는 rating direction과 actual rating의 차이다.

User-specific rating scale을 반영하기 위해 normalized deviation을 정의한다.

\[
\mu_u=\frac{1}{|\mathcal{N}(u)|}\sum_{i\in\mathcal{N}(u)}y_{ui}
\]

\[
\sigma_u=\text{Std}(\{y_{ui}:i\in\mathcal{N}(u)\})
\]

\[
\Delta_{ui}^{u}=\frac{y_{ui}-\mu_u}{\sigma_u+\epsilon}
\]

Item rating distribution bias도 고려한다.

\[
\mu_i=\frac{1}{|\mathcal{N}(i)|}\sum_{u\in\mathcal{N}(i)}y_{ui}
\]

\[
\Delta_{ui}^{i}=\frac{y_{ui}-\mu_i}{\sigma_i+\epsilon}
\]

Bias feature는 다음과 같이 둘 수 있다.

\[
b_{ui}=[\Delta_{ui}^{u},\Delta_{ui}^{i},\log|\mathcal{N}(u)|,\log|\mathcal{N}(i)|]
\]

Soft inconsistency probability는 다음과 같다.

\[
p_{inc}(ui)=\sigma(\text{MLP}([z_{ui}^{Y},z_{ui}^{S},z_{ui}^{R},a_{ui},d_{ui},b_{ui}]))
\]

Agreement gate는 다음처럼 정의한다.

\[
g_{ui}=1-p_{inc}(ui)
\]

### 3.6 Inconsistency-Aware Gating

최종 prediction은 세 가지 신호의 gated combination이다.

\[
\hat{y}_{ui}=f_Y(z_{ui}^{Y})+g_{ui}f_S(z_{ui}^{S})+(1-g_{ui})\eta f_R(z_{ui}^{R})
\]

여기서:

- \(f_Y(z^Y)\): rating graph 기반 utility prediction
- \(f_S(z^S)\): rating과 공유되는 review signal
- \(f_R(z^R)\): review-specific residual signal
- \(g_{ui}\in[0,1]\): rating-review agreement gate
- \(\eta\): residual signal의 영향력을 제한하는 coefficient

해석은 다음과 같다.

- Consistent interaction: \(g_{ui}\approx 1\), shared review signal \(z^S\)를 강하게 사용한다.
- Inconsistent interaction: \(g_{ui}\approx 0\), \(z^S\)의 강제 alignment를 줄이고 \(z^R\)을 약하게 활용한다.

중요한 점은 inconsistent review를 제거하지 않는다는 것이다. 대신 residual factor로 재해석한다.

---

## 4. Loss Function

### 4.1 Rating Prediction Loss

Rating regression에서는 MSE를 사용한다.

\[
\mathcal{L}_{pred}=\frac{1}{|\mathcal{B}|}\sum_{(u,i)\in\mathcal{B}}(y_{ui}-\hat{y}_{ui})^2
\]

MAE를 보조 loss로 둘 수도 있다.

\[
\mathcal{L}_{mae}=\frac{1}{|\mathcal{B}|}\sum_{(u,i)\in\mathcal{B}}|y_{ui}-\hat{y}_{ui}|
\]

Ranking task로 확장하면 BPR loss를 사용할 수 있다.

\[
\mathcal{L}_{BPR}=-\sum_{(u,i,j)}\log\sigma(\hat{y}_{ui}-\hat{y}_{uj})
\]

### 4.2 Shared Alignment Contrastive Loss

\(z^Y\)와 \(z^S\)를 positive pair로 둔다.

기본 InfoNCE는 다음과 같다.

\[
\ell_{ui}^{align}=-\log\frac{\exp(\text{sim}(z_{ui}^{Y},z_{ui}^{S})/\tau)}{\sum_{(u',i')\in\mathcal{B}}\exp(\text{sim}(z_{ui}^{Y},z_{u'i'}^{S})/\tau)}
\]

하지만 모든 review를 강제로 align하면 안 되므로 agreement weight를 둔다.

\[
w_{ui}^{agr}=g_{ui}
\]

또는 stop-gradient를 적용한다.

\[
w_{ui}^{agr}=\text{sg}(1-p_{inc}(ui))
\]

최종 alignment loss는 다음과 같다.

\[
\mathcal{L}_{align}=\frac{1}{|\mathcal{B}|}\sum_{(u,i)\in\mathcal{B}}w_{ui}^{agr}\ell_{ui}^{align}
\]

Consistent할수록 alignment가 강하고, inconsistent할수록 약하다.

### 4.3 Residual Separation Loss

Residual \(z^R\)이 rating utility \(z^Y\)와 같은 정보를 중복해서 담지 않도록 한다.

#### Option 1. Orthogonality

\[
\mathcal{L}_{orth}=||(Z^Y)^\top Z^R||_F^2+||(Z^S)^\top Z^R||_F^2
\]

#### Option 2. CLUB / MI Upper Bound Minimization

CLUB는 mutual information upper bound를 최소화한다.

\[
I(z^R;z^Y)\leq \mathbb{E}_{p(z^R,z^Y)}[\log q_\theta(z^Y|z^R)]-\mathbb{E}_{p(z^R)p(z^Y)}[\log q_\theta(z^Y|z^R)]
\]

이를 loss로 두고 최소화한다.

\[
\mathcal{L}_{MI}^{R,Y}=\widehat{I}_{CLUB}(z^R,z^Y)
\]

Minimal version에서는 orthogonality가 구현하기 쉽고, full version에서는 CLUB를 넣는 것이 좋다.

### 4.4 Information Bottleneck Loss

\(z^S\)는 rating-relevant minimal sufficient representation이 되어야 한다.

Variational form은 다음과 같다.

\[
q_\phi(z^S|h^X)=\mathcal{N}(\mu_S,\sigma_S^2I)
\]

\[
q_\phi(z^R|h^X)=\mathcal{N}(\mu_R,\sigma_R^2I)
\]

IB regularization은 다음과 같다.

\[
\mathcal{L}_{IB}^{S}=D_{KL}(q_\phi(z^S|h^X)||p(z^S))
\]

\[
\mathcal{L}_{IB}^{R}=D_{KL}(q_\phi(z^R|h^X)||p(z^R))
\]

전체 IB loss는 다음과 같다.

\[
\mathcal{L}_{IB}=\beta_S\mathcal{L}_{IB}^{S}+\beta_R\mathcal{L}_{IB}^{R}
\]

해석은 다음과 같다.

- \(z^S\): rating prediction에 충분하지만 불필요한 review noise는 제거한다.
- \(z^R\): review-specific signal을 담되 rating에 과적합하지 않게 제한한다.

Minimal version에서는 dropout, projection bottleneck, \(L_2\) regularization으로 대체 가능하다.

### 4.5 Inconsistency-aware Regularization

Gate collapse를 막기 위해 entropy regularization을 둔다.

\[
\mathcal{L}_{ent}=-\frac{1}{|\mathcal{B}|}\sum_{(u,i)\in\mathcal{B}}[g_{ui}\log g_{ui}+(1-g_{ui})\log(1-g_{ui})]
\]

하지만 entropy를 무조건 maximize하면 gate가 0.5에 머무를 수 있다. 따라서 target prior를 둔다.

\[
\bar{g}=\frac{1}{|\mathcal{B}|}\sum g_{ui}
\]

\[
\mathcal{L}_{prior}=|\bar{g}-\rho|
\]

여기서 \(\rho\)는 expected agreement ratio이다.

Rating distribution bias를 줄이기 위한 counterfactual prediction도 가능하다.

\[
\hat{y}_{ui}^{cf}=f_Y(z_{ui}^{Y})+g_{ui}^{cf}f_S(z_{ui}^{S})+(1-g_{ui}^{cf})\eta f_R(z_{ui}^{R})
\]

여기서 \(g_{ui}^{cf}\)는 user/item rating bias feature를 평균값으로 대체한 gate이다.

\[
b_{ui}\rightarrow \bar{b}
\]

Bias sensitivity를 제한한다.

\[
\mathcal{L}_{cf}=||\hat{y}_{ui}-\hat{y}_{ui}^{cf}||_2^2
\]

### 4.6 Total Objective

전체 loss는 다음과 같다.

\[
\mathcal{L}=\mathcal{L}_{pred}+\lambda_{align}\mathcal{L}_{align}+\lambda_{sep}\mathcal{L}_{sep}+\lambda_{rec}\mathcal{L}_{rec}+\lambda_{IB}\mathcal{L}_{IB}+\lambda_{gate}\mathcal{L}_{gate}+\lambda_{cf}\mathcal{L}_{cf}
\]

여기서:

\[
\mathcal{L}_{sep}=\mathcal{L}_{orth}+\mathcal{L}_{MI}^{R,Y}
\]

\[
\mathcal{L}_{gate}=\mathcal{L}_{ent}+\mathcal{L}_{prior}
\]

각 coefficient의 의미는 다음과 같다.

- \(\lambda_{align}\): shared factor와 rating anchor의 alignment 강도
- \(\lambda_{sep}\): residual factor separation 강도
- \(\lambda_{rec}\): \(z^S,z^R\)이 review information을 유지하도록 하는 reconstruction 강도
- \(\lambda_{IB}\): information bottleneck compression 강도
- \(\lambda_{gate}\): gate collapse 방지 강도
- \(\lambda_{cf}\): rating distribution/popularity bias robustness 강도

---

## 5. Training Procedure

### Step 1. Rating Graph Encoder Pretraining

먼저 rating graph encoder만 학습한다.

\[
\hat{y}_{ui}^{Y}=f_Y(z_{ui}^{Y})
\]

\[
\mathcal{L}_{pre}^{Y}=||y_{ui}-\hat{y}_{ui}^{Y}||_2^2
\]

이 단계의 목적은 stable rating utility anchor를 만드는 것이다.

### Step 2. Review Encoder and Multi-Intent Extraction

Review encoder와 prototype intent module을 학습한다.

Auxiliary review-only predictor를 둔다.

\[
\tilde{y}_{ui}^{X}=f_X(h_{ui}^{X})
\]

\[
\mathcal{L}_{aux}^{X}=||y_{ui}-\tilde{y}_{ui}^{X}||_2^2
\]

이 auxiliary prediction은 review를 rating에 완전히 align하기 위한 것이 아니라 soft divergence signal \(d_{ui}=|y_{ui}-\tilde{y}_{ui}^{X}|\)를 얻기 위한 것이다.

### Step 3. Shared/Residual Disentanglement

Pretrained \(z^Y\)를 anchor로 두고 \(z^S,z^R\)를 학습한다.

\[
\mathcal{L}_{align}+\mathcal{L}_{sep}+\mathcal{L}_{rec}
\]

이 단계에서 \(z^S\)는 rating anchor와 가까워지고, \(z^R\)은 rating anchor와 분리된다.

### Step 4. Inconsistency-Aware Gating

Gate \(g_{ui}\)를 학습한다.

\[
\hat{y}_{ui}=f_Y(z^Y)+gf_S(z^S)+(1-g)\eta f_R(z^R)
\]

최종 prediction loss와 gate regularization을 함께 적용한다.

### End-to-End Fine-tuning

마지막에는 전체 모델을 end-to-end로 fine-tuning한다.

권장 학습 순서는 다음과 같다.

1. Rating graph pretraining
2. Review auxiliary pretraining
3. Disentanglement warm-up
4. Gate warm-up
5. End-to-end fine-tuning

완전 end-to-end 학습도 가능하지만, \(z^S,z^R,g\)가 동시에 collapse할 위험이 있으므로 staged training이 더 안정적이다.

---

## 6. Experiments

### 6.1 Dataset

사용 가능한 dataset은 다음과 같다.

- Amazon Review Dataset
- Yelp
- TripAdvisor
- BeerAdvocate / RateBeer
- Epinions / Ciao

Rating sparsity setting을 포함하는 것이 좋다.

- Full interaction setting
- 80% interaction setting
- 50% interaction setting
- 20% interaction setting
- Cold user / cold item split

### 6.2 Baselines

#### Basic rating prediction

- MF
- PMF
- NeuMF
- LightGCN
- NGCF

#### Review-based rating prediction

- DeepCoNN
- NARRE
- MPCN
- TransNet
- ANR
- A3NCF
- Multi-factor collaborative prediction models

#### Review-aware graph / contrastive models

- RGCL
- DGCLR
- Review-aware contrastive alignment models
- Hierarchical contrastive review recommendation models

#### Aspect-aware models

- ANR
- Aspect-aware multi-criteria recommendation
- Aspect performance-aware hypergraph recommendation

#### Disentangled / multi-intent models

- DGCF
- DCCF
- IDCL-style intent disentanglement
- Multi-intention contrastive recommendation

#### Debiasing / causal models

- Sentiment debiasing recommendation
- Counterfactual recommendation baselines
- Popularity debiasing models

### 6.3 Main Metrics

Rating prediction에서는 다음 metrics를 사용한다.

\[
RMSE=\sqrt{\frac{1}{N}\sum(y-\hat{y})^2}
\]

\[
MAE=\frac{1}{N}\sum|y-\hat{y}|
\]

Ranking task로 확장하는 경우 다음 metrics를 사용할 수 있다.

- NDCG@K
- Recall@K
- Hit Ratio@K
- MRR@K

### 6.4 Ablation Study

반드시 포함해야 할 ablation은 다음과 같다.

1. **without \(z^R\)**: residual review-specific factor 제거
2. **without \(z^S\)**: shared rating-aligned factor 제거
3. **without inconsistency gate**: fixed combination 사용
4. **hard sentiment mismatch vs proposed soft inconsistency**: sentiment mismatch label과 soft alignment score 비교
5. **without IB**: bottleneck 제거
6. **without CLUB / MI separation**: residual separation 제거
7. **without multi-intent**: single review representation 사용
8. **removing inconsistent interactions**: 기존 noise removal 전략과 비교
9. **using only consistent interactions**: consistent-only training의 한계 검증
10. **random removal with rating distribution matched**: inconsistent removal 효과가 rating distribution bias 때문인지 검증
11. **without user-normalized rating deviation**: user rating scale normalization 효과 검증
12. **without item/popularity bias features**: item distribution bias 고려 효과 검증

### 6.5 Analysis

#### Consistent vs inconsistent test split

Test set을 soft inconsistency score 기준으로 나눈다.

\[
p_{inc}(ui)>\gamma
\]

이면 high-inconsistency group, 아니면 low-inconsistency group으로 정의한다.

두 그룹에서 RMSE/MAE를 따로 보고한다.

#### Rating distribution controlled evaluation

Consistent/inconsistent split이 rating distribution에 의해 왜곡되지 않도록 rating bin별 stratified evaluation을 수행한다.

- 1-star
- 2-star
- 3-star
- 4-star
- 5-star

각 bin 안에서 consistent/inconsistent 성능을 비교한다.

#### Gate analysis

\(g_{ui}\)가 실제 high-inconsistency interaction에서 낮아지는지 확인한다.

분석 대상은 다음과 같다.

- High rating + negative review
- Low rating + positive review
- Mixed review
- Neutral review + extreme rating

#### Representation visualization

- \(z^S\), \(z^R\)의 t-SNE / UMAP
- \(z^S\)가 rating level별로 정렬되는지 확인
- \(z^R\)이 review semantic cluster 또는 latent intent cluster를 형성하는지 확인

#### Case study

예시 interaction은 다음과 같다.

- 5점 + “battery life is terrible”
- 2점 + “design is beautiful”
- 4점 + “shipping was slow but product works great”

각 case에서 다음 값을 보고한다.

- \(g_{ui}\)
- Top intent prototype
- \(f_Y(z^Y)\)
- \(f_S(z^S)\)
- \(f_R(z^R)\)

#### User-specific rating scale normalization

User normalization 전후로 \(p_{inc}\) 분포가 어떻게 바뀌는지 분석한다.

---

## 7. Novelty and Difference from Existing Work

### 7.1 Difference from simple review-rating alignment

기존 contrastive review recommendation은 user, item, review representation을 같은 latent space에 align하는 경우가 많다.

RADIANT는 review 전체를 rating에 맞추지 않는다.

\[
z^S \leftrightarrow z^Y
\]

만 align하고,

\[
z^R \not\leftrightarrow z^Y
\]

는 분리한다.

### 7.2 Difference from sentiment bias removal

Sentiment debiasing은 review sentiment가 rating prediction을 왜곡한다고 보고 sentiment effect를 제거하려는 경우가 많다.

RADIANT는 sentiment mismatch를 핵심 label로 쓰지 않는다. 대신 rating utility와 review-expressed factor의 latent alignment를 추정한다.

즉 inconsistency는 다음이 아니다.

\[
\text{sentiment}(x)\neq\text{sentiment}(y)
\]

대신 다음에 가깝다.

\[
z^Y \not\approx z^S
\]

### 7.3 Difference from user likes/dislikes differentiation

User likes/dislikes differentiation은 positive/negative preference를 분리한다.

RADIANT는 좋아함/싫어함을 나누는 것이 아니라 다음 두 정보를 분리한다.

- Rating-aligned shared factor
- Review-specific residual factor

즉 dislike modeling은 direction polarity 중심이고, RADIANT는 utility-alignment 중심이다.

### 7.4 Difference from aspect-aware recommendation

Aspect-aware recommendation은 명시적 aspect 또는 aspect attention을 사용한다.

RADIANT는 명시 aspect extraction을 필수로 두지 않는다. 기본형은 latent prototype/intent를 사용한다.

이는 aspect label이 없거나 extractor가 부정확한 상황에서 더 안전하다.

### 7.5 Difference from multimodal shared/specific disentanglement

Multimodal recommendation에서도 shared/specific representation disentanglement가 있다.

하지만 RADIANT의 특수성은 modality 간 관계가 아니라 same experience에서 생성된 rating scalar와 review text의 비대칭 관계를 모델링한다는 점이다.

Rating은 utility anchor이고, review는 partial expression이다.

### 7.6 Core novelty

핵심 novelty는 다음 문장으로 정리할 수 있다.

> We reinterpret rating-review inconsistency not as noise or sentiment contradiction, but as a utility-expression mismatch where reviews may reveal low-weight or residual aspects not dominant in the rating decision.

---

## 8. Expected Contribution

논문 contribution은 다음 4개로 정리할 수 있다.

1. **Rating-review inconsistency를 utility-direction mismatch로 재정의**  
   기존 sentiment mismatch 또는 noise 관점과 달리, rating utility factor와 review-expressed factor의 soft mismatch로 정의한다.

2. **Rating-anchored shared/residual disentanglement 제안**  
   Review representation을 rating-aligned shared factor \(z^S\)와 review-specific residual factor \(z^R\)로 분리한다.

3. **Inconsistency-aware gating mechanism 제안**  
   Inconsistent review를 제거하지 않고, residual low-weight aspect signal로 약하게 활용한다.

4. **Rating distribution bias를 고려한 controlled evaluation protocol 제안**  
   Consistent/inconsistent 비교가 rating distribution bias에 의해 왜곡되지 않도록 stratified, user-normalized, random-removal-matched evaluation을 수행한다.

---

## 9. Potential Weakness and Fix

### Weakness 1. \(z^S\)와 \(z^R\) 분리가 명확하지 않을 수 있음

두 representation이 같은 정보를 담거나 한쪽이 collapse할 수 있다.

#### Fix

- Reconstruction loss 추가
- Orthogonality loss 추가
- CLUB 기반 MI upper bound minimization
- \(z^S\), \(z^R\) 각각의 prediction contribution 분석
- Stop-gradient를 활용한 stable alignment

### Weakness 2. Gate collapse

Gate가 항상 1 또는 항상 0이 될 수 있다.

#### Fix

- Entropy regularization
- Prior constraint \(|\bar{g}-\rho|\)
- Warm-up 동안 fixed gate 사용
- \(f_Y\), \(f_S\), \(f_R\) prediction contribution scale normalization
- Residual coefficient \(\eta\)를 작게 시작해서 annealing

### Weakness 3. Review sentiment extractor에 의존하면 오류 전파 가능

만약 sentiment mismatch를 label로 쓰면 sentiment classifier 오류가 모델 학습을 망칠 수 있다.

#### Fix

- Sentiment는 hard label로 쓰지 않는다.
- Sentiment는 soft auxiliary feature로만 사용한다.
- 핵심 inconsistency는 \(sim(z^Y,z^S)\), \(d_{ui}=|y-\tilde{y}^X|\), user-normalized deviation으로 정의한다.

### Weakness 4. Multi-intent \(K\) 선택이 어려움

너무 작으면 factor가 섞이고, 너무 크면 sparse/collapse가 발생한다.

#### Fix

- \(K\in\{2,4,8,16\}\) sensitivity analysis
- Prototype usage entropy regularization
- Unused prototype penalty
- Dirichlet prior 또는 diversity loss 사용

Prototype diversity는 다음과 같이 줄 수 있다.

\[
\mathcal{L}_{proto}=\sum_{k\neq k'}\cos(p_k,p_{k'})^2
\]

### Weakness 5. 모델이 너무 복잡할 수 있음

Graph encoder, review encoder, disentanglement, IB, CLUB, gate, counterfactual까지 모두 넣으면 구현과 학습이 불안정해질 수 있다.

#### Fix: Minimal version과 Full version 분리

---

## 10. Minimal Version vs Full Version

### 10.1 Minimal Version

논문 1차 구현에 적합한 구조는 다음과 같다.

1. LightGCN rating encoder
2. Cached review embedding
3. Prototype attention multi-intent
4. \(z^S,z^R\) projection
5. Weighted InfoNCE
6. Orthogonality loss
7. Inconsistency-aware gate
8. MSE prediction loss

Minimal objective는 다음과 같다.

\[
\mathcal{L}=\mathcal{L}_{pred}+\lambda_{align}\mathcal{L}_{align}+\lambda_{orth}\mathcal{L}_{orth}+\lambda_{rec}\mathcal{L}_{rec}+\lambda_{gate}\mathcal{L}_{gate}
\]

### 10.2 Full Version

확장 연구용 구조는 다음과 같다.

1. Transformer/BERT review encoder
2. User/item review document encoder
3. Variational Information Bottleneck
4. CLUB MI upper bound minimization
5. Counterfactual debiasing
6. Causal gate regularization
7. Rating-distribution controlled training

---

## 11. 최종 방법론 문장

본 연구의 최종 방법론 방향은 다음 문장으로 요약할 수 있다.

> We propose a rating-anchored disentangled review learning framework that treats ratings as overall utility anchors and reviews as partial textual expressions. Instead of forcing all review information to align with rating prediction, the model separates review representations into rating-aligned shared factors and review-specific residual factors. A soft inconsistency-aware gate then determines how much each factor contributes to rating prediction, allowing inconsistent reviews to serve as low-weight aspect signals rather than noise.

한글로는 다음과 같이 정리할 수 있다.

> 본 연구는 rating을 전체 utility anchor로, review를 부분적 textual expression으로 보고, review representation을 rating-aligned shared factor와 review-specific residual factor로 분리한다. 또한 soft inconsistency-aware gate를 통해 inconsistent review를 제거하지 않고 low-weight aspect signal로 활용한다.
