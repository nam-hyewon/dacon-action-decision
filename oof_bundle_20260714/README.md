# DACON action-decision — OOF 번들 (2026-07-14)

모든 로짓은 **session-grouped 5-fold** (`folds_seed42.csv`) 의 out-of-fold 예측이다.
train 70,000행 전체가 정확히 한 번씩 held-out 으로 예측됐다.

## 파일

| 파일 | 설명 |
|---|---|
| `folds_seed42.csv` | **★ fold 정의. 이거 없이는 아래 로짓이 무의미하다.** id → fold(0~4) |
| `index.csv` | 모든 OOF 의 **공통 행 순서**: id, fold, true_id, true_action (70,000행) |
| `oof/<name>/logits.npy` | (70000, 14) float32. **행 순서 = index.csv 와 동일** |
| `oof/<name>/meta.json` | 단일 macro-F1, fold별 점수, **score_type** |
| `recipes/*.json` | 현재 제출 레시피 (가중치/온도/class_bias) |

## 사용법

```python
import numpy as np, pandas as pd
from sklearn.metrics import f1_score

idx = pd.read_csv("index.csv")
y, fold = idx["true_id"].to_numpy(), idx["fold"].to_numpy()

large = np.load("oof/klue_roberta_large_eda_state_oof/logits.npy")
tab   = np.load("oof/tabular_histgb_oof/logits.npy")
# ... 가중합 -> argmax -> f1_score(y, pred, average="macro")
```

## ⚠️ 결합할 때 반드시 주의

**score_type 이 두 종류다:**
- **트랜스포머** (large/klue/diet/rob8/deb/soup) → `raw_logit` (스케일 큼)
- **tabular** (histgb/lightgbm/catboost/*_view) → `log_prob` (exp 합 = 1)

**섞을 때 스케일을 맞추지 않으면 망가진다.** 현 제출 레시피는 **raw 그대로** 가중합한다
(z-score 안 함). 온도(temperature)로 스케일을 조정한다 — `recipes/` 참고.

## ⚠️ 정직한 검증 필수 (LOFO)

가중치·bias 를 **전체 OOF 에서 튜닝하고 전체 OOF 에서 평가하면 반드시 과적합된다.**
fold F 를 빼고 나머지 4개에서 튜닝 → F 에서 평가 (LOFO) 를 5번 반복할 것.
이 방법으로 검증하면 아래 값들이 나온다.

## 현재 성능

| OOF | macro-F1 | score_type |
|---|---|---|
| `ens_large_tab_diet_lgb_oof` | 0.7622 | raw_logit |
| `kluebert_mid5_lr15_seed42_7_soup_oof` | 0.7350 | raw_logit |
| `klue_roberta_large_eda_state_oof` | 0.7334 | raw_logit |
| `kluebert_mid5_lr15_oof` | 0.7284 | raw_logit |
| `deberta_base_eda_state4_seed43_oof` | 0.7259 | raw_logit |
| `kluebert_diet_oof` | 0.7250 | raw_logit |
| `roberta_eda_state8_lr15_oof` | 0.7197 | raw_logit |
| `roberta_eda_state4_oof` | 0.7139 | raw_logit |
| `tabular_histgb_oof` | 0.4590 | log_prob(합=1) |
| `tabular_lightgbm_oof` | 0.4497 | log_prob(합=1) |
| `tabular_catboost_oof` | 0.4262 | log_prob(합=1) |
| `tabular_histgb_surface_view_oof` | 0.3392 | log_prob(합=1) |
| `tabular_histgb_no_prompt_surface_oof` | 0.3228 | log_prob(합=1) |
| `tabular_histgb_history_view_oof` | 0.2279 | log_prob(합=1) |

**현재 best 제출물** = `ens_large_tab_diet_lgb_oof` (레시피: `large_tab_diet_lgb_gated40`)
- OOF 70k: **0.7622**  /  정직 fold1~4: 0.7614  /  fold0(튜닝셋): 0.7649
- **실제 LB: 0.7547** (로컬 → LB 갭 ≈ **-0.0075**. 로컬 개선이 이 갭보다 작으면 LB 에서 사라진다)
- 런타임 6분47초 (제한 10분). base 트랜스포머 1개(diet) + large 40% gated.

## ❌ 이미 시도해서 실패한 것 (반복하지 말 것 — 전부 LOFO 정직 검증함)

| 시도 | 결과 |
|---|---|
| OOF 로 가중치·bias 재튜닝 (14k → 56k) | +0.0007 (노이즈) |
| 멤버 교체/추가 (soup, deb 넣기) | -0.0007 ~ -0.0017 |
| 그리디 최적 조합 | +0.0005 (5 fold 중 3개만 승) |
| 게이트 비율 40% → 100% | +0.0001 (40% 가 이미 최적) |
| 클래스 가중 손실 / focal | 약한 클래스가 **오히려 하락** |
| 스태킹 / 라우터 / 4-way 스페셜리스트 | ≤ 0 |
| 탐색 힌트 텍스트 모드 4종 | -0.014 ~ -0.082 |

## 진짜 병목 = 데이터 한계 (모델 문제 아님)

정보수집 4클래스(read/grep/list/glob) = **데이터의 41%**, OOF F1 평균 **0.594**.

- **강한 LLM 에게 애매한 20문항을 직접 풀렸더니 9/20 (45%)** — 우리 작은 모델(10/20, 50%)보다 못 맞혔다.
- 전체 컨텍스트를 줘도 **45% 그대로**, 확신도만 0.66 → 0.93 으로 올랐다 = 환원불가 라벨 노이즈.
- deberta 4-way 전용 모델도 0.5697. 어떤 방법으로도 0.59 를 못 넘는다.

**⇒ 점수를 올리려면 앙상블이 아니라 base 모델 자체가 좋아져야 한다.**
