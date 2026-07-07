# Experiments

마지막 업데이트: 2026-07-07

이 문서는 지금까지 저장된 validation 산출물 기준 실험 정리다. 점수는 모두 local validation Macro-F1이며, DACON leaderboard 점수와 다를 수 있다.

## 현재 결론

- 단일 RoBERTa 계열에서는 `roberta_eda_state4`가 0.7194로 가장 안정적이었다.
- `eda_state_mid`는 원본 `eda_state`보다 단일 점수는 조금 낮지만, specialist booster와 결합했을 때 이득이 컸다.
- class weight를 넣은 `mid5_lr15_weighted`는 단일 0.7162, specialist 적용 후 0.7351까지 올라갔다.
- 새로 확인한 `kluebert_mid5_lr15`는 단일 Macro-F1 0.7247로, 지금까지 확인된 단일 모델 중 가장 강하다.
- 현재 validation 최고는 `ensemble_deberta_full5_router3_best_roberta_eda4_zscore + specialist`의 0.7396이다.
- 거의 같은 급의 후보는 `ensemble_mid5_weighted_base_deberta_full5_router3_fixed_zscore + specialist_fine`의 0.7386이다. 이 후보는 현재 repo 안 산출물이 더 일관적으로 남아 있어 재현/분석용 기준으로 좋다.

## 주요 실험 결과

| 실험 | 구성 | Macro-F1 | Accuracy | 메모 |
|---|---:|---:|---:|---|
| `roberta` baseline | `klue/roberta-base`, 초기 text, 3ep | 0.6550 | - | 초기 기준선 |
| `roberta_full4` | `TEXT_MODE=full`, 4ep, lr 2e-5 | 0.6396 | - | full context는 오히려 약함 |
| `roberta_eda_state4` | `TEXT_MODE=eda_state`, 4ep, lr 2e-5 | 0.7194 | - | 단일 RoBERTa 최고권 |
| `roberta_eda_state4 + specialist` | router/prompt specialist | 0.7318 | 0.7406 | +0.0125 |
| `roberta_eda_state4 + transition_prior` | transition prior | 0.7207 | 0.7312 | +0.0014로 효과 작음 |
| `roberta_mid4_lr15` | `eda_state_mid`, 4ep, lr 1.5e-5 | 0.7081 | - | mid 단독은 낮음 |
| `roberta_mid4_lr15 + specialist` | specialist | 0.7266 | 0.7329 | +0.0185 |
| `roberta_mid5_lr15` | `eda_state_mid`, 5ep, lr 1.5e-5 | 0.7187 | - | 5ep가 mid4보다 개선 |
| `roberta_mid5_lr15 + specialist_fine` | fine specialist | 0.7342 | 0.7364 | +0.0155 |
| `roberta_mid5_lr2` | `eda_state_mid`, 5ep, lr 2e-5 | 0.7068 | - | lr 2e-5는 불리 |
| `roberta_state5_lr2` | `eda_state`, 5ep, lr 2e-5 | 0.6453 | - | 해당 run은 실패권 |
| `koelectra_mid5_lr15` | KoELECTRA, mid, 5ep, lr 1.5e-5 | 0.7010 | - | RoBERTa보다 낮음 |
| `koelectra_mid5_lr15 + specialist` | specialist | 0.7198 | 0.7306 | +0.0188 |
| `koelectra_mid5_lr2` | KoELECTRA, mid, 5ep, lr 2e-5 | 0.7125 | - | KoELECTRA는 lr 2e-5가 더 나음 |
| `kluebert_mid5_lr15` | KLUE-BERT, mid, 5ep, lr 1.5e-5 | 0.7247 | 0.7371 | 단일 모델 최고. 사용자 제공 로그 기준 |
| `mid5_lr15_weighted` | RoBERTa mid, class weight 추정 | 0.7162 | - | 단일은 mid5와 비슷 |
| `mid5_lr15_weighted + specialist_fine` | specialist | 0.7351 | 0.7386 | 단일 모델 후처리 최고 |
| `deberta_full5 + router3 fixed zscore` | DeBERTa 계열 내부 앙상블 | 0.7161 | 0.7275 | router 비중 0.05 |
| `deberta_eda3 + roberta_eda4` | logit ensemble | 0.7204 | 0.7341 | specialist 전 |
| `deberta_eda3 + roberta_eda4 + specialist` | specialist | 0.7322 | 0.7423 | +0.0118 |
| `deberta_full5_router3_best + roberta_eda4 zscore` | logit ensemble | 0.7285 | 0.7406 | specialist 전 강함 |
| `deberta_full5_router3_best + roberta_eda4 zscore + specialist` | specialist | 0.7396 | 0.7472 | 현재 val 최고 |
| `mid5_weighted_base + deberta_full5_router3_fixed zscore` | logit ensemble | 0.7246 | 0.7366 | specialist 전 |
| `mid5_weighted_base + deberta_full5_router3_fixed zscore + specialist_fine` | specialist | 0.7386 | 0.7446 | 현재 재현 기준 후보 |
| `mid5_weighted_specialist + deberta_full5_router3_fixed zscore` | specialist 모델을 먼저 앙상블 | 0.7371 | 0.7439 | 후처리 순서상 손해 가능 |
| `mid5_weighted_specialist + deberta_full5_router3_fixed zscore + specialist_fine` | ensemble 후 specialist | 0.7386 | 0.7445 | base 앙상블 후 specialist와 거의 동일 |
| `mid5_weighted_specialist + koelectra_specialist zscore` | RoBERTa + KoELECTRA | 0.7351 | 0.7386 | 다양성 이득 거의 없음 |

## 관찰

### Text mode

- `full`은 정보량은 많지만 truncation/noise가 커서 성능이 낮았다.
- `eda_state`는 핵심 상태 피처를 앞쪽에 배치한 효과가 컸고, 4ep 기준 0.7194로 강했다.
- `eda_state_mid`는 압축된 형태라 단일 모델은 살짝 약하지만, specialist가 보정할 여지가 커졌다.
- 현재 `eda_state_mid` token 길이는 평균 445, p95 516, `MAX_LEN=512` 초과 약 5.9%라 truncation은 관리 가능한 수준이다.

### Epoch / LR

- `roberta_mid5_lr15`: 0.7187
- `roberta_mid5_lr2`: 0.7068
- 같은 mid 구성에서는 lr 1.5e-5가 2e-5보다 좋았다.
- `roberta_mid5_lr2` 로그상 epoch별 Macro-F1은 0.6382 -> 0.6790 -> 0.6945 -> 0.7039 -> 0.7068로 계속 오르긴 했지만, 최종점 자체는 낮았다.
- `koelectra_mid5_lr15`: 0.7010
- `koelectra_mid5_lr2`: 0.7125
- KoELECTRA는 반대로 lr 2e-5가 더 좋았다. 다만 KLUE-BERT 1.5e-5가 0.7247로 KoELECTRA보다 확실히 강했다.

### Backbone 비교

- `kluebert_mid5_lr15`: 0.7247, accuracy 0.7371
- `roberta_eda_state4`: 0.7194
- `roberta_mid5_lr15`: 0.7187
- `koelectra_mid5_lr2`: 0.7125
- `koelectra_mid5_lr15`: 0.7010

KLUE-BERT는 단일 모델 기준으로 가장 강하다. 특히 `apply_patch` 0.9361, `run_tests` 0.7882, `lint_or_typecheck` 0.6189, `plan_task` 0.5936이 좋아 전체 Macro-F1을 끌어올렸다. 반면 `web_search`는 KoELECTRA lr2가 0.5880으로 KLUE-BERT 0.5774보다 조금 낫고, `ask_user`도 거의 비슷하다. 따라서 KLUE-BERT는 단일/ensemble 주력 후보, KoELECTRA는 특정 minority class 다양성 후보로 보는 게 맞다.

### Class weight / loss

- `mid5_lr15_weighted`는 단일 0.7162라 `roberta_mid5_lr15` 0.7187보다 낮다.
- 하지만 specialist 후 0.7351로 올라가며, 후처리와 궁합이 좋았다.

#### Focal loss (2026-07-07 결과)

- 실제로 돌린 건 `focal2`(γ=2) 하나뿐이다. `focal1`/`focal3`/`focal2_w`는 미실행.
- `mid5_lr15_focal2` 단일 Macro-F1 **0.7009**로 CE 대비 하락 (같은 lr 1.5e-5 CE 범위 0.697~0.716보다 우위 없음). 에폭별 0.62→0.67→0.67→0.70→0.70, best는 epoch 5라 학습 자체는 정상.
- **핵심 실패**: focal이 노린 희귀/어려운 클래스가 오히려 악화됐다. `ask_user` −0.047, `plan_task` −0.045, `list_directory` −0.045. 개선된 건 이미 쉬운 `write_file`(+0.015)과 소수 `web_search`(+0.014)뿐이라 방향이 어긋난다.
- **반전**: 단독은 약해도 CE와 다른 실수를 해서 앙상블/soup 재료로는 유용. `roberta_mid6_lr15 + focal2` 쌍이 0.7191로 관측됨. focal2 모델은 삭제하지 말고 앙상블 재료로 보관.
- **결론**: γ만 바꾸는 focal1/focal3에 시간 쓰지 말 것. 희귀 클래스는 focal보다 class weight(sqrt_inv)나 specialist 후처리가 더 직접적. `focal2_w`(focal+class weight)만 희귀 클래스 관점에서 1회 시도 가치 있음(미검증).

### Specialist booster

- 대부분의 상위 실험에서 specialist가 +0.01~+0.02 Macro-F1을 만든다.
- 특히 `web_search`, `lint_or_typecheck`, `run_tests`, `list_directory`, `ask_user`, `plan_task` 개선 폭이 크다.
- 단, ensemble 이후 specialist에서는 `read_file`이 소폭 하락하는 경우가 있어 target/gamma 선택을 더 보수적으로 다듬을 여지가 있다.

### Transition prior

- `roberta_eda_state4`: 0.7194 -> 0.7207, +0.0014
- `roberta_mid5_lr15`: 0.7187 -> 0.7190, +0.0003
- specialist 이후 transition prior도 +0.00004 수준이라 현재 우선순위는 낮다.

### Ensemble

- z-score logit ensemble은 안정적으로 효과가 있었다.
- specialist를 적용한 모델끼리 섞는 것보다, base logits를 먼저 앙상블하고 마지막에 specialist를 거는 쪽이 약간 더 낫거나 비슷했다.
- KoELECTRA는 단일 다양성은 있지만 RoBERTa/DeBERTa 상위 조합에 비해 ensemble 이득이 작았다.

## 현재 약한 클래스

`ensemble_mid5_weighted_base_deberta_full5_router3_fixed_zscore + specialist_fine` 기준 낮은 F1:

| class | F1 | precision | recall | support |
|---|---:|---:|---:|---:|
| `list_directory` | 0.4598 | 0.3923 | 0.5554 | 866 |
| `read_file` | 0.5586 | 0.5422 | 0.5759 | 1851 |
| `web_search` | 0.5752 | 0.6471 | 0.5176 | 255 |
| `grep_search` | 0.6159 | 0.6596 | 0.5777 | 1982 |
| `lint_or_typecheck` | 0.6439 | 0.7273 | 0.5777 | 457 |
| `ask_user` | 0.6450 | 0.6654 | 0.6259 | 540 |
| `glob_pattern` | 0.6674 | 0.7521 | 0.5998 | 1057 |
| `plan_task` | 0.6778 | 0.6417 | 0.7183 | 536 |

핵심 혼동은 파일 탐색 계열(`read_file`, `grep_search`, `list_directory`, `glob_pattern`)과 의사결정 계열(`ask_user`, `plan_task`, `respond_only`)이다. Macro-F1 대회라 support가 작은 `web_search`, `lint_or_typecheck`, `ask_user`, `plan_task` 보정이 전체 점수에 크게 먹힌다.

## 다음 핵심 실험 계획

### 1순위: `kluebert_mid5_lr15` 후처리 및 앙상블 투입

목표: 단일 0.7247 모델을 기존 best 파이프라인에 얹어 0.74대 후반 가능성을 확인한다.

바로 할 일:

- `kluebert_mid5_lr15`에 `specialist_fine` 적용
- `kluebert_mid5_lr15` logits와 기존 강한 후보 z-score ensemble
- ensemble 후 specialist를 마지막에 적용

우선 조합:

- `kluebert_mid5_lr15 + mid5_lr15_weighted`
- `kluebert_mid5_lr15 + roberta_eda_state4`
- `kluebert_mid5_lr15 + ensemble_deberta_full5_router3_best_roberta_eda4_zscore`
- `kluebert_mid5_lr15 + ensemble_mid5_weighted_base_deberta_full5_router3_fixed_zscore`

판정 기준:

- specialist 전 ensemble이 0.730 이상이면 계속 유지한다.
- specialist 후 0.740을 넘으면 제출 후보 1순위로 승격한다.
- `web_search`, `list_directory`, `read_file` 중 하나가 크게 무너지면 target/gamma를 보수적으로 조정한다.

### 2순위: Focal / class weight loss 실험 (대부분 종결)

`focal2` 결과로 focal 방향은 사실상 접었다 (위 "Focal loss" 절 참고). 단독 0.7009로 CE보다 낮고, 노렸던 희귀 클래스가 오히려 악화됐다.

남은 것:

- `focal1`/`focal3`(γ만 변경)은 진행하지 않는다. 방향이 안 맞는다.
- `focal2_w`(focal+class weight, sqrt_inv)만 희귀 클래스 recall 관점에서 1회 시도 여지. 단, 우선순위는 kluebert 후처리/앙상블보다 낮다.
- `focal2` 모델은 앙상블/soup 재료로 보관 (`mid6_lr15 + focal2` = 0.7191).

### 3순위: 상위 ensemble의 specialist target 재튜닝

현재 최고권에서 specialist는 이득이 크지만 `read_file` 하락이 보인다.

실험 방향:

- `read_file`을 specialist target에서 제외하거나 gamma를 낮춘다.
- target을 `web_search`, `list_directory`, `lint_or_typecheck`, `run_tests`, `ask_user`, `plan_task` 중심으로 제한한다.
- 현재 best 두 후보에 모두 적용한다.

대상:

- `ensemble_deberta_full5_router3_best_roberta_eda4_zscore`
- `ensemble_mid5_weighted_base_deberta_full5_router3_fixed_zscore`

### 4순위: seed ensemble

모델 구조를 더 늘리기보다 같은 강한 설정의 seed variance를 활용한다.

우선 설정:

- `TEXT_MODE=eda_state_mid`
- `MODEL_NAME=klue/roberta-base`
- `EPOCHS=5`
- `LR=1.5e-5`
- class weight on/off 각각 1개 이상
- `SEED=7`, `SEED=123`

판정:

- seed 단일이 0.715 이상이면 z-score ensemble에 추가한다.
- 단일이 낮아도 기존 모델과 예측 차이가 크고 약한 클래스가 보완되면 보관한다.

### 5순위: `eda_state` 재현성 확인

`roberta_eda_state4`가 강했지만 `roberta_state5_lr2`는 낮았다. 설정 차이인지 run 문제인지 확인이 필요하다.

실행 후보:

```bash
TEXT_MODE=eda_state SAVE_NAME=roberta_state4_lr15 EPOCHS=4 LR=1.5e-5 python src/train_roberta.py
TEXT_MODE=eda_state SAVE_NAME=roberta_state4_lr2 EPOCHS=4 LR=2e-5 python src/train_roberta.py
```

목표:

- `eda_state4` 0.719 근처가 재현되면 `mid`와 별도 축으로 ensemble 가치가 있다.
- 재현되지 않으면 과거 산출물만 후보로 유지하고 새 학습 우선순위를 낮춘다.

## 제출 후보

현재 validation 기준 우선 제출 후보:

1. `ensemble_deberta_full5_router3_best_roberta_eda4_zscore + specialist`
   - Macro-F1 0.7396
   - Accuracy 0.7472
   - local val 최고

2. `ensemble_mid5_weighted_base_deberta_full5_router3_fixed_zscore + specialist_fine`
   - Macro-F1 0.7386
   - Accuracy 0.7446
   - 현재 산출물/분석 경로가 명확한 재현 기준 후보

3. `mid5_lr15_weighted + specialist_fine`
   - Macro-F1 0.7351
   - Accuracy 0.7386
   - 단일 모델 기반이라 제출 패키징 리스크가 낮음

4. `kluebert_mid5_lr15`
   - Macro-F1 0.7247
   - Accuracy 0.7371
   - specialist/ensemble 전 단일 모델 후보. 후처리 후 제출 후보로 올라갈 가능성이 큼

## 바로 다음 액션

가장 먼저 할 실험은 `kluebert_mid5_lr15`의 specialist 적용과 z-score ensemble이다. 단일 점수가 이미 0.7247이라 기존 best 조합에 새로운 강한 축을 추가할 가능성이 가장 높다.

그 다음은 `focal1`, `focal2`, `focal2_w`다. loss 실험은 minority class를 직접 겨냥할 수 있지만, 새로 나온 KLUE-BERT 결과가 더 즉시 활용 가능한 강한 후보라 우선순위를 한 칸 내린다.
