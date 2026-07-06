# EDA 결과 정리

## 개요

- 데이터: `train.jsonl` 70,000건, 14개 action 클래스
- 목적: 모델 성능 향상을 위한 유용한 피처 탐색
- 평가지표: Macro-F1

---

## Step 1. 클래스 분포

| action | 건수 | 비율 |
|--------|------|------|
| edit_file | 11,171 | 16.0% |
| grep_search | 9,912 | 14.2% |
| read_file | 9,257 | 13.2% |
| glob_pattern | 5,284 | 7.5% |
| respond_only | 5,178 | 7.4% |
| run_bash | 5,068 | 7.2% |
| apply_patch | 4,823 | 6.9% |
| run_tests | 4,561 | 6.5% |
| list_directory | 4,329 | 6.2% |
| ask_user | 2,701 | 3.9% |
| plan_task | 2,679 | 3.8% |
| lint_or_typecheck | 2,283 | 3.3% |
| write_file | 1,481 | 2.1% |
| web_search | 1,273 | 1.8% |

**인사이트**
- 최다 클래스(edit_file) vs 최소 클래스(web_search) 약 **8.8배 차이**
- 하위 클래스(web_search, write_file, lint_or_typecheck)를 잘 맞추는 게 Macro-F1 점수 올리는 핵심
- 베이스라인이 `class_weight='balanced'` 쓰는 이유

---

## Step 2. current_prompt 길이 분포

| 구분 | 값 |
|------|-----|
| 전체 평균 | 61.0자 |
| 전체 최소 | 2자 |
| 전체 최대 | 346자 |

| 그룹 | action | 평균 길이 |
|------|--------|-----------|
| 긴 그룹 | plan_task | 87.6자 |
| | ask_user | 85.8자 |
| | web_search | 85.2자 |
| 짧은 그룹 | run_bash | 39.2자 |
| | lint_or_typecheck | 39.5자 |
| | run_tests | 40.1자 |

**인사이트**
- 짧은 그룹 vs 긴 그룹 약 **2배 차이**로 명확히 구분됨
- → `current_prompt` 길이를 피처로 추가할 가치 있음 ✅

---

## Step 3. history 패턴 분석

### 기본 통계
| 구분 | 값 |
|------|-----|
| history 평균 턴 수 | 6.9 |
| 첫 턴(history 없음) | 9,000건 (12.9%) |

### 첫 턴일 때 action 분포
| action | 건수 |
|--------|------|
| list_directory | 1,818 |
| read_file | 1,488 |
| plan_task | 1,121 |
| grep_search | 1,046 |
| run_bash | 1,011 |

### 직전 action → 다음 action 전이 패턴 (상위 3개)
| 직전 action | 다음 action |
|-------------|-------------|
| edit_file | run_tests(2,444) → edit_file(1,638) → apply_patch(1,076) |
| grep_search | edit_file(2,065) → read_file(1,819) → grep_search(1,726) |
| apply_patch | lint_or_typecheck(757) → respond_only(713) → apply_patch(689) |
| run_tests | edit_file(1,040) → respond_only(774) → grep_search(547) |

**인사이트**
- 직전 action이 다음 action을 **강하게 예측**함
- 세션 초반(첫 턴)은 탐색형 action(list_directory, read_file) 위주
- 실제 코딩 흐름: 탐색 → 수정 → 테스트 → 린트 패턴 확인
- → 직전 action, turn_index 피처 추가 가치 있음 ✅

---

## Step 4. session_meta 피처 분석

### last_ci_status별 상위 5개 action
| CI 상태 | 상위 action |
|---------|-------------|
| passed | edit_file, grep_search, read_file, **respond_only**, glob_pattern |
| failed | edit_file, grep_search, read_file, **apply_patch**, **run_tests** |
| none | grep_search, edit_file, read_file, glob_pattern, run_bash |

### git_dirty별 상위 5개 action
| git_dirty | 상위 action |
|-----------|-------------|
| True | edit_file, grep_search, read_file, **apply_patch**, respond_only |
| False | read_file, grep_search, edit_file, **list_directory**, glob_pattern |

### turn_index 통계
| 구분 | 값 |
|------|-----|
| 평균 | 5.2 |
| 최소 | 1 |
| 최대 | 18 |

### user_tier별 상위 5개 action
| tier | 상위 action |
|------|-------------|
| enterprise | edit_file, read_file, grep_search, run_bash, apply_patch |
| pro | edit_file, grep_search, read_file, glob_pattern, respond_only |
| free | edit_file, grep_search, read_file, respond_only, run_bash |

**인사이트**
- `last_ci_status`: CI failed일 때 apply_patch, run_tests 몰림 → 피처 추가 가치 있음 ✅
- `git_dirty`: True일 때 apply_patch 많음 / False일 때 list_directory 많음 → 피처 추가 가치 있음 ✅
- `turn_index`: 세션 초반/후반 패턴 다름 → 피처 추가 가치 있음 ✅
- `user_tier`: 요금제별 패턴 차이 거의 없음 → 피처 추가 우선순위 낮음 ❌

---

## Step 5. 언어별 분포

| language_pref | 건수 | 1위 | 2위 | 3위 | 4위 | 5위 |
|---------------|------|-----|-----|-----|-----|-----|
| ko | 45,028 | edit_file(16.1%) | grep_search(14.0%) | read_file(13.1%) | respond_only(7.5%) | glob_pattern(7.4%) |
| en | 17,802 | edit_file(15.6%) | grep_search(14.4%) | read_file(13.2%) | glob_pattern(8.1%) | respond_only(7.4%) |
| mixed | 7,170 | edit_file(16.1%) | grep_search(14.1%) | read_file(13.9%) | run_bash(7.8%) | glob_pattern(7.1%) |

**인사이트**
- 1~3위 모두 동일 (edit_file, grep_search, read_file)
- 4~5위 소폭 차이 있으나 비율 차이 미미
- → `language_pref` 피처 추가 우선순위 낮음 ❌

---

## 피처 추가 우선순위 최종 정리

| 순위 | 피처 | 근거 | 예상 효과 |
|------|------|------|-----------|
| 1순위 | 직전 action | 전이 패턴 매우 강함 | 높음 |
| 2순위 | prompt 길이 | 두 그룹 명확히 구분 | 높음 |
| 3순위 | turn_index | 초반/후반 패턴 다름 | 중간 |
| 4순위 | last_ci_status | CI 상태별 action 다름 | 중간 |
| 5순위 | git_dirty | 작업 상태별 action 다름 | 중간 |
| 6순위 | user_tier | 차이 거의 없음 | 낮음 |
| 7순위 | language_pref | 차이 거의 없음 | 낮음 |

---

## 베이스라인 점수

| 구분 | Macro-F1 |
|------|----------|
| 베이스라인 (current_prompt만 사용) | 0.436 |
| 피처 추가 후 (예정) | TBD |