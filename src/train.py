# %% [markdown]
# ## 1. 라이브러리 불러오기
# 
# 학습에 필요한 라이브러리를 불러옵니다. 표준 라이브러리(csv, json, os)로 데이터를 읽고, `scikit-learn` 으로 TF-IDF 벡터화와 로지스틱 회귀를, `joblib` 으로 학습한 모델을 저장합니다.
# 

# %%
import csv
import json
import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# 예측 대상 14개 클래스 (Macro-F1 계산에 사용)
ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]

# %% [markdown]
# ## 2. 데이터 불러오기
# 
# `train.jsonl` 은 한 줄에 샘플 하나(JSON)이고, `train_labels.csv` 는 `id,action` 두 컬럼입니다. `id` 를 기준으로 두 파일을 연결하여 입력 `X`(= `current_prompt`)와 정답 `y`(= `action`)를 만듭니다.
# 

# %%
DATA_DIR = "../data"

# train.jsonl: 한 줄 = 샘플 하나
samples = [json.loads(line)
           for line in open(os.path.join(DATA_DIR, "train.jsonl"), encoding="utf-8")
           if line.strip()]

# train_labels.csv: id -> action 매핑
labels = {row["id"]: row["action"]
          for row in csv.DictReader(open(os.path.join(DATA_DIR, "train_labels.csv"), encoding="utf-8"))}

# 입력 X = current_prompt, 정답 y = action
X = [s["current_prompt"] for s in samples]
y = [labels[s["id"]] for s in samples]

print("samples:", len(X), "| classes:", len(set(y)))

# %% [markdown]
# ## 3. 학습 / 검증 데이터 분할
# 
# 모델이 학습에 쓰지 않은 데이터에서도 잘 맞히는지(일반화)를 확인하려면, 학습 데이터의 일부를 떼어 **검증 세트**로 사용합니다. `stratify=y` 로 학습/검증의 클래스 비율을 동일하게 유지하고, `random_state` 로 재현성을 확보합니다.
# 

# %%
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42,
)
print("train:", len(X_train), "| val:", len(X_val))

# %% [markdown]
# ## 4. 모델 정의와 학습 (TF-IDF + 로지스틱 회귀)
# 
# 두 단계를 하나의 **파이프라인**으로 묶습니다.
# 
# 1. **TF-IDF**: 텍스트를 단어/2-gram의 중요도 기반 숫자 벡터로 변환합니다. (ngram_range=(1,2), min_df=2, max_features=80000, sublinear_tf=True)
# 2. **로지스틱 회귀**: 그 벡터로 14개 클래스를 분류합니다. 클래스 불균형을 보정하기 위해 `class_weight='balanced'` 를, 규제 강도는 `C=2.0` 으로 둡니다.
# 

# %%
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=80_000,
        sublinear_tf=True, lowercase=True,
    )),
    ("clf", LogisticRegression(
        max_iter=500, class_weight="balanced", C=2.0,
    )),
])

pipe.fit(X_train, y_train)
print("학습 완료")

# %% [markdown]
# ## 5. 검증 — Macro-F1
# 
# 검증 세트로 성능을 확인합니다. **Macro-F1** 은 14개 클래스 각각의 F1 점수를 동일한 가중치로 평균하므로, 자주 등장하지 않는 클래스도 똑같이 중요하게 평가합니다.
# 

# %%
val_pred = pipe.predict(X_val)
macro_f1 = f1_score(y_val, val_pred, labels=ALL_CLASSES, average="macro", zero_division=0)
print(f"Validation Macro-F1: {macro_f1:.4f}")

# %% [markdown]
# ## 6. 전체 데이터로 재학습 & 모델 저장
# 
# 검증으로 성능을 확인했으니, 이제 **전체 학습 데이터**로 다시 학습하여 최종 모델을 만듭니다(일반적으로 데이터를 더 많이 쓸수록 유리). 학습한 파이프라인을 `../submit/model/tfidf_logreg.pkl` 로 저장합니다.
# 
# 이 파일을 추론용 `script.py`, `requirements.txt` 와 함께 `baseline_submit.zip` 으로 묶어 제출합니다.
# 

# %%
# 전체 학습 데이터로 재학습
pipe.fit(X, y)

# 저장 (추론용 script.py가 ../submit/model/tfidf_logreg.pkl 을 불러옵니다)
os.makedirs("../submit/model", exist_ok=True)
joblib.dump(pipe, "../submit/model/tfidf_logreg.pkl", compress=3)
print("저장 완료: ../submit/model/tfidf_logreg.pkl")