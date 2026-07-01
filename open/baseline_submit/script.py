"""[Baseline] TF-IDF + 로지스틱 회귀 — 추론 (script.py)

AI 코딩 에이전트의 **다음 행동(action)** 을 14개 클래스 중 하나로 예측하는 베이스라인의
'추론' 코드입니다. 학습 노트북이 저장한 모델(`./model/tfidf_logreg.pkl`)을 불러와
테스트 데이터(`./data/test.jsonl`)에 대한 예측을 수행하고, 제출 파일
(`./output/submission.csv`)을 생성합니다.

[ 코드 제출 방식 ]
이 대회는 결과 CSV가 아니라 '코드'를 제출합니다. 아래 구조의 zip을 제출하면,
평가 서버가 zip을 풀고 `script.py` 를 그대로 실행하여 채점합니다.

    baseline_submit.zip
    ├── model/
    │   └── tfidf_logreg.pkl     # 학습 노트북이 저장한 모델
    ├── script.py                # 이 파일 (서버가 실행)
    └── requirements.txt         # 필요한 라이브러리

서버는 `./data/test.jsonl` 와 `./data/sample_submission.csv` 를 제공하며,
이 코드는 `./output/submission.csv` 를 만들어 내야 합니다.
"""
# =======================
# 1. 라이브러리 불러오기
# =======================
# csv/json/os: 표준 라이브러리로 데이터를 읽고 결과를 씁니다.
# joblib: 학습 때 저장한 모델(.pkl)을 불러옵니다.
import csv
import json
import os

import joblib


# =======================
# 2. 입력 전처리 유틸 (학습 때와 동일하게 current_prompt만 사용)
# =======================
# 추론 입력은 반드시 '학습 때와 똑같은 방식'으로 만들어야 합니다.
# 학습에서 입력으로 current_prompt 문자열만 사용했으므로, 여기서도 동일하게 뽑습니다.

REQUIRED_KEYS = ("id", "session_meta", "history", "current_prompt")


def load_jsonl(path):
    """평가 데이터(jsonl) 로드. 한 줄당 샘플 하나."""
    samples = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no} JSON 파싱 실패: {e}")
            samples.append(obj)
    return samples


def validate_samples(samples):
    """필수 키 존재 여부 검증 (학습 데이터와 동일 스키마)."""
    n_bad = 0
    for s in samples:
        for k in REQUIRED_KEYS:
            if k not in s:
                n_bad += 1
                break
    if n_bad:
        print(f" 경고: 필수 키 누락 샘플 {n_bad}건 (빈 텍스트로 처리)")
    return n_bad


def extract_text(sample):
    """모델 입력 텍스트 추출 — 학습 때와 동일하게 current_prompt만 사용.

    TF-IDF 파이프라인이 토큰화/벡터화를 모두 포함하므로 여기서는
    문자열만 뽑는다. current_prompt가 없거나 문자열이 아니면 빈 문자열.
    """
    text = sample.get("current_prompt", "")
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return text


def build_features(samples):
    """샘플 리스트 → (ids, 모델 입력 텍스트 리스트)."""
    ids = []
    texts = []
    for s in samples:
        ids.append(s.get("id", ""))
        texts.append(extract_text(s))
    n_empty = sum(1 for t in texts if not t.strip())
    if n_empty:
        print(f" 경고: current_prompt가 비어있는 샘플 {n_empty}건")
    return ids, texts


# =======================
# 3. 제출 파일 생성 유틸 (sample_submission의 id 순서/형식 기준)
# =======================
# 제출 파일은 sample_submission.csv 와 '같은 id 순서, 같은 컬럼(id, action)'이어야
# 합니다. 모델 예측을 id 기준으로 sample_submission에 채워 넣습니다.

def load_sample_submission(path):
    """sample_submission.csv 로드 — 제출 파일의 id 순서/컬럼 기준."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if fieldnames is None or fieldnames[:2] != ["id", "action"]:
        raise ValueError(
            f"sample_submission 컬럼이 (id, action)이 아님: {fieldnames}")
    return fieldnames, rows


def merge_predictions(sub_rows, ids, preds):
    """sample_submission의 id 순서에 맞춰 예측값 병합.

    예측에 없는 id는 sample_submission의 기존 값(placeholder)을 유지한다.
    """
    pred_map = dict(zip(ids, preds))
    n_missing = 0
    for row in sub_rows:
        p = pred_map.get(row["id"])
        if p is None:
            n_missing += 1
        else:
            row["action"] = p
    if n_missing:
        print(f" 경고: 예측이 없어 placeholder를 유지한 id {n_missing}건")
    return sub_rows


def save_submission(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# =======================
# 4. 추론 실행 (모델 로드 → 예측 → 제출 파일 생성)
# =======================

def main():
    # ---- 경로 변수 (필요에 따라 수정) ----
    TEST_DIR = "./data"            # test.jsonl, sample_submission.csv 위치
    MODEL_DIR = "./model"          # tfidf_logreg.pkl 위치
    OUT_DIR = "./output"
    TEST_PATH = os.path.join(TEST_DIR, "test.jsonl")
    SAMPLE_SUB_PATH = os.path.join(TEST_DIR, "sample_submission.csv")
    MODEL_PATH = os.path.join(MODEL_DIR, "tfidf_logreg.pkl")
    OUT_PATH = os.path.join(OUT_DIR, "submission.csv")

    # ---- 모델 로드 ----
    # 학습 노트북이 저장한 파이프라인(TF-IDF + 로지스틱 회귀)을 그대로 불러옵니다.
    print("Load model...")
    model = joblib.load(MODEL_PATH)
    classes = list(getattr(model, "classes_", []))
    print(f" OK. classes={len(classes)}")

    # ---- 테스트 데이터 로드 ----
    print("Load test data...")
    samples = load_jsonl(TEST_PATH)
    validate_samples(samples)
    print(f" samples={len(samples)}")

    # ---- 전처리 (학습과 동일: current_prompt 추출) ----
    print("Build features...")
    ids, texts = build_features(samples)
    print(f" texts={len(texts)}")

    # ---- 예측 ----
    # 파이프라인이 TF-IDF 변환과 분류를 함께 수행하므로 텍스트를 그대로 넣습니다.
    print("Inference model...")
    preds = model.predict(texts) if texts else []
    preds = [str(p) for p in preds]
    print(f" preds={len(preds)}")

    # ---- sample_submission 기반 결과 생성 (action 컬럼에 예측 클래스 채움) ----
    print("Build submission...")
    fieldnames, sub_rows = load_sample_submission(SAMPLE_SUB_PATH)
    sub_rows = merge_predictions(sub_rows, ids, preds)
    save_submission(OUT_PATH, fieldnames, sub_rows)
    print(f"Saved: {OUT_PATH} (rows={len(sub_rows)})")


if __name__ == "__main__":
    main()
