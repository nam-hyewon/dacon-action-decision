# dacon-action-decision

AI Agent Action Decision Prediction Challenge

Team: **YOUNGCRAZYCREW**

---

## 🚀 제출 방법 (현재 모델: klue + Specialist Booster)

현재 제출 파이프라인은 **klue/roberta-base 분류기 + Specialist Booster** 조합입니다.
로컬 검증 Macro-F1: base `0.7194` → booster 적용 `0.7318`.

추론 흐름 (`submit/script.py` 안에서 전부 처리):

1. klue 모델(`model/roberta_eda_state4`, text_mode=`eda_state`)로 **fp32 logits** 계산
2. 저장된 Specialist Booster(`specialists.joblib`) 적용 — router 텍스트 기반 one-vs-rest TF-IDF 보정
3. 튜닝된 per-class gamma로 보정 후 argmax → `submission.csv`

> ⚠️ **fp32 필수.** booster gamma가 fp32 검증 logit에 맞춰 튜닝됨. fp16으로 돌리면 booster 이득이 깎임.
> ⚠️ 텍스트 로직을 바꾸면 klue(`eda_state`)와 specialist(`router`) **둘 다** 함께 맞춰야 함 (독립 feature 파이프라인).

### 0. 사전 준비 (한 번만)

`data/`에 원본 4개 파일(`train.jsonl`, `train_labels.csv`, `test.jsonl`, `sample_submission.csv`)이 있어야 합니다.

```bash
pip install -r submit/requirements.txt
```

> 이 저장소의 `.venv`는 Windows용이라 Linux에선 사용 불가. 시스템 `python`을 그대로 사용하세요 (activate 하지 말 것).

### 1. Specialist Booster 학습·저장 (한 번만 — 이미 완료됨)

```bash
cd src
python build_specialists.py
```

- 결과: `submit/model/roberta_eda_state4/specialist_boosters/specialists.joblib`
- 끝에 `boost Macro-F1: 0.7318 → OK` 가 뜨면 정상.
- klue 모델이나 텍스트 로직을 바꾸지 않으면 다시 돌릴 필요 없음.

### 2. 로컬 추론 테스트

```bash
cd submit
cp -r ../data data          # submit/data 없으면 복사
python script.py
cat output/submission.csv    # 5줄 예측 나오면 OK
```

### 3. submit.zip 만들기

`zip` 명령이 없는 환경이면 python으로 생성:

```bash
cd submit
python - <<'PY'
import zipfile
files = [
    "script.py",
    "requirements.txt",
    "model/roberta_eda_state4/config.json",
    "model/roberta_eda_state4/labels.json",
    "model/roberta_eda_state4/model.safetensors",
    "model/roberta_eda_state4/tokenizer.json",
    "model/roberta_eda_state4/tokenizer_config.json",
    "model/roberta_eda_state4/specialist_boosters/specialists.joblib",
]
with zipfile.ZipFile("submit.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(f)
print("done")
PY
```

`zip` 명령이 되는 환경이면: `zip -r submit.zip script.py requirements.txt model/roberta_eda_state4`
(단, `val_logits.npy` 등 불필요 파일까지 포함되어 커지므로 위 python 방식 권장.)

zip 내부 구조:

```text
submit.zip
├── script.py
├── requirements.txt
└── model/roberta_eda_state4/
    ├── config.json
    ├── labels.json
    ├── model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── specialist_boosters/specialists.joblib
```

⚠️ `data/`, `output/`, `src/`, 다른 model 폴더(deberta 등)는 **넣지 말 것.**

### 4. DACON 업로드

1. 대회 페이지 → **제출하기** (코드 제출형)
2. `submit/submit.zip` 업로드
3. 서버가 자동으로: `requirements.txt` 설치 → 숨겨진 test.jsonl을 `./data/`에 넣음 → `python script.py` 실행 → `output/submission.csv` 채점

**요약**: 코드를 안 바꿨으면 → `submit/submit.zip`을 그대로 업로드.
바꿨으면 → `1 → 2 → 3` 재실행 후 업로드.

---

## Project Structure

```text
dacon/
├── data/
│   ├── train.jsonl
│   ├── train_labels.csv
│   ├── test.jsonl
│   └── sample_submission.csv
├── src/
│   ├── train.py                    # TF-IDF baseline (legacy)
│   ├── train_roberta.py            # klue/roberta-base 학습 (eda_state 등 text_mode)
│   ├── tune_specialist_boosters.py # booster gamma 튜닝 (검증용)
│   └── build_specialists.py        # 튜닝된 booster를 추론용으로 학습·저장
├── submit/
│   ├── model/
│   │   └── roberta_eda_state4/      # 제출 모델 (klue) + specialist_boosters/specialists.joblib
│   ├── script.py                   # 제출 추론 (klue + booster)
│   ├── script_roberta.py           # klue 단일 모델 추론 (booster 없음, 참고용)
│   └── requirements.txt
└── README.md
```

---

## Environment

Python 3.11

Install dependencies if needed:

```bash
pip install -r submit/requirements.txt
```

---

## 1. Download Dataset

Download the official dataset from DACON and place the files under:

```text
data/
├── train.jsonl
├── train_labels.csv
├── test.jsonl
└── sample_submission.csv
```

---

## 2. Train the Baseline Model (legacy — TF-IDF)

> 아래는 초기 TF-IDF + LogReg 베이스라인 흐름입니다. 현재 제출 모델은 위 "🚀 제출 방법" 섹션을 참고하세요.

Move to the `src` directory and run:

```bash
cd src
python train.py
```

This will:

- Train the TF-IDF + Logistic Regression baseline
- Evaluate on the validation split
- Save the trained model to

```text
submit/model/tfidf_logreg.pkl
```

---

## 3. Test Inference Locally

For local testing only, copy the sample data into the submit folder:

```bash
cd ..
cp -r data submit/data
```

Run inference:

```bash
cd submit
python script.py
```

If successful, the following file will be generated:

```text
submit/output/submission.csv
```

---

## 4. Create Submission File

Inside the `submit` directory:

```bash
zip -r submit.zip model script.py requirements.txt
```

The final structure should be:

```text
submit.zip
├── model/
├── script.py
└── requirements.txt
```

⚠️ Do **NOT** include:

- data/
- output/
- src/

---

## Notes

- The official evaluation server automatically provides the hidden test dataset.
- `test.jsonl` included in the released dataset contains only 5 sample instances for local testing.
- The evaluation metric is **Macro-F1** over 14 action classes.