# dacon-action-decision

AI Agent Action Decision Prediction Challenge

Team: **YOUNGCRAZYCREW**

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
│   └── train.py
├── submit/
│   ├── model/
│   ├── script.py
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

## 2. Train the Baseline Model

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