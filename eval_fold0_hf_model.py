import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, accuracy_score, classification_report
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.train_roberta_fold0 import load_data

ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]
label2id = {c: i for i, c in enumerate(ALL_CLASSES)}
id2label = {i: c for i, c in enumerate(ALL_CLASSES)}

MODEL_DIR = Path(os.environ["MODEL_DIR"])
MAX_LEN = int(os.environ.get("MAX_LEN", "512"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))

print("MODEL_DIR:", MODEL_DIR)
print("MAX_LEN:", MAX_LEN)
print("BATCH_SIZE:", BATCH_SIZE)

texts, labels, rows = load_data()

ref_val_path = Path("team_4model_ensemble_07626/ensemble_07626/fold0_team/deberta_fold0/val_predictions.csv")
ref = pd.read_csv(ref_val_path)
val_ids = ref["id"].astype(str).tolist()

id_to_idx = {str(row["id"]): i for i, row in enumerate(rows)}
idxs = [id_to_idx[x] for x in val_ids]

val_texts = [texts[i] for i in idxs]
val_labels = [labels[i] for i in idxs]

# labels can be either string class names or already numeric class ids
if len(val_labels) > 0 and isinstance(val_labels[0], (int, np.integer)):
    y_true = np.array(val_labels, dtype=int)
    val_label_names = [id2label[int(i)] for i in y_true]
else:
    val_label_names = [str(x) for x in val_labels]
    y_true = np.array([label2id[x] for x in val_label_names], dtype=int)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
print("valid n:", len(val_texts))

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()

all_logits = []

with torch.no_grad():
    for start in tqdm(range(0, len(val_texts), BATCH_SIZE)):
        batch_texts = val_texts[start:start+BATCH_SIZE]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        all_logits.append(out.logits.detach().cpu().numpy())

logits = np.concatenate(all_logits, axis=0)
pred = logits.argmax(axis=1)

macro_f1 = f1_score(y_true, pred, average="macro")
acc = accuracy_score(y_true, pred)

print("macro_f1:", macro_f1)
print("accuracy:", acc)
print(classification_report(y_true, pred, target_names=ALL_CLASSES, digits=4))

np.save(MODEL_DIR / "val_logits.npy", logits)

pred_df = pd.DataFrame({
    "id": val_ids,
    "label": val_label_names,
    "prediction": [id2label[int(i)] for i in pred],
})
pred_df.to_csv(MODEL_DIR / "val_predictions.csv", index=False)

info = {
    "model_dir": str(MODEL_DIR),
    "macro_f1": float(macro_f1),
    "accuracy": float(acc),
    "num_val": int(len(val_texts)),
    "max_len": MAX_LEN,
    "batch_size": BATCH_SIZE,
    "all_classes": ALL_CLASSES,
}

soup_info_path = MODEL_DIR / "soup_info.json"
if soup_info_path.exists():
    info["soup_info"] = json.load(open(soup_info_path, encoding="utf-8"))

json.dump(info, open(MODEL_DIR / "val_info.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

print("saved:")
print(" -", MODEL_DIR / "val_logits.npy")
print(" -", MODEL_DIR / "val_predictions.csv")
print(" -", MODEL_DIR / "val_info.json")
