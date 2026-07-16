from pathlib import Path
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIRS = [
    Path("submit/model/deberta_small_eda_kd_from_ce_a01_t2_lr5e6"),
    Path("submit/model/deberta_small_eda_kd_from_ce_a02_t2_lr5e6"),
    Path("submit/model/deberta_small_eda_kd_from_ce_a03_t2_lr5e6"),
]

OUT_DIR = Path("submit/model/deberta_small_eda_kd_soup_a01_a02_a03")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading models...")
models = [AutoModelForSequenceClassification.from_pretrained(p, torch_dtype=torch.float32) for p in MODEL_DIRS]

state_dicts = [m.state_dict() for m in models]
soup_state = {}

for k in state_dicts[0].keys():
    soup_state[k] = sum(sd[k].float() for sd in state_dicts) / len(state_dicts)

print("Building soup model...")
soup_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIRS[0], torch_dtype=torch.float32)
soup_model.load_state_dict(soup_state)

tok = AutoTokenizer.from_pretrained(MODEL_DIRS[0], use_fast=True)

soup_model.save_pretrained(OUT_DIR, safe_serialization=True)
tok.save_pretrained(OUT_DIR)

# copy labels.json
label_path = MODEL_DIRS[0] / "labels.json"
if label_path.exists():
    (OUT_DIR / "labels.json").write_text(label_path.read_text())

print("Saved:", OUT_DIR)
