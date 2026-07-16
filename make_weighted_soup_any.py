import os
import json
import shutil
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer

seed_str = os.environ.get("SOUP_SEEDS", "42,43,44")
SEEDS = [int(x) for x in seed_str.split(",") if x.strip()]

BASE = Path("submit/model")
OUT_DIR = BASE / ("deberta_base_eda_state4_fold0_weighted_soup_" + "_".join(map(str, SEEDS)))

IN_DIRS = [BASE / f"deberta_base_eda_state4_fold0_seed{s}" for s in SEEDS]

f1s = []
for d in IN_DIRS:
    info_path = d / "val_info.json"
    if not info_path.exists():
        raise FileNotFoundError(info_path)
    info = json.load(open(info_path, encoding="utf-8"))
    f1s.append(float(info["best_val_macro_f1"]))

total = sum(f1s)
weights = [f / total for f in f1s]

print("Weighted soup inputs:")
for seed, d, f1, w in zip(SEEDS, IN_DIRS, f1s, weights):
    print(f"seed={seed} | f1={f1:.6f} | weight={w:.6f} | dir={d}")
    if not d.exists():
        raise FileNotFoundError(d)
    if not (d / "model.safetensors").exists() and not (d / "pytorch_model.bin").exists():
        raise FileNotFoundError(f"no model weights in {d}")

first = AutoModelForSequenceClassification.from_pretrained(IN_DIRS[0])
avg_state = {k: v.detach().cpu().float() * weights[0] for k, v in first.state_dict().items()}
del first

for d, w in zip(IN_DIRS[1:], weights[1:]):
    print("Adding", d, "weight", w)
    model = AutoModelForSequenceClassification.from_pretrained(d)
    sd = model.state_dict()
    for k in avg_state:
        avg_state[k] += sd[k].detach().cpu().float() * w
    del model

soup = AutoModelForSequenceClassification.from_pretrained(IN_DIRS[0])
soup.load_state_dict(avg_state, strict=True)

if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)

soup.save_pretrained(OUT_DIR, safe_serialization=True)

tokenizer = AutoTokenizer.from_pretrained(IN_DIRS[0])
tokenizer.save_pretrained(OUT_DIR)

meta = {
    "type": "weighted_soup",
    "seeds": SEEDS,
    "input_dirs": [str(d) for d in IN_DIRS],
    "f1s": f1s,
    "weights": weights,
    "weight_rule": "normalized best_val_macro_f1",
}
json.dump(meta, open(OUT_DIR / "soup_info.json", "w", encoding="utf-8"), indent=2)

print("DONE:", OUT_DIR)
