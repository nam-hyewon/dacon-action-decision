#!/usr/bin/env python3

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]
C2I = {x: i for i, x in enumerate(CLASSES)}

ROOT = Path("large + tab + diet + rob8 (0.76468)")
PRED_CSV = ROOT / "kluebert_diet_fold0/val_predictions.csv"

MODEL_PATHS = {
    "large": ROOT / "klue_roberta_large_eda_state_fold0/val_logits.npy",
    "tab": ROOT / "tabular_histgb_fold0/val_logits.npy",
    "diet": ROOT / "kluebert_diet_fold0/val_logits.npy",
    "rob8": ROOT / "roberta_eda_state8_lr15_fold0/val_logits.npy",
}

WEIGHTS = {
    "large": 0.24114308710234722,
    "tab": 0.4438507372276068,
    "diet": 0.21685297669057865,
    "rob8": 0.09815319897946724,
}

TEMPS = {
    "large": 1.0,
    "tab": 1.0,
    "diet": 1.0,
    "rob8": 0.85,
}

BIAS = np.array([
    0.0, -0.3000011444091797, 0.39999818801879883,
    0.6499979496002197, -0.3000011444091797, 0.0, 0.0,
    -0.000001430511474609375, -0.2500011920928955,
    -0.35000109672546387, -0.2500011920928955,
    -0.10000133514404297, 0.4999980926513672,
    1.3499972820281982,
], dtype=np.float64)


def parse_sid_step(sid):
    sid = str(sid)

    m = re.match(r"^(.*)-step_(\d+)$", sid)

    if not m:
        raise ValueError(f"bad id: {sid}")

    return m.group(1), int(m.group(2))


samples = []

with open("data/train.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if line:
            samples.append(json.loads(line))


label_df = pd.read_csv("data/train_labels.csv")

label_map = dict(zip(
    label_df["id"].astype(str),
    label_df["action"].astype(str),
))

ids = np.array([
    str(x["id"])
    for x in samples
])

labels = np.array([
    C2I[label_map[sid]]
    for sid in ids
])

sessions = []
steps = []

for sid in ids:
    session, step = parse_sid_step(sid)

    sessions.append(session)
    steps.append(step)

sessions = np.array(sessions)
steps = np.array(steps)


pred_df = pd.read_csv(PRED_CSV)

val_ids = pred_df["id"].astype(str).values
val_set = set(val_ids)

id_to_idx = {
    sid: i
    for i, sid in enumerate(ids)
}

val_idx = np.array([
    id_to_idx[sid]
    for sid in val_ids
])

ref_mask = np.array([
    sid not in val_set
    for sid in ids
])

y = labels[val_idx]


# base
raw = {
    k: np.load(p).astype(np.float64)
    for k, p in MODEL_PATHS.items()
}

score = sum(
    WEIGHTS[k] * (raw[k] / TEMPS[k])
    for k in WEIGHTS
)

score += BIAS

base_pred = score.argmax(axis=1)

BASE_F1 = f1_score(
    y,
    base_pred,
    labels=range(len(CLASSES)),
    average="macro",
    zero_division=0,
)

print("BASE F1:", BASE_F1)
print()


# session overlap
ref_sessions = set(sessions[ref_mask])
val_sessions = sessions[val_idx]

overlap = np.array([
    s in ref_sessions
    for s in val_sessions
])

print("===== SESSION OVERLAP =====")
print("reference rows:", int(ref_mask.sum()))
print("validation rows:", len(val_idx))
print("reference unique sessions:", len(ref_sessions))
print("validation unique sessions:", len(set(val_sessions)))
print("val rows with session in reference:", int(overlap.sum()))
print("overlap rate:", float(overlap.mean()))
print()


# Build session -> observed reference step/action sequence
session_ref = defaultdict(list)

for session, step, label in zip(
    sessions[ref_mask],
    steps[ref_mask],
    labels[ref_mask],
):
    session_ref[session].append(
        (int(step), int(label))
    )

for session in session_ref:
    session_ref[session].sort()


print("===== REFERENCE STEPS PER OVERLAP SESSION =====")

counts = []

for session in val_sessions:
    counts.append(
        len(session_ref.get(session, []))
    )

print(pd.Series(counts).describe())
print()


# ============================================================
# Simple session-local predictors
# ============================================================

methods = {
    "nearest_step": [],
    "previous_step": [],
    "next_step": [],
    "session_majority": [],
}

valid_by_method = {
    k: []
    for k in methods
}


for global_idx in val_idx:
    session = sessions[global_idx]
    step = int(steps[global_idx])

    ref = session_ref.get(session, [])

    # nearest observed step
    if ref:
        nearest = min(
            ref,
            key=lambda x: abs(x[0] - step),
        )

        methods["nearest_step"].append(nearest[1])
        valid_by_method["nearest_step"].append(True)

        cnt = Counter(label for _, label in ref)
        methods["session_majority"].append(
            cnt.most_common(1)[0][0]
        )
        valid_by_method["session_majority"].append(True)
    else:
        methods["nearest_step"].append(-1)
        valid_by_method["nearest_step"].append(False)

        methods["session_majority"].append(-1)
        valid_by_method["session_majority"].append(False)

    prevs = [
        x
        for x in ref
        if x[0] < step
    ]

    if prevs:
        methods["previous_step"].append(prevs[-1][1])
        valid_by_method["previous_step"].append(True)
    else:
        methods["previous_step"].append(-1)
        valid_by_method["previous_step"].append(False)

    nexts = [
        x
        for x in ref
        if x[0] > step
    ]

    if nexts:
        methods["next_step"].append(nexts[0][1])
        valid_by_method["next_step"].append(True)
    else:
        methods["next_step"].append(-1)
        valid_by_method["next_step"].append(False)


print("===== SIMPLE SESSION PREDICTORS =====")

for name in methods:
    pred = np.array(methods[name])
    valid = np.array(valid_by_method[name])

    if valid.sum() == 0:
        continue

    print()
    print(name)
    print("coverage:", int(valid.sum()), float(valid.mean()))
    print(
        "session predictor acc:",
        accuracy_score(
            y[valid],
            pred[valid],
        ),
    )
    print(
        "base acc covered:",
        accuracy_score(
            y[valid],
            base_pred[valid],
        ),
    )


# ============================================================
# Session prediction as gate/override
# ============================================================

rows = []

for name in methods:
    sess_pred = np.array(methods[name])
    valid = np.array(valid_by_method[name])

    for margin_thr in [
        0.1,
        0.25,
        0.5,
        1.0,
        2.0,
        5.0,
        999.0,
    ]:
        sorted_score = np.sort(
            score,
            axis=1,
        )

        margin = (
            sorted_score[:, -1]
            - sorted_score[:, -2]
        )

        use = (
            valid
            & (margin <= margin_thr)
        )

        pred = base_pred.copy()
        pred[use] = sess_pred[use]

        f1 = f1_score(
            y,
            pred,
            labels=range(len(CLASSES)),
            average="macro",
            zero_division=0,
        )

        rows.append({
            "method": name,
            "margin_thr": margin_thr,
            "macro_f1": f1,
            "gain": f1 - BASE_F1,
            "accuracy": accuracy_score(y, pred),
            "use_n": int(use.sum()),
            "changed_n": int(
                (
                    pred[use]
                    != base_pred[use]
                ).sum()
            ),
        })


result = pd.DataFrame(rows).sort_values(
    ["macro_f1", "accuracy"],
    ascending=False,
)

print()
print("===== SESSION OVERRIDE TOP =====")
print(result.head(50).to_string(index=False))

result.to_csv(
    "session_overlap_override_results.csv",
    index=False,
)

print()
print("Saved session_overlap_override_results.csv")
