#!/usr/bin/env python3

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

ROOT = Path(
    "repro_bundle/large_tab_diet_rob8_repro"
)

PRED_CSV = (
    ROOT
    / "submit/model/kluebert_diet_fold0/val_predictions.csv"
)

MODEL_PATHS = {
    "large": (
        ROOT
        / "submit/model/klue_roberta_large_eda_state_fold0/val_logits.npy"
    ),
    "tab": (
        ROOT
        / "submit/model/tabular_histgb_fold0/val_logits.npy"
    ),
    "diet": (
        ROOT
        / "submit/model/kluebert_diet_fold0/val_logits.npy"
    ),
    "rob8": (
        ROOT
        / "submit/model/roberta_eda_state8_lr15_fold0/val_logits.npy"
    ),
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
    0.0,
    -0.3000011444091797,
    0.39999818801879883,
    0.6499979496002197,
    -0.3000011444091797,
    0.0,
    0.0,
    -0.000001430511474609375,
    -0.2500011920928955,
    -0.35000109672546387,
    -0.2500011920928955,
    -0.10000133514404297,
    0.4999980926513672,
    1.3499972820281982,
], dtype=np.float64)


pred_df = pd.read_csv(PRED_CSV)

lex_df = pd.read_csv(
    "lexical_expert_fold0_predictions.csv"
)

assert np.array_equal(
    pred_df["id"].astype(str).values,
    lex_df["id"].astype(str).values,
)

y = pred_df["true_id"].astype(int).values

lex_pred = (
    lex_df["lex_pred_id"]
    .astype(int)
    .values
)

lex_score = np.load(
    "lexical_expert_fold0_decision.npy"
).astype(np.float64)


raw = {
    name: np.load(path).astype(np.float64)
    for name, path in MODEL_PATHS.items()
}

score = sum(
    WEIGHTS[name]
    * (raw[name] / TEMPS[name])
    for name in WEIGHTS
)

score += BIAS

base_pred = score.argmax(axis=1)


base_f1 = f1_score(
    y,
    base_pred,
    labels=range(len(CLASSES)),
    average="macro",
    zero_division=0,
)

lex_f1 = f1_score(
    y,
    lex_pred,
    labels=range(len(CLASSES)),
    average="macro",
    zero_division=0,
)

base_correct = base_pred == y
lex_correct = lex_pred == y

benefit = (
    (~base_correct)
    & lex_correct
)

harm = (
    base_correct
    & (~lex_correct)
)

agree = base_pred == lex_pred
disagree = ~agree


oracle_pred = base_pred.copy()
oracle_pred[benefit] = lex_pred[benefit]

oracle_f1 = f1_score(
    y,
    oracle_pred,
    labels=range(len(CLASSES)),
    average="macro",
    zero_division=0,
)


print("=" * 90)
print("EXACT REPRO BASE + LEXICAL")
print("=" * 90)

print("BASE F1:", base_f1)
print("BASE ACC:", accuracy_score(y, base_pred))

print()
print("LEXICAL F1:", lex_f1)
print("LEXICAL ACC:", accuracy_score(y, lex_pred))

print()
print("COMPLEMENTARITY")
print(
    "base correct / lex correct:",
    int((base_correct & lex_correct).sum()),
)
print(
    "base correct / lex wrong:",
    int((base_correct & ~lex_correct).sum()),
)
print(
    "base wrong / lex correct:",
    int(benefit.sum()),
)
print(
    "both wrong:",
    int((~base_correct & ~lex_correct).sum()),
)
print(
    "lex recovery among base errors:",
    float(
        lex_correct[~base_correct].mean()
    ),
)

print()
print("AGREEMENT")
print("agree n:", int(agree.sum()))
print(
    "agree acc:",
    accuracy_score(
        y[agree],
        base_pred[agree],
    ),
)
print("disagree n:", int(disagree.sum()))
print(
    "base acc disagreement:",
    accuracy_score(
        y[disagree],
        base_pred[disagree],
    ),
)
print(
    "lex acc disagreement:",
    accuracy_score(
        y[disagree],
        lex_pred[disagree],
    ),
)

print()
print("ORACLE UNION")
print("recoverable:", int(benefit.sum()))
print("harm possible:", int(harm.sum()))
print("oracle F1:", oracle_f1)
print("oracle gain:", oracle_f1 - base_f1)


# Save exact-base quantities for router.
np.save(
    "exact_repro_base_score.npy",
    score,
)

np.save(
    "exact_repro_base_pred.npy",
    base_pred,
)

print()
print("Saved:")
print(" - exact_repro_base_score.npy")
print(" - exact_repro_base_pred.npy")
