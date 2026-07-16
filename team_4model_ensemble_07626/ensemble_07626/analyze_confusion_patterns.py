import re
import json
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.metrics import classification_report, confusion_matrix

ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]

df = pd.read_csv("ensemble_greedy4_weightgrid_row_zscore/class_bias/val_predictions_with_bias.csv")

true_col = "true_action"
pred_col = "bias_pred_action"

print("=== CLASSIFICATION REPORT ===")
print(classification_report(
    df[true_col],
    df[pred_col],
    labels=ALL_CLASSES,
    digits=4,
    zero_division=0,
))

bad = df[df[true_col] != df[pred_col]].copy()

pairs = (
    bad.groupby([true_col, pred_col])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

print("\n=== TOP CONFUSION PAIRS ===")
print(pairs.head(30).to_string(index=False))

pairs.head(50).to_csv("top_confusion_pairs.csv", index=False)

# pair별 샘플 저장
out_lines = []
for rank, row in pairs.head(20).iterrows():
    t = row[true_col]
    p = row[pred_col]
    sub = bad[(bad[true_col] == t) & (bad[pred_col] == p)].copy()

    out_lines.append("\n" + "=" * 120)
    out_lines.append(f"TRUE={t}  PRED={p}  count={len(sub)}")
    out_lines.append("=" * 120)

    cols = [
        "id", "current_prompt", "last_action", "prev_action",
        "turn_index", "git_dirty", "last_ci_status", "main_lang",
        "bias_confidence",
    ]
    cols = [c for c in cols if c in sub.columns]

    for _, r in sub.head(20).iterrows():
        out_lines.append("-" * 100)
        for c in cols:
            out_lines.append(f"{c}: {r.get(c)}")

Path = __import__("pathlib").Path
Path("top_confusion_examples.txt").write_text("\n".join(out_lines), encoding="utf-8")

print("\nSaved:")
print(" - top_confusion_pairs.csv")
print(" - top_confusion_examples.txt")
