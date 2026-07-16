from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score


ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]

label2id = {c: i for i, c in enumerate(ALL_CLASSES)}
id2label = {i: c for i, c in enumerate(ALL_CLASSES)}

ROOT = Path(".")
BASE = (
    ROOT
    / "team_4model_ensemble_07626"
    / "ensemble_07626"
    / "fold0_team"
)

PRED_PATH = (
    BASE
    / "all5_bias_meta_crossfit"
    / "crossfit_final_pred_idx.npy"
)

REF_PATH = (
    BASE
    / "deberta_base_eda_state4_seed43_oof"
    / "val_predictions.csv"
)

OUT_DIR = BASE / "sota_confusion_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


pred = np.load(PRED_PATH).astype(int)

df = pd.read_csv(REF_PATH)
df["id"] = df["id"].astype(str)

label_col = None
for c in ["label", "true_action", "true_label", "action", "target"]:
    if c in df.columns:
        label_col = c
        break

if label_col is None:
    raise ValueError(df.columns.tolist())

y_true = np.array(
    [label2id[x] for x in df[label_col].astype(str)],
    dtype=int,
)

if len(pred) != len(y_true):
    raise ValueError((len(pred), len(y_true)))

print("n =", len(y_true))
print(
    "macro_f1 =",
    f1_score(y_true, pred, average="macro"),
)

cm = confusion_matrix(
    y_true,
    pred,
    labels=np.arange(len(ALL_CLASSES)),
)

# ------------------------------------------------------------
# Directed confusion pairs
# ------------------------------------------------------------

rows = []

for true_idx, true_name in enumerate(ALL_CLASSES):
    support = cm[true_idx].sum()

    for pred_idx, pred_name in enumerate(ALL_CLASSES):
        if true_idx == pred_idx:
            continue

        count = int(cm[true_idx, pred_idx])

        rows.append({
            "true": true_name,
            "pred": pred_name,
            "count": count,
            "true_support": int(support),
            "error_rate_within_true": count / max(support, 1),
        })

directed = (
    pd.DataFrame(rows)
    .sort_values(
        ["count", "error_rate_within_true"],
        ascending=False,
    )
)

directed.to_csv(
    OUT_DIR / "directed_confusions.csv",
    index=False,
)

print()
print("=" * 100)
print("TOP 50 DIRECTED CONFUSIONS")
print("=" * 100)

print(
    directed.head(50).to_string(index=False)
)


# ------------------------------------------------------------
# Symmetric pair confusion
# ------------------------------------------------------------

pair_rows = []

for i in range(len(ALL_CLASSES)):
    for j in range(i + 1, len(ALL_CLASSES)):
        i_to_j = int(cm[i, j])
        j_to_i = int(cm[j, i])
        total = i_to_j + j_to_i

        pair_rows.append({
            "class_a": ALL_CLASSES[i],
            "class_b": ALL_CLASSES[j],
            "a_to_b": i_to_j,
            "b_to_a": j_to_i,
            "total_confusion": total,
        })

pairs = (
    pd.DataFrame(pair_rows)
    .sort_values(
        "total_confusion",
        ascending=False,
    )
)

pairs.to_csv(
    OUT_DIR / "symmetric_confusion_pairs.csv",
    index=False,
)

print()
print("=" * 100)
print("TOP 40 SYMMETRIC CONFUSION PAIRS")
print("=" * 100)

print(
    pairs.head(40).to_string(index=False)
)


# ------------------------------------------------------------
# Per-class error concentration
# ------------------------------------------------------------

class_rows = []

for i, name in enumerate(ALL_CLASSES):
    support = int(cm[i].sum())
    correct = int(cm[i, i])
    errors = support - correct

    wrong_counts = cm[i].copy()
    wrong_counts[i] = 0

    top_wrong_idx = np.argsort(-wrong_counts)[:5]

    row = {
        "class": name,
        "support": support,
        "correct": correct,
        "errors": errors,
        "error_rate": errors / max(support, 1),
    }

    for rank, j in enumerate(top_wrong_idx, start=1):
        row[f"confusion_{rank}"] = ALL_CLASSES[j]
        row[f"confusion_{rank}_count"] = int(wrong_counts[j])

    class_rows.append(row)

class_df = (
    pd.DataFrame(class_rows)
    .sort_values("errors", ascending=False)
)

class_df.to_csv(
    OUT_DIR / "per_class_error_concentration.csv",
    index=False,
)

print()
print("=" * 100)
print("PER-CLASS ERROR CONCENTRATION")
print("=" * 100)

print(
    class_df.to_string(index=False)
)


# ------------------------------------------------------------
# Candidate specialist clusters
# ------------------------------------------------------------

clusters = {
    "file4": [
        "read_file",
        "grep_search",
        "list_directory",
        "glob_pattern",
    ],
    "exec3": [
        "run_bash",
        "run_tests",
        "lint_or_typecheck",
    ],
    "decision3": [
        "ask_user",
        "plan_task",
        "web_search",
    ],
    "edit3": [
        "edit_file",
        "write_file",
        "apply_patch",
    ],
}

cluster_rows = []

for cluster_name, members in clusters.items():
    idx = np.array([label2id[x] for x in members])

    internal_errors = 0
    total_errors = 0
    support = 0

    for i in idx:
        support += int(cm[i].sum())

        for j in range(len(ALL_CLASSES)):
            if i == j:
                continue

            count = int(cm[i, j])
            total_errors += count

            if j in idx:
                internal_errors += count

    cluster_rows.append({
        "cluster": cluster_name,
        "members": "|".join(members),
        "support": support,
        "total_errors_from_cluster": total_errors,
        "internal_confusion_errors": internal_errors,
        "internal_error_share": (
            internal_errors / max(total_errors, 1)
        ),
    })

cluster_df = (
    pd.DataFrame(cluster_rows)
    .sort_values(
        "internal_confusion_errors",
        ascending=False,
    )
)

cluster_df.to_csv(
    OUT_DIR / "candidate_cluster_stats.csv",
    index=False,
)

print()
print("=" * 100)
print("CANDIDATE SPECIALIST CLUSTERS")
print("=" * 100)

print(
    cluster_df.to_string(index=False)
)

print()
print("Saved:", OUT_DIR)
