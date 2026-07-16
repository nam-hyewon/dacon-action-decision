#!/usr/bin/env python3

import importlib.util
import json
from collections import defaultdict
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

OLD_ROOT = Path("../dacon-action-decision")
TEAM_ROOT = OLD_ROOT / "large + tab + diet + rob8 (0.76468)"

FOLD_PATH = Path("data/folds_seed42.csv")

MODEL_PATHS = {
    "large": TEAM_ROOT / "klue_roberta_large_eda_state_fold0/val_logits.npy",
    "tab": TEAM_ROOT / "tabular_histgb_fold0/val_logits.npy",
    "diet": TEAM_ROOT / "kluebert_diet_fold0/val_logits.npy",
    "rob8": TEAM_ROOT / "roberta_eda_state8_lr15_fold0/val_logits.npy",
}

PRED_CSV = (
    TEAM_ROOT
    / "kluebert_diet_fold0"
    / "val_predictions.csv"
)

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


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_all_train():
    candidates = [
        Path("data/train.jsonl"),
        OLD_ROOT / "data/train.jsonl",
    ]

    for p in candidates:
        if p.exists():
            print("train:", p)
            return load_jsonl(p)

    # fallback
    rows = []
    for p in sorted(Path("data").glob("*.jsonl")):
        if "test" not in p.name.lower():
            rows.extend(load_jsonl(p))

    if not rows:
        raise FileNotFoundError("train jsonl not found")

    return rows


def load_team_module():
    p = Path("submit/script_large_tab_diet_rob8_gated40.py")
    spec = importlib.util.spec_from_file_location("team_submit", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def detect_id(row):
    for k in ["id", "sample_id", "uid"]:
        if k in row:
            return str(row[k])
    raise KeyError(f"id field not found: {row.keys()}")


def detect_label(row):
    for k in ["action", "label", "target", "next_action"]:
        if k in row and str(row[k]) in C2I:
            return str(row[k])
    raise KeyError(f"label field not found for {detect_id(row)}")


def detect_fold_columns(df):
    id_candidates = ["id", "sample_id", "uid"]
    fold_candidates = ["fold", "fold_id", "kfold"]

    id_col = next((x for x in id_candidates if x in df.columns), None)
    fold_col = next((x for x in fold_candidates if x in df.columns), None)

    if id_col is None or fold_col is None:
        raise ValueError(
            f"cannot identify fold columns. columns={list(df.columns)}"
        )

    return id_col, fold_col


def make_signature(df, cols):
    return (
        df[list(cols)]
        .fillna("<NA>")
        .astype(str)
        .agg("|||".join, axis=1)
        .values
    )


def build_lookup(keys, labels):
    counts = defaultdict(lambda: np.zeros(len(CLASSES), dtype=np.int64))

    for key, y in zip(keys, labels):
        counts[key][int(y)] += 1

    return counts


def evaluate_lookup(
    name,
    ref_keys,
    ref_y,
    val_keys,
    y,
    base_score,
):
    lookup = build_lookup(ref_keys, ref_y)

    # base statistics
    base_pred = base_score.argmax(1)
    sorted_score = np.sort(base_score, axis=1)
    margin = sorted_score[:, -1] - sorted_score[:, -2]

    rows = []

    supports = [2, 3, 5, 10, 20]
    purities = [0.60, 0.70, 0.80, 0.90, 0.95]
    margin_thrs = [None, 0.25, 0.5, 1.0, 2.0]

    for min_support in supports:
        for min_purity in purities:
            for margin_thr in margin_thrs:
                pred = base_pred.copy()

                eligible = np.zeros(len(y), dtype=bool)
                retr_pred = np.full(len(y), -1, dtype=np.int64)
                retr_purity = np.zeros(len(y), dtype=np.float64)
                retr_support = np.zeros(len(y), dtype=np.int64)

                for i, key in enumerate(val_keys):
                    cnt = lookup.get(key)
                    if cnt is None:
                        continue

                    support = int(cnt.sum())
                    cls = int(cnt.argmax())
                    purity = float(cnt[cls] / support)

                    retr_pred[i] = cls
                    retr_purity[i] = purity
                    retr_support[i] = support

                    ok = (
                        support >= min_support
                        and purity >= min_purity
                    )

                    if margin_thr is not None:
                        ok = ok and margin[i] <= margin_thr

                    eligible[i] = ok

                pred[eligible] = retr_pred[eligible]

                f1 = f1_score(
                    y,
                    pred,
                    labels=range(len(CLASSES)),
                    average="macro",
                    zero_division=0,
                )
                acc = accuracy_score(y, pred)

                covered = int(eligible.sum())

                if covered:
                    retr_acc = float(
                        (retr_pred[eligible] == y[eligible]).mean()
                    )
                    base_acc_cov = float(
                        (base_pred[eligible] == y[eligible]).mean()
                    )
                    changed = int(
                        (
                            retr_pred[eligible]
                            != base_pred[eligible]
                        ).sum()
                    )
                else:
                    retr_acc = 0.0
                    base_acc_cov = 0.0
                    changed = 0

                rows.append({
                    "signature": name,
                    "min_support": min_support,
                    "min_purity": min_purity,
                    "margin_thr": (
                        -1.0 if margin_thr is None else margin_thr
                    ),
                    "macro_f1": f1,
                    "gain": f1 - BASE_F1,
                    "accuracy": acc,
                    "coverage_n": covered,
                    "coverage_rate": covered / len(y),
                    "changed_n": changed,
                    "retrieval_acc_covered": retr_acc,
                    "base_acc_covered": base_acc_cov,
                })

    result = pd.DataFrame(rows).sort_values(
        ["macro_f1", "accuracy"],
        ascending=False,
    )

    print("\n" + "=" * 100)
    print("SIGNATURE:", name)
    print("=" * 100)
    print(result.head(15).to_string(index=False))

    return result


print("Loading train...")
samples = load_all_train()

print("n train:", len(samples))

team = load_team_module()

print("Building teammate tabular features...")

print("samples type:", type(samples))
print("len samples:", len(samples))
print("first 5 types:", [type(x).__name__ for x in samples[:5]])

bad = [
    (i, type(x).__name__, repr(x)[:200])
    for i, x in enumerate(samples)
    if not isinstance(x, dict)
]

print("non-dict samples:", len(bad))
print("first bad samples:", bad[:10])

assert len(bad) == 0, bad[:10]

feat = pd.DataFrame(
    [team.make_features(sample) for sample in samples]
).reset_index(drop=True)

ids = np.array(
    [detect_id(x) for x in samples],
    dtype=str,
)

# ------------------------------------------------------------------
# labels are stored separately in train_labels.csv
# ------------------------------------------------------------------

label_df = pd.read_csv("data/train_labels.csv")

label_map = dict(
    zip(
        label_df["id"].astype(str),
        label_df["action"].astype(str),
    )
)

missing_label_ids = [
    sid for sid in ids
    if sid not in label_map
]

if missing_label_ids:
    raise ValueError(
        f"missing labels for {len(missing_label_ids)} samples; "
        f"first={missing_label_ids[:5]}"
    )

labels = np.array(
    [C2I[label_map[sid]] for sid in ids],
    dtype=np.int64,
)

print("feature shape:", feat.shape)
print("feature columns:")
print(list(feat.columns))

# ------------------------------------------------------------------
# fold0 IDs come directly from teammate validation predictions.
# All remaining train samples form the retrieval reference DB.
# ------------------------------------------------------------------

pred_df = pd.read_csv(PRED_CSV)
val_ids = pred_df["id"].astype(str).values

val_id_set = set(val_ids.tolist())

ref_mask = np.array(
    [sid not in val_id_set for sid in ids],
    dtype=bool,
)

print("reference rows:", int(ref_mask.sum()))
print("fold0 rows:", len(val_ids))

id_to_global = {
    sid: i
    for i, sid in enumerate(ids)
}

missing_val_ids = [
    sid for sid in val_ids
    if sid not in id_to_global
]

if missing_val_ids:
    raise ValueError(
        f"missing {len(missing_val_ids)} validation IDs in train; "
        f"first={missing_val_ids[:5]}"
    )

val_idx = np.array(
    [id_to_global[sid] for sid in val_ids],
    dtype=np.int64,
)

assert not np.any(ref_mask[val_idx])

y = labels[val_idx]

if "true_id" in pred_df.columns:
    csv_y = pred_df["true_id"].astype(int).values
    print("label alignment exact:", np.array_equal(y, csv_y))
    assert np.array_equal(y, csv_y)

# ------------------------------------------------------------------
# reconstruct current recipe base
# ------------------------------------------------------------------

raw = {
    k: np.load(p).astype(np.float64)
    for k, p in MODEL_PATHS.items()
}

for k, x in raw.items():
    print(k, x.shape)

base_score = sum(
    WEIGHTS[k] * (raw[k] / TEMPS[k])
    for k in WEIGHTS
)
base_score += BIAS

base_pred = base_score.argmax(1)

BASE_F1 = f1_score(
    y,
    base_pred,
    labels=range(len(CLASSES)),
    average="macro",
    zero_division=0,
)

print("\nBASE")
print("macro_f1:", BASE_F1)
print("accuracy:", accuracy_score(y, base_pred))

# ------------------------------------------------------------------
# State signatures
# ------------------------------------------------------------------

signature_sets = {
    "tail3": [
        "action_tail3",
    ],

    "tail3_lastres": [
        "action_tail3",
        "last_result",
    ],

    "tail3_lastres_ci": [
        "action_tail3",
        "last_result",
        "last_ci_status",
    ],

    "tail3_lastres_promptflags": [
        "action_tail3",
        "last_result",
        "pf_code",
        "pf_path",
        "pf_error",
        "pf_qmark",
    ],

    "lastprev_result": [
        "last_action",
        "prev_action",
        "last_result",
    ],

    "lastprev_result_promptflags": [
        "last_action",
        "prev_action",
        "last_result",
        "pf_code",
        "pf_path",
        "pf_error",
        "pf_qmark",
    ],

    "tail3_result_step": [
        "action_tail3",
        "last_result",
        "step_bucket",
    ],

    "tail3_result_state": [
        "action_tail3",
        "last_result",
        "last_ci_status",
        "git_dirty",
        "step_bucket",
    ],

    "tail3_intent": [
        "action_tail3",
        "kw_test",
        "kw_lint",
        "kw_search",
        "kw_web",
        "kw_edit",
        "kw_plan",
        "kw_ask",
    ],

    "tail3_result_intent": [
        "action_tail3",
        "last_result",
        "kw_test",
        "kw_lint",
        "kw_search",
        "kw_web",
        "kw_edit",
        "kw_plan",
        "kw_ask",
    ],
}

all_results = []

for name, cols in signature_sets.items():
    missing = [x for x in cols if x not in feat.columns]

    if missing:
        print("SKIP", name, "missing:", missing)
        continue

    keys = make_signature(feat, cols)

    result = evaluate_lookup(
        name=name,
        ref_keys=keys[ref_mask],
        ref_y=labels[ref_mask],
        val_keys=keys[val_idx],
        y=y,
        base_score=base_score,
    )

    all_results.append(result)

out = pd.concat(all_results, ignore_index=True)
out = out.sort_values(
    ["macro_f1", "accuracy"],
    ascending=False,
)

out.to_csv(
    "trajectory_state_retrieval_results.csv",
    index=False,
)

print("\n\n" + "#" * 110)
print("GLOBAL TOP 50")
print("#" * 110)

print(out.head(50).to_string(index=False))

print("\nSaved trajectory_state_retrieval_results.csv")
