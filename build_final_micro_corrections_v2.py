#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


N_CLASSES = 14

ACTIONS = [
    "read_file",
    "grep_search",
    "list_directory",
    "glob_pattern",
    "edit_file",
    "write_file",
    "apply_patch",
    "run_bash",
    "run_tests",
    "lint_or_typecheck",
    "ask_user",
    "plan_task",
    "web_search",
    "respond_only",
]

ACTION2ID = {
    a: i for i, a in enumerate(ACTIONS)
}

BASE_SCORE_PATH = Path(
    "exact_repro_base_v2_score.npy"
)

BASE_PRED_PATH = Path(
    "exact_repro_base_v2_pred.npy"
)

VAL_REF = Path(
    "repro_bundle/large_tab_diet_rob8_repro/"
    "submit/model/kluebert_diet_fold0/"
    "val_predictions.csv"
)

TRAIN_JSONL = Path("data/train.jsonl")
TRAIN_LABELS = Path("data/train_labels.csv")

OUT_DIR = Path("final_micro_corrections_v2")


def macro_f1(y, pred):
    return f1_score(
        y,
        pred,
        labels=range(N_CLASSES),
        average="macro",
        zero_division=0,
    )


def get_last_action(row):
    hist = row.get("history") or []

    for item in reversed(hist):
        action = item.get("action")

        if action in ACTION2ID:
            return action

    return None


def load_train_prior():
    label_df = pd.read_csv(TRAIN_LABELS)

    label_col = None

    for c in [
        "action",
        "label",
        "true_action",
    ]:
        if c in label_df.columns:
            label_col = c
            break

    if label_col is None:
        raise RuntimeError(
            f"cannot find action column: "
            f"{label_df.columns.tolist()}"
        )

    id_to_action = dict(
        zip(
            label_df["id"].astype(str),
            label_df[label_col].astype(str),
        )
    )

    counts = np.ones(
        (N_CLASSES, N_CLASSES),
        dtype=np.float64,
    )

    with TRAIN_JSONL.open() as f:
        for line in f:
            row = json.loads(line)

            row_id = str(row["id"])

            true_action = id_to_action.get(
                row_id
            )

            last_action = get_last_action(row)

            if (
                true_action not in ACTION2ID
                or last_action not in ACTION2ID
            ):
                continue

            counts[
                ACTION2ID[last_action],
                ACTION2ID[true_action],
            ] += 1.0

    prior = counts / counts.sum(
        axis=1,
        keepdims=True,
    )

    return np.log(
        np.maximum(prior, 1e-12)
    )


def load_val_last_actions(ref_ids):
    wanted = set(ref_ids)

    found = {}

    with TRAIN_JSONL.open() as f:
        for line in f:
            row = json.loads(line)

            row_id = str(row["id"])

            if row_id not in wanted:
                continue

            action = get_last_action(row)

            found[row_id] = (
                ACTION2ID[action]
                if action in ACTION2ID
                else -1
            )

    missing = [
        x for x in ref_ids
        if x not in found
    ]

    if missing:
        raise RuntimeError(
            f"missing validation rows in train.jsonl: "
            f"{len(missing)}"
        )

    return np.asarray(
        [found[x] for x in ref_ids],
        dtype=np.int64,
    )


def apply_transition(
    score,
    last_action_ids,
    log_prior,
    alpha,
):
    out = score.copy()

    valid = last_action_ids >= 0

    out[valid] += (
        alpha
        * log_prior[
            last_action_ids[valid]
        ]
    )

    return out


def greedy_class_bias(
    y,
    score,
):
    current = score.copy()
    current_pred = current.argmax(axis=1)
    current_f1 = macro_f1(y, current_pred)

    total_bias = np.zeros(
        N_CLASSES,
        dtype=np.float64,
    )

    steps = [
        0.20,
        0.10,
        0.05,
        0.025,
        0.0125,
    ]

    trace = []

    for step in steps:
        improved = True

        while improved:
            improved = False
            best = None

            for cls in range(N_CLASSES):
                for delta in [-step, step]:
                    trial = current.copy()
                    trial[:, cls] += delta

                    pred = trial.argmax(axis=1)
                    f1 = macro_f1(y, pred)

                    gain = f1 - current_f1

                    if (
                        best is None
                        or gain > best["gain"]
                    ):
                        best = {
                            "cls": cls,
                            "delta": delta,
                            "f1": f1,
                            "gain": gain,
                        }

            if (
                best is not None
                and best["gain"] > 1e-10
            ):
                cls = best["cls"]
                delta = best["delta"]

                current[:, cls] += delta
                total_bias[cls] += delta

                current_f1 = best["f1"]
                improved = True

                item = {
                    "step": step,
                    "class_id": cls,
                    "class_name": ACTIONS[cls],
                    "delta": delta,
                    "cumulative_bias": (
                        total_bias[cls]
                    ),
                    "f1": current_f1,
                    "gain_from_previous": (
                        best["gain"]
                    ),
                }

                trace.append(item)

                print(
                    "BIAS KEEP:",
                    item,
                    flush=True,
                )

    return (
        current,
        total_bias,
        current_f1,
        trace,
    )


def main():
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ref = pd.read_csv(VAL_REF)

    ref_ids = (
        ref["id"]
        .astype(str)
        .tolist()
    )

    y = ref["true_id"].to_numpy(
        dtype=np.int64
    )

    score = np.load(
        BASE_SCORE_PATH
    ).astype(np.float64)

    saved_base_pred = np.load(
        BASE_PRED_PATH
    ).astype(np.int64)

    raw_pred = score.argmax(axis=1)

    print(
        "saved base pred == score argmax:",
        np.array_equal(
            saved_base_pred,
            raw_pred,
        )
    )

    base_f1 = macro_f1(
        y,
        saved_base_pred,
    )

    print()
    print("=" * 120)
    print("BASE")
    print("=" * 120)

    print("F1:", base_f1)

    log_prior = load_train_prior()

    last_action_ids = load_val_last_actions(
        ref_ids
    )

    # --------------------------------------------------
    # Stage 1: transition prior.
    # --------------------------------------------------

    alpha_grid = np.concatenate([
        np.arange(
            0.0,
            0.051,
            0.0025,
        ),
        np.arange(
            0.055,
            0.151,
            0.005,
        ),
        np.arange(
            0.16,
            0.301,
            0.01,
        ),
    ])

    prior_rows = []

    best_prior = None

    for alpha in alpha_grid:
        trial = apply_transition(
            score,
            last_action_ids,
            log_prior,
            float(alpha),
        )

        pred = trial.argmax(axis=1)

        f1 = macro_f1(y, pred)

        row = {
            "alpha": float(alpha),
            "f1": f1,
            "gain": f1 - base_f1,
            "changed": int(
                (pred != saved_base_pred).sum()
            ),
            "beneficial": int(
                (
                    (saved_base_pred != y)
                    & (pred == y)
                ).sum()
            ),
            "harmful": int(
                (
                    (saved_base_pred == y)
                    & (pred != y)
                ).sum()
            ),
        }

        prior_rows.append(row)

        if (
            best_prior is None
            or f1 > best_prior["f1"]
        ):
            best_prior = {
                **row,
                "score": trial,
            }

    prior_df = pd.DataFrame(
        prior_rows
    ).sort_values(
        "f1",
        ascending=False,
    )

    prior_df.to_csv(
        OUT_DIR / "transition_sweep.csv",
        index=False,
    )

    print()
    print("=" * 120)
    print("TRANSITION PRIOR")
    print("=" * 120)

    print(
        prior_df.head(20).to_string(
            index=False
        )
    )

    if best_prior["f1"] > base_f1:
        current_score = best_prior["score"]
        current_f1 = best_prior["f1"]
        kept_alpha = best_prior["alpha"]

        print(
            "KEEP TRANSITION PRIOR:",
            kept_alpha,
            current_f1,
        )

    else:
        current_score = score.copy()
        current_f1 = base_f1
        kept_alpha = 0.0

        print("REJECT TRANSITION PRIOR")

    # --------------------------------------------------
    # Stage 2: tiny residual class bias.
    # --------------------------------------------------

    before_bias_f1 = current_f1

    (
        bias_score,
        bias,
        bias_f1,
        bias_trace,
    ) = greedy_class_bias(
        y,
        current_score,
    )

    if bias_f1 > before_bias_f1:
        current_score = bias_score
        current_f1 = bias_f1
        kept_bias = bias

        print(
            "KEEP CLASS BIAS"
        )

    else:
        kept_bias = np.zeros(
            N_CLASSES,
            dtype=np.float64,
        )

        print(
            "REJECT CLASS BIAS"
        )

    final_pred = current_score.argmax(
        axis=1
    )

    print()
    print("=" * 120)
    print("FINAL")
    print("=" * 120)

    print(
        "BASE F1:",
        base_f1,
    )

    print(
        "FINAL F1:",
        current_f1,
    )

    print(
        "TOTAL GAIN:",
        current_f1 - base_f1,
    )

    print(
        "TRANSITION ALPHA:",
        kept_alpha,
    )

    print(
        "ADDITIONAL CLASS BIAS:"
    )

    for i, value in enumerate(
        kept_bias
    ):
        print(
            i,
            ACTIONS[i],
            float(value),
        )

    print(
        "CHANGED:",
        int(
            (
                final_pred
                != saved_base_pred
            ).sum()
        ),
    )

    print(
        "BENEFICIAL:",
        int(
            (
                (saved_base_pred != y)
                & (final_pred == y)
            ).sum()
        ),
    )

    print(
        "HARMFUL:",
        int(
            (
                (saved_base_pred == y)
                & (final_pred != y)
            ).sum()
        ),
    )

    recipe = {
        "base": "exact_repro_base_v2",
        "base_f1": float(base_f1),
        "final_f1": float(current_f1),
        "gain": float(
            current_f1 - base_f1
        ),
        "transition_alpha": float(
            kept_alpha
        ),
        "additional_class_bias": [
            float(x)
            for x in kept_bias
        ],
    }

    (
        OUT_DIR / "final_recipe.json"
    ).write_text(
        json.dumps(
            recipe,
            indent=2,
        )
    )

    pd.DataFrame(
        bias_trace
    ).to_csv(
        OUT_DIR / "bias_trace.csv",
        index=False,
    )

    np.save(
        OUT_DIR / "final_val_score.npy",
        current_score.astype(
            np.float32
        ),
    )

    np.save(
        OUT_DIR / "final_val_pred.npy",
        final_pred.astype(
            np.int64
        ),
    )


if __name__ == "__main__":
    main()
