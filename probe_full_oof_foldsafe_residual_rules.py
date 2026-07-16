#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


ROOT = Path("oof_bundle_20260714")
OOF_ROOT = ROOT / "oof"
INDEX_PATH = ROOT / "index.csv"

BASE_NAME = "ens_large_tab_diet_lgb_oof"

N_CLASSES = 14

CLASS_NAMES = [
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


def macro_f1(y, pred):
    return f1_score(
        y,
        pred,
        labels=range(N_CLASSES),
        average="macro",
        zero_division=0,
    )


def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)

    return e / np.maximum(
        e.sum(axis=1, keepdims=True),
        1e-12,
    )


def load_all_logits():
    result = {}

    for path in sorted(
        OOF_ROOT.glob("*/logits.npy")
    ):
        name = path.parent.name

        arr = np.load(path)

        if arr.shape != (70000, 14):
            raise RuntimeError(
                f"{name}: bad shape {arr.shape}"
            )

        if not np.isfinite(arr).all():
            raise RuntimeError(
                f"{name}: nonfinite"
            )

        result[name] = arr.astype(np.float64)

    return result


def build_state(logits):
    base_logits = logits[BASE_NAME]
    base_prob = softmax(base_logits)
    base_pred = base_prob.argmax(axis=1)

    order = np.argsort(
        base_prob,
        axis=1,
    )

    second = order[:, -2]

    top1 = base_prob[
        np.arange(len(base_prob)),
        order[:, -1],
    ]

    top2 = base_prob[
        np.arange(len(base_prob)),
        order[:, -2],
    ]

    margin = top1 - top2

    voter_names = [
        name
        for name in logits
        if name != BASE_NAME
    ]

    voter_preds = np.column_stack([
        logits[name].argmax(axis=1)
        for name in voter_names
    ])

    vote_counts = np.column_stack([
        (voter_preds == cls).sum(axis=1)
        for cls in range(N_CLASSES)
    ])

    alt_class = vote_counts.argmax(axis=1)

    alt_votes = vote_counts[
        np.arange(len(vote_counts)),
        alt_class,
    ]

    return {
        "base_pred": base_pred,
        "base_prob": base_prob,
        "second": second,
        "margin": margin,
        "alt_class": alt_class,
        "alt_votes": alt_votes,
        "voter_names": voter_names,
        "n_voters": len(voter_names),
    }


def enumerate_rules(state):
    rules = []

    margins = [
        0.01,
        0.02,
        0.03,
        0.05,
        0.08,
        0.10,
        0.15,
        0.20,
        0.30,
        1.00,
    ]

    n_voters = state["n_voters"]

    for base_cls in range(N_CLASSES):
        for alt_cls in range(N_CLASSES):
            if alt_cls == base_cls:
                continue

            for min_votes in range(
                2,
                n_voters + 1,
            ):
                for margin_thr in margins:
                    for require_top2 in [
                        False,
                        True,
                    ]:
                        rules.append({
                            "base_cls": base_cls,
                            "alt_cls": alt_cls,
                            "min_votes": min_votes,
                            "margin_thr": margin_thr,
                            "require_top2": require_top2,
                        })

    return rules


def rule_mask(rule, state, idx):
    mask = (
        (state["base_pred"][idx] == rule["base_cls"])
        & (state["alt_class"][idx] == rule["alt_cls"])
        & (state["alt_votes"][idx] >= rule["min_votes"])
        & (state["margin"][idx] <= rule["margin_thr"])
    )

    if rule["require_top2"]:
        mask &= (
            state["second"][idx]
            == rule["alt_cls"]
        )

    return mask


def find_rules(
    y,
    state,
    train_idx,
):
    candidates = []

    for rule in enumerate_rules(state):
        mask = rule_mask(
            rule,
            state,
            train_idx,
        )

        n = int(mask.sum())

        if n < 40:
            continue

        pred_alt = state["alt_class"][
            train_idx
        ][mask]

        truth = y[train_idx][mask]

        correct = int(
            (pred_alt == truth).sum()
        )

        precision = correct / n

        # Precision-first.
        if precision < 0.60:
            continue

        item = dict(rule)

        item.update({
            "train_n": n,
            "train_correct": correct,
            "train_precision": precision,
        })

        candidates.append(item)

    return candidates


def greedy_select(
    y,
    state,
    train_idx,
    candidates,
    max_rounds=30,
):
    current = state["base_pred"][
        train_idx
    ].copy()

    current_f1 = macro_f1(
        y[train_idx],
        current,
    )

    used = np.zeros(
        len(train_idx),
        dtype=bool,
    )

    selected = []

    for round_idx in range(max_rounds):
        best = None

        for rule in candidates:
            mask = rule_mask(
                rule,
                state,
                train_idx,
            )

            mask &= ~used

            n = int(mask.sum())

            if n < 20:
                continue

            trial = current.copy()

            trial[mask] = rule["alt_cls"]

            f1 = macro_f1(
                y[train_idx],
                trial,
            )

            gain = f1 - current_f1

            if (
                best is None
                or gain > best["gain"]
            ):
                best = {
                    "rule": rule,
                    "mask": mask,
                    "gain": gain,
                    "f1": f1,
                }

        if (
            best is None
            or best["gain"] <= 0
        ):
            break

        rule = best["rule"]
        mask = best["mask"]

        current[mask] = rule["alt_cls"]
        used[mask] = True

        current_f1 = best["f1"]

        selected.append(
            dict(rule)
        )

    return selected, current_f1


def apply_rules(
    state,
    idx,
    selected,
):
    pred = state["base_pred"][
        idx
    ].copy()

    used = np.zeros(
        len(idx),
        dtype=bool,
    )

    for rule in selected:
        mask = rule_mask(
            rule,
            state,
            idx,
        )

        mask &= ~used

        pred[mask] = rule["alt_cls"]

        used[mask] = True

    return pred


def main():
    index = pd.read_csv(
        INDEX_PATH
    )

    y = index["true_id"].to_numpy(
        dtype=np.int64
    )

    folds = index["fold"].to_numpy(
        dtype=np.int64
    )

    logits = load_all_logits()

    print("=" * 120)
    print("OOF INVENTORY")
    print("=" * 120)

    for name, arr in logits.items():
        pred = arr.argmax(axis=1)

        print(
            f"{name:55s} "
            f"F1={macro_f1(y, pred):.9f}"
        )

    if BASE_NAME not in logits:
        raise RuntimeError(
            f"missing base {BASE_NAME}"
        )

    state = build_state(logits)

    print()
    print(
        "BASE:",
        BASE_NAME
    )

    print(
        "N VOTERS:",
        state["n_voters"]
    )

    print(
        "VOTERS:"
    )

    for name in state["voter_names"]:
        print(" -", name)

    oof_pred = state[
        "base_pred"
    ].copy()

    fold_rows = []
    selected_rows = []

    print()
    print("=" * 120)
    print("FOLD-SAFE RULE CV")
    print("=" * 120)

    for fold in sorted(
        np.unique(folds)
    ):
        train_idx = np.where(
            folds != fold
        )[0]

        val_idx = np.where(
            folds == fold
        )[0]

        base_val_f1 = macro_f1(
            y[val_idx],
            state["base_pred"][val_idx],
        )

        print()
        print(
            f"===== HOLDOUT FOLD {fold} ====="
        )

        print(
            "train:",
            len(train_idx),
            "val:",
            len(val_idx),
        )

        candidates = find_rules(
            y,
            state,
            train_idx,
        )

        print(
            "eligible train regimes:",
            len(candidates),
        )

        selected, train_f1 = greedy_select(
            y,
            state,
            train_idx,
            candidates,
        )

        val_pred = apply_rules(
            state,
            val_idx,
            selected,
        )

        val_f1 = macro_f1(
            y[val_idx],
            val_pred,
        )

        oof_pred[val_idx] = val_pred

        changed = (
            val_pred
            != state["base_pred"][val_idx]
        )

        beneficial = int(
            (
                changed
                & (
                    state["base_pred"][val_idx]
                    != y[val_idx]
                )
                & (
                    val_pred == y[val_idx]
                )
            ).sum()
        )

        harmful = int(
            (
                changed
                & (
                    state["base_pred"][val_idx]
                    == y[val_idx]
                )
                & (
                    val_pred != y[val_idx]
                )
            ).sum()
        )

        print(
            "selected rules:",
            len(selected),
        )

        print(
            "base val F1:",
            base_val_f1,
        )

        print(
            "rule val F1:",
            val_f1,
        )

        print(
            "val gain:",
            val_f1 - base_val_f1,
        )

        print(
            "changed:",
            int(changed.sum()),
            "beneficial:",
            beneficial,
            "harmful:",
            harmful,
        )

        fold_rows.append({
            "fold": int(fold),
            "base_f1": base_val_f1,
            "rule_f1": val_f1,
            "gain": val_f1 - base_val_f1,
            "n_rules": len(selected),
            "changed": int(changed.sum()),
            "beneficial": beneficial,
            "harmful": harmful,
        })

        for order, rule in enumerate(
            selected,
            start=1,
        ):
            row = {
                "holdout_fold": int(fold),
                "order": order,
                **rule,
            }

            row["base_name"] = (
                CLASS_NAMES[
                    rule["base_cls"]
                ]
            )

            row["alt_name"] = (
                CLASS_NAMES[
                    rule["alt_cls"]
                ]
            )

            selected_rows.append(row)

    base_full_f1 = macro_f1(
        y,
        state["base_pred"],
    )

    rule_oof_f1 = macro_f1(
        y,
        oof_pred,
    )

    print()
    print("=" * 120)
    print("FINAL FULL OOF")
    print("=" * 120)

    print(
        "BASE OOF F1:",
        base_full_f1,
    )

    print(
        "RULE OOF F1:",
        rule_oof_f1,
    )

    print(
        "GAIN:",
        rule_oof_f1 - base_full_f1,
    )

    changed = (
        oof_pred != state["base_pred"]
    )

    print(
        "CHANGED:",
        int(changed.sum()),
    )

    print(
        "BENEFICIAL:",
        int(
            (
                changed
                & (
                    state["base_pred"] != y
                )
                & (
                    oof_pred == y
                )
            ).sum()
        ),
    )

    print(
        "HARMFUL:",
        int(
            (
                changed
                & (
                    state["base_pred"] == y
                )
                & (
                    oof_pred != y
                )
            ).sum()
        ),
    )

    pd.DataFrame(
        fold_rows
    ).to_csv(
        "full_oof_foldsafe_rule_results.csv",
        index=False,
    )

    pd.DataFrame(
        selected_rows
    ).to_csv(
        "full_oof_foldsafe_selected_rules.csv",
        index=False,
    )

    np.save(
        "full_oof_foldsafe_rule_pred.npy",
        oof_pred.astype(np.int64),
    )


if __name__ == "__main__":
    main()
