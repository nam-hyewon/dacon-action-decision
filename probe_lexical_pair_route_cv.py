#!/usr/bin/env python3

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold


CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]

REPRO_PRED = (
    "repro_bundle/large_tab_diet_rob8_repro/"
    "submit/model/kluebert_diet_fold0/val_predictions.csv"
)

pred_df = pd.read_csv(REPRO_PRED)

y = pred_df["true_id"].astype(int).values

base_pred = np.load(
    "exact_repro_base_pred.npy"
).astype(np.int64)

lex_df = pd.read_csv(
    "lexical_expert_fold0_predictions.csv"
)

assert np.array_equal(
    pred_df["id"].astype(str).values,
    lex_df["id"].astype(str).values,
)

lex_pred = (
    lex_df["lex_pred_id"]
    .astype(int)
    .values
)

disagree = base_pred != lex_pred

benefit = (
    (base_pred != y)
    & (lex_pred == y)
)

harm = (
    (base_pred == y)
    & (lex_pred != y)
)

neutral = (
    disagree
    & ~benefit
    & ~harm
)

BASE_F1 = f1_score(
    y,
    base_pred,
    labels=range(len(CLASSES)),
    average="macro",
    zero_division=0,
)

print("BASE F1:", BASE_F1)
print("disagreement:", int(disagree.sum()))
print("benefit:", int(benefit.sum()))
print("harm:", int(harm.sum()))
print("neutral:", int(neutral.sum()))


idx = np.where(disagree)[0]

# Stratify by benefit flag only for stable positive allocation.
strat_y = benefit[idx].astype(np.int64)

skf = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42,
)


configs = []

for min_support in [
    3, 5, 10, 15, 20, 30, 50, 75, 100,
]:
    for min_net in [
        1, 2, 3, 5, 10, 15, 20,
    ]:
        for min_benefit_rate in [
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
            0.55,
            0.60,
        ]:
            configs.append(
                (
                    min_support,
                    min_net,
                    min_benefit_rate,
                )
            )


all_rows = []

for min_support, min_net, min_benefit_rate in configs:

    oof_use = np.zeros(
        len(y),
        dtype=bool,
    )

    selected_pairs_by_fold = []

    for fold, (tr_local, va_local) in enumerate(
        skf.split(idx, strat_y)
    ):
        tr_idx = idx[tr_local]
        va_idx = idx[va_local]

        stats = {}

        for i in tr_idx:
            key = (
                int(base_pred[i]),
                int(lex_pred[i]),
            )

            if key not in stats:
                stats[key] = {
                    "support": 0,
                    "benefit": 0,
                    "harm": 0,
                    "neutral": 0,
                }

            stats[key]["support"] += 1
            stats[key]["benefit"] += int(benefit[i])
            stats[key]["harm"] += int(harm[i])
            stats[key]["neutral"] += int(neutral[i])

        selected = set()

        for key, s in stats.items():
            support = s["support"]
            ben = s["benefit"]
            hm = s["harm"]

            net = ben - hm
            benefit_rate = ben / support

            if (
                support >= min_support
                and net >= min_net
                and benefit_rate >= min_benefit_rate
            ):
                selected.add(key)

        selected_pairs_by_fold.append(selected)

        for i in va_idx:
            key = (
                int(base_pred[i]),
                int(lex_pred[i]),
            )

            if key in selected:
                oof_use[i] = True


    pred = base_pred.copy()
    pred[oof_use] = lex_pred[oof_use]

    f1 = f1_score(
        y,
        pred,
        labels=range(len(CLASSES)),
        average="macro",
        zero_division=0,
    )

    all_rows.append({
        "min_support": min_support,
        "min_net": min_net,
        "min_benefit_rate": min_benefit_rate,
        "macro_f1": f1,
        "gain": f1 - BASE_F1,
        "accuracy": accuracy_score(y, pred),
        "use_n": int(oof_use.sum()),
        "benefit_n": int(benefit[oof_use].sum()),
        "harm_n": int(harm[oof_use].sum()),
        "neutral_n": int(neutral[oof_use].sum()),
        "net_correct_gain": int(
            benefit[oof_use].sum()
            - harm[oof_use].sum()
        ),
    })


result = pd.DataFrame(all_rows).sort_values(
    ["macro_f1", "accuracy"],
    ascending=False,
)

print()
print("=" * 120)
print("PAIR ROUTE TOP 50")
print("=" * 120)

print(
    result.head(50).to_string(
        index=False,
    )
)

result.to_csv(
    "lexical_pair_route_cv_results.csv",
    index=False,
)


# ============================================================
# Full-fold pair diagnostics, analysis only
# ============================================================

diag = []

for a in range(len(CLASSES)):
    for b in range(len(CLASSES)):
        if a == b:
            continue

        mask = (
            disagree
            & (base_pred == a)
            & (lex_pred == b)
        )

        if not mask.any():
            continue

        ben = int(benefit[mask].sum())
        hm = int(harm[mask].sum())
        neu = int(neutral[mask].sum())
        n = int(mask.sum())

        diag.append({
            "base": CLASSES[a],
            "lex": CLASSES[b],
            "n": n,
            "benefit": ben,
            "harm": hm,
            "neutral": neu,
            "net": ben - hm,
            "benefit_rate": ben / n,
            "harm_rate": hm / n,
        })


diag_df = pd.DataFrame(diag).sort_values(
    ["net", "benefit"],
    ascending=False,
)

print()
print("=" * 120)
print("FULL-FOLD TRANSITION DIAGNOSTICS")
print("ANALYSIS ONLY -- NOT CV SCORE")
print("=" * 120)

print(
    diag_df.head(50).to_string(
        index=False,
    )
)

diag_df.to_csv(
    "lexical_transition_diagnostics.csv",
    index=False,
)

print()
print("Saved:")
print(" - lexical_pair_route_cv_results.csv")
print(" - lexical_transition_diagnostics.csv")
