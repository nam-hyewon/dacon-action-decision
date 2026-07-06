#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def parse_float_list(value):
    return [float(x.strip()) for x in str(value).split(",") if x.strip()]


def infer_id2action(val_df, n_classes):
    if "true_action" in val_df.columns:
        pairs = val_df[["true_id", "true_action"]].drop_duplicates()
        mapping = {
            int(r["true_id"]): str(r["true_action"])
            for _, r in pairs.iterrows()
        }
        return [mapping.get(i, str(i)) for i in range(n_classes)]
    return [str(i) for i in range(n_classes)]


def main():
    parser = argparse.ArgumentParser(
        description="Tune shrink scale for already-applied specialist logits."
    )
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--base_logits", required=True)
    parser.add_argument("--specialist_logits", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--scales",
        default="0,0.25,0.4,0.5,0.6,0.75,0.9,1.0,1.1,1.25",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    val_df = pd.read_csv(args.val_csv)
    base_logits = np.load(args.base_logits).astype(np.float64)
    specialist_logits = np.load(args.specialist_logits).astype(np.float64)
    if base_logits.shape != specialist_logits.shape:
        raise ValueError(
            f"logit shape mismatch: {base_logits.shape} vs {specialist_logits.shape}"
        )
    if len(val_df) != len(base_logits):
        raise ValueError(
            f"row mismatch: val_csv={len(val_df)} logits={len(base_logits)}"
        )

    y_true = val_df["true_id"].astype(int).values
    n_classes = base_logits.shape[1]
    id2action = infer_id2action(val_df, n_classes)
    delta = specialist_logits - base_logits

    base_pred = base_logits.argmax(axis=1)
    full_pred = specialist_logits.argmax(axis=1)
    base_f1 = f1_score(
        y_true,
        base_pred,
        labels=list(range(n_classes)),
        average="macro",
        zero_division=0,
    )
    full_f1 = f1_score(
        y_true,
        full_pred,
        labels=list(range(n_classes)),
        average="macro",
        zero_division=0,
    )
    base_acc = accuracy_score(y_true, base_pred)
    full_acc = accuracy_score(y_true, full_pred)

    rows = []
    best = None
    best_logits = None
    best_pred = None
    for scale in parse_float_list(args.scales):
        logits = base_logits + scale * delta
        pred = logits.argmax(axis=1)
        macro_f1 = f1_score(
            y_true,
            pred,
            labels=list(range(n_classes)),
            average="macro",
            zero_division=0,
        )
        acc = accuracy_score(y_true, pred)
        row = {
            "scale": scale,
            "macro_f1": macro_f1,
            "delta_vs_base_f1": macro_f1 - base_f1,
            "delta_vs_full_specialist_f1": macro_f1 - full_f1,
            "accuracy": acc,
            "delta_vs_base_accuracy": acc - base_acc,
            "delta_vs_full_specialist_accuracy": acc - full_acc,
            "changed_vs_base": int((pred != base_pred).sum()),
            "changed_vs_full_specialist": int((pred != full_pred).sum()),
        }
        rows.append(row)
        if best is None or (macro_f1, acc) > (best["macro_f1"], best["accuracy"]):
            best = row
            best_logits = logits
            best_pred = pred

    res = pd.DataFrame(rows).sort_values(["macro_f1", "accuracy"], ascending=False)
    res.to_csv(out_dir / "specialist_shrink_grid.csv", index=False)

    out_pred = val_df.copy()
    out_pred["shrink_pred_id"] = best_pred
    out_pred["shrink_pred_action"] = [id2action[i] for i in best_pred]
    out_pred["shrink_correct"] = best_pred == y_true
    out_pred["shrink_changed_vs_base"] = best_pred != base_pred
    out_pred["shrink_changed_vs_full_specialist"] = best_pred != full_pred
    out_pred.to_csv(out_dir / "val_predictions_with_specialist_shrink.csv", index=False)
    np.save(out_dir / "val_logits_with_specialist_shrink.npy", best_logits)

    summary = {
        "base": {"macro_f1": float(base_f1), "accuracy": float(base_acc)},
        "full_specialist": {
            "macro_f1": float(full_f1),
            "accuracy": float(full_acc),
        },
        "best": best,
        "out_dir": str(out_dir),
    }
    with open(out_dir / "specialist_shrink_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Base:", json.dumps(summary["base"], indent=2))
    print("Full specialist:", json.dumps(summary["full_specialist"], indent=2))
    print("\nTop shrink grid:")
    print(res.head(30).to_string(index=False))
    print("\nBest:")
    print(json.dumps(best, ensure_ascii=False, indent=2))
    print("\nSaved:")
    print(" -", out_dir / "specialist_shrink_grid.csv")
    print(" -", out_dir / "specialist_shrink_summary.json")
    print(" -", out_dir / "val_predictions_with_specialist_shrink.csv")
    print(" -", out_dir / "val_logits_with_specialist_shrink.npy")


if __name__ == "__main__":
    main()
