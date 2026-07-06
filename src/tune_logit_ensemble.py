import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def parse_float_list(value):
    return [float(x.strip()) for x in str(value).split(",") if x.strip()]


def load_pair(csv_path, logits_path):
    df = pd.read_csv(csv_path)
    logits = np.load(logits_path)
    if len(df) != len(logits):
        raise ValueError(f"row mismatch: {csv_path} has {len(df)} rows, logits has {len(logits)}")
    if "id" not in df.columns or "true_id" not in df.columns:
        raise ValueError(f"{csv_path} must contain id and true_id columns")
    return df, logits.astype(np.float32)


def align_to_base(base_df, other_df, other_logits):
    id_to_idx = {str(sid): i for i, sid in enumerate(other_df["id"].astype(str).tolist())}
    indices = []
    missing = []
    for sid in base_df["id"].astype(str).tolist():
        idx = id_to_idx.get(sid)
        if idx is None:
            missing.append(sid)
        else:
            indices.append(idx)
    if missing:
        raise ValueError(f"missing {len(missing)} ids in other csv. first={missing[:3]}")
    return other_logits[np.array(indices)]


def normalize_logits(logits, mode):
    if mode == "none":
        return logits
    if mode == "zscore":
        mean = logits.mean(axis=1, keepdims=True)
        std = logits.std(axis=1, keepdims=True)
        return (logits - mean) / np.maximum(std, 1e-6)
    raise ValueError(f"unknown normalize={mode}")


def infer_id2action(df, n_classes):
    pairs = []
    for id_col, action_col in [("true_id", "true_action"), ("pred_id", "pred_action")]:
        if id_col in df.columns and action_col in df.columns:
            pairs.append(df[[id_col, action_col]].rename(columns={id_col: "id", action_col: "action"}))
    if not pairs:
        return [str(i) for i in range(n_classes)]
    all_pairs = pd.concat(pairs, ignore_index=True).drop_duplicates()
    out = []
    for i in range(n_classes):
        acts = sorted(set(all_pairs.loc[all_pairs["id"].astype(int) == i, "action"].astype(str)))
        out.append(acts[0] if len(acts) == 1 else str(i))
    return out


def softmax(logits):
    x = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(x)
    return p / p.sum(axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser(description="Tune weighted ensemble of two validation logits files.")
    parser.add_argument("--a_csv", required=True)
    parser.add_argument("--a_logits", required=True)
    parser.add_argument("--b_csv", required=True)
    parser.add_argument("--b_logits", required=True)
    parser.add_argument("--a_name", default="model_a")
    parser.add_argument("--b_name", default="model_b")
    parser.add_argument("--weights", default=",".join(f"{i / 20:.2f}" for i in range(21)))
    parser.add_argument("--normalize", choices=["none", "zscore"], default="none")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    a_df, a_logits = load_pair(args.a_csv, args.a_logits)
    b_df, b_logits_raw = load_pair(args.b_csv, args.b_logits)
    b_logits = align_to_base(a_df, b_df, b_logits_raw)

    if a_logits.shape != b_logits.shape:
        raise ValueError(f"logit shape mismatch: {a_logits.shape} vs {b_logits.shape}")

    y_true = a_df["true_id"].astype(int).values
    n_classes = a_logits.shape[1]
    id2action = infer_id2action(a_df, n_classes)

    a_logits = normalize_logits(a_logits, args.normalize)
    b_logits = normalize_logits(b_logits, args.normalize)

    rows = []
    best = None
    best_logits = None
    best_pred = None
    base_a = a_logits.argmax(axis=1)
    base_b = b_logits.argmax(axis=1)

    for wa in parse_float_list(args.weights):
        wb = 1.0 - wa
        ens = wa * a_logits + wb * b_logits
        pred = ens.argmax(axis=1)
        macro_f1 = f1_score(y_true, pred, labels=list(range(n_classes)), average="macro", zero_division=0)
        acc = accuracy_score(y_true, pred)
        row = {
            f"weight_{args.a_name}": wa,
            f"weight_{args.b_name}": wb,
            "macro_f1": macro_f1,
            "accuracy": acc,
            f"changed_vs_{args.a_name}": int((pred != base_a).sum()),
            f"changed_vs_{args.b_name}": int((pred != base_b).sum()),
        }
        rows.append(row)
        if best is None or (macro_f1, acc) > (best["macro_f1"], best["accuracy"]):
            best = row
            best_logits = ens
            best_pred = pred

    grid = pd.DataFrame(rows).sort_values(["macro_f1", "accuracy"], ascending=False)
    grid.to_csv(out_dir / "ensemble_grid.csv", index=False)

    probs = softmax(best_logits)
    out_pred = a_df.copy()
    out_pred["ensemble_pred_id"] = best_pred
    out_pred["ensemble_pred_action"] = [id2action[i] for i in best_pred]
    out_pred["ensemble_confidence"] = probs.max(axis=1)
    out_pred["ensemble_correct"] = best_pred == y_true
    out_pred.to_csv(out_dir / "val_predictions_ensemble.csv", index=False)
    np.save(out_dir / "val_logits_ensemble.npy", best_logits)

    summary = {
        "model_a": args.a_name,
        "model_b": args.b_name,
        "normalize": args.normalize,
        "best": best,
        "a_csv": args.a_csv,
        "a_logits": args.a_logits,
        "b_csv": args.b_csv,
        "b_logits": args.b_logits,
    }
    with open(out_dir / "ensemble_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Top ensemble grid:")
    print(grid.head(20).to_string(index=False))
    print("\nBest:")
    print(json.dumps(best, ensure_ascii=False, indent=2))
    print("\nSaved:")
    print(" -", out_dir / "ensemble_grid.csv")
    print(" -", out_dir / "ensemble_summary.json")
    print(" -", out_dir / "val_predictions_ensemble.csv")
    print(" -", out_dir / "val_logits_ensemble.npy")


if __name__ == "__main__":
    main()
