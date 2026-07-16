import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report

ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]
label2id = {c: i for i, c in enumerate(ALL_CLASSES)}
id2label = {i: c for i, c in enumerate(ALL_CLASSES)}

ROOT = Path(".")
ENS = ROOT / "team_4model_ensemble_07626" / "ensemble_07626"

MODELS = {
    "deberta": ENS / "fold0_team/deberta_best_single",
    "kluebert": ENS / "fold0_team/kluebert_fold0",
    "tabular": ENS / "fold0_team/tabular_fold0_seed_ensemble",
}

def row_zscore(x, eps=1e-6):
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + eps)

def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

def find_label_col(df):
    for c in ["label", "true_label", "action", "target"]:
        if c in df.columns:
            return c
    return None

def get_y_true(ref_df):
    label_col = find_label_col(ref_df)
    if label_col is not None:
        vals = ref_df[label_col].tolist()
        if len(vals) > 0 and isinstance(vals[0], (int, np.integer)):
            return np.array(vals, dtype=int)
        return np.array([label2id[str(x)] for x in vals], dtype=int)

    labels_df = pd.read_csv(ROOT / "data/train_labels.csv")
    label_col = find_label_col(labels_df)
    if label_col is None:
        raise ValueError(f"Cannot find label column in train_labels.csv: {labels_df.columns.tolist()}")

    id_to_label = dict(zip(labels_df["id"].astype(str), labels_df[label_col].astype(str)))
    ids = ref_df["id"].astype(str).tolist()
    return np.array([label2id[id_to_label[i]] for i in ids], dtype=int)

def load_model(name, path):
    logits_path = path / "val_logits.npy"
    pred_path = path / "val_predictions.csv"
    if not logits_path.exists():
        raise FileNotFoundError(f"{name}: missing {logits_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"{name}: missing {pred_path}")

    logits = np.load(logits_path)
    df = pd.read_csv(pred_path)
    ids = df["id"].astype(str).tolist()
    return logits, df, ids

def grid_weights(names, step=0.05):
    units = int(round(1 / step))
    n = len(names)
    for counts in itertools.product(range(units + 1), repeat=n):
        if sum(counts) != units:
            continue
        weights = np.array(counts, dtype=np.float32) / units
        yield dict(zip(names, weights.tolist()))

def combine(logits_dict, weights):
    out = None
    for name, w in weights.items():
        if w == 0:
            continue
        x = logits_dict[name]
        out = x * w if out is None else out + x * w
    return out

def eval_pred(y_true, logits):
    pred = logits.argmax(axis=1)
    return f1_score(y_true, pred, average="macro"), accuracy_score(y_true, pred)

def main():
    loaded = {}
    ref_ids = None
    ref_df = None

    print("===== loading models =====")
    for name, path in MODELS.items():
        logits, df, ids = load_model(name, path)
        if ref_ids is None:
            ref_ids = ids
            ref_df = df
        else:
            if ref_ids != ids:
                raise ValueError(f"id order mismatch: {name}")

        loaded[name] = logits
        print(f"{name:8s} logits={logits.shape} path={path}")

    y_true = get_y_true(ref_df)
    print("valid n:", len(y_true))

    # scale normalization
    z = {k: row_zscore(v) for k, v in loaded.items()}

    print("\n===== single models, row-zscore logits =====")
    for name in ["deberta", "kluebert", "tabular"]:
        f1, acc = eval_pred(y_true, z[name])
        print(f"{name:8s} macro_f1={f1:.6f} acc={acc:.6f}")

    results = []

    # 4-model weighted average
    names4 = ["deberta", "kluebert", "tabular"]
    best4 = None
    for weights in grid_weights(names4, step=0.05):
        logits = combine(z, weights)
        f1, acc = eval_pred(y_true, logits)
        if best4 is None or f1 > best4["macro_f1"]:
            best4 = {"name": "best4", "macro_f1": f1, "accuracy": acc, "weights": weights, "logits": logits}

    print("\n===== BEST 4-model weighted average =====")
    print("macro_f1:", best4["macro_f1"])
    print("accuracy:", best4["accuracy"])
    print("weights:", best4["weights"])
    results.append(best4)

    # 3-model backup without deberta
    names3 = ["kluebert", "tabular"]
    best3 = None
    for weights in grid_weights(names3, step=0.05):
        logits = combine(z, weights)
        f1, acc = eval_pred(y_true, logits)
        if best3 is None or f1 > best3["macro_f1"]:
            best3 = {"name": "best3_rkt", "macro_f1": f1, "accuracy": acc, "weights": weights, "logits": logits}

    print("\n===== BEST 3-model RKT weighted average =====")
    print("macro_f1:", best3["macro_f1"])
    print("accuracy:", best3["accuracy"])
    print("weights:", best3["weights"])
    results.append(best3)

    # gated: use deberta only when confident, otherwise use RKT backup
    print("\n===== gated: deberta if confident else RKT =====")
    deb_prob = softmax(z["deberta"])
    deb_conf = deb_prob.max(axis=1)
    backup_logits = best3["logits"]

    best_gate = None
    for th in np.arange(0.30, 0.96, 0.01):
        use_deb = deb_conf >= th
        logits = backup_logits.copy()
        logits[use_deb] = z["deberta"][use_deb]
        f1, acc = eval_pred(y_true, logits)
        if best_gate is None or f1 > best_gate["macro_f1"]:
            best_gate = {
                "name": "gate_deb_else_rkt",
                "macro_f1": f1,
                "accuracy": acc,
                "threshold": float(th),
                "use_deberta_n": int(use_deb.sum()),
                "backup_weights": best3["weights"],
                "logits": logits,
            }

    print("macro_f1:", best_gate["macro_f1"])
    print("accuracy:", best_gate["accuracy"])
    print("threshold:", best_gate["threshold"])
    print("use_deberta_n:", best_gate["use_deberta_n"], "/", len(y_true))
    results.append(best_gate)

    # gated blend
    print("\n===== gated blend: alpha*deberta + (1-alpha)*RKT =====")
    best_blend = None
    for th in np.arange(0.30, 0.96, 0.01):
        use_deb = deb_conf >= th
        for alpha in np.arange(0.50, 1.01, 0.05):
            logits = backup_logits.copy()
            logits[use_deb] = alpha * z["deberta"][use_deb] + (1 - alpha) * backup_logits[use_deb]
            f1, acc = eval_pred(y_true, logits)
            if best_blend is None or f1 > best_blend["macro_f1"]:
                best_blend = {
                    "name": "gate_blend_deb_rkt",
                    "macro_f1": f1,
                    "accuracy": acc,
                    "threshold": float(th),
                    "alpha": float(alpha),
                    "use_deberta_n": int(use_deb.sum()),
                    "backup_weights": best3["weights"],
                    "logits": logits,
                }

    print("macro_f1:", best_blend["macro_f1"])
    print("accuracy:", best_blend["accuracy"])
    print("threshold:", best_blend["threshold"])
    print("alpha:", best_blend["alpha"])
    print("use_deberta_n:", best_blend["use_deberta_n"], "/", len(y_true))
    results.append(best_blend)

    results_sorted = sorted(results, key=lambda x: x["macro_f1"], reverse=True)

    print("\n===== SUMMARY =====")
    for r in results_sorted:
        cfg = {k: v for k, v in r.items() if k not in ["logits"]}
        print(json.dumps(cfg, ensure_ascii=False))

    out_dir = Path("ensemble_tune_outputs")
    out_dir.mkdir(exist_ok=True)

    json_results = []
    for r in results_sorted:
        item = {k: v for k, v in r.items() if k not in ["logits"]}
        item["macro_f1"] = float(item["macro_f1"])
        item["accuracy"] = float(item["accuracy"])
        json_results.append(item)

    json.dump(
        {
            "models": {k: str(v) for k, v in MODELS.items()},
            "normalization": "row_zscore_logits",
            "results": json_results,
        },
        open(out_dir / "fold0_ensemble_tune_results.json", "w", encoding="utf-8"),
        indent=2,
        ensure_ascii=False,
    )

    best = results_sorted[0]
    best_logits = best["logits"]
    np.save(out_dir / "best_fold0_logits.npy", best_logits)

    pred = best_logits.argmax(axis=1)
    pred_df = pd.DataFrame({
        "id": ref_ids,
        "label_id": y_true,
        "label": [id2label[int(i)] for i in y_true],
        "prediction": [id2label[int(i)] for i in pred],
    })
    pred_df.to_csv(out_dir / "best_fold0_predictions.csv", index=False)

    print("\n===== BEST DETAIL =====")
    print(classification_report(y_true, pred, target_names=ALL_CLASSES, digits=4))

    print("\nSaved:")
    print(" -", out_dir / "fold0_ensemble_tune_results.json")
    print(" -", out_dir / "best_fold0_logits.npy")
    print(" -", out_dir / "best_fold0_predictions.csv")

if __name__ == "__main__":
    main()
