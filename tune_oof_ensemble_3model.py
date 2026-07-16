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
BASE = ROOT / "team_4model_ensemble_07626" / "ensemble_07626" / "fold0_team"

MODELS = {
    "deberta": BASE / "deberta_base_eda_state4_seed43_oof",
    "kluebert": BASE / "kluebert_mid5_lr15_oof",
    "tabular": BASE / "tabular_histgb_oof",
}

OUT_DIR = BASE / "oof_ensemble_tune_3model"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def row_zscore(x, eps=1e-6):
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + eps)


def load_one(name, d):
    logits_path = d / "val_logits.npy"
    pred_path = d / "val_predictions.csv"

    if not logits_path.exists():
        raise FileNotFoundError(f"{name}: missing {logits_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"{name}: missing {pred_path}")

    logits = np.load(logits_path)
    df = pd.read_csv(pred_path)
    df["id"] = df["id"].astype(str)

    if len(df) != logits.shape[0]:
        raise ValueError(f"{name}: df/logits length mismatch {len(df)} vs {logits.shape}")

    # Normalize true-label column name across OOF files.
    # Our OOF files use "label"; teammate KLUEBERT OOF may use "true_action".
    if "label" not in df.columns:
        if "true_action" in df.columns:
            df["label"] = df["true_action"].astype(str)
        elif "true_label" in df.columns:
            df["label"] = df["true_label"].astype(str)
        elif "action" in df.columns:
            df["label"] = df["action"].astype(str)
        elif "target" in df.columns:
            df["label"] = df["target"].astype(str)
        else:
            raise ValueError(f"{name}: cannot find true label column. columns={df.columns.tolist()}")

    return logits, df


def grid_weights(names, step=0.05):
    units = int(round(1 / step))
    for counts in itertools.product(range(units + 1), repeat=len(names)):
        if sum(counts) != units:
            continue
        yield dict(zip(names, [c / units for c in counts]))


def combine(logits_dict, weights):
    out = None
    for name, w in weights.items():
        if w == 0:
            continue
        part = logits_dict[name] * w
        out = part if out is None else out + part
    return out


def main():
    print("===== loading OOF models =====")

    ref_ids = None
    ref_df = None
    raw_logits = {}

    for name, d in MODELS.items():
        logits, df = load_one(name, d)
        ids = df["id"].astype(str).tolist()

        if ref_ids is None:
            ref_ids = ids
            ref_df = df.copy()
        else:
            if ids != ref_ids:
                if set(ids) == set(ref_ids):
                    print(f"[reorder] {name}: same id set but different order -> reorder")
                    id_to_pos = {id_: i for i, id_ in enumerate(ids)}
                    order = [id_to_pos[id_] for id_ in ref_ids]
                    logits = logits[order]
                    df = df.iloc[order].reset_index(drop=True)
                    assert df["id"].astype(str).tolist() == ref_ids
                else:
                    raise ValueError(
                        f"{name}: id set mismatch. "
                        f"missing={len(set(ref_ids) - set(ids))}, "
                        f"extra={len(set(ids) - set(ref_ids))}"
                    )

        raw_logits[name] = logits
        print(f"{name:8s} logits={logits.shape} path={d}")

    labels = ref_df["label"].astype(str).tolist()
    y_true = np.array([label2id[x] for x in labels], dtype=int)

    z_logits = {name: row_zscore(logits) for name, logits in raw_logits.items()}

    print("\n===== single OOF scores =====")
    for name, logits in z_logits.items():
        pred = logits.argmax(axis=1)
        f1 = f1_score(y_true, pred, average="macro")
        acc = accuracy_score(y_true, pred)
        print(f"{name:8s} macro_f1={f1:.6f} acc={acc:.6f}")

    names = list(MODELS.keys())

    print("\n===== grid search ensemble weights =====")
    best = None
    all_results = []

    for weights in grid_weights(names, step=0.05):
        logits = combine(z_logits, weights)
        pred = logits.argmax(axis=1)

        f1 = f1_score(y_true, pred, average="macro")
        acc = accuracy_score(y_true, pred)

        item = {
            "macro_f1": float(f1),
            "accuracy": float(acc),
            "weights": weights,
        }
        all_results.append(item)

        if best is None or f1 > best["macro_f1"]:
            best = {
                "macro_f1": float(f1),
                "accuracy": float(acc),
                "weights": weights,
                "logits": logits,
                "pred": pred,
            }

    print("\n===== BEST OOF ENSEMBLE =====")
    print("macro_f1:", best["macro_f1"])
    print("accuracy:", best["accuracy"])
    print("weights:", best["weights"])

    print("\n===== classification report =====")
    print(classification_report(y_true, best["pred"], target_names=ALL_CLASSES, digits=4))

    results_df = pd.DataFrame(all_results).sort_values("macro_f1", ascending=False)
    results_df.to_csv(OUT_DIR / "weight_search_results.csv", index=False)

    np.save(OUT_DIR / "best_oof_logits.npy", best["logits"])

    out_df = ref_df.copy()
    out_df["prediction"] = [id2label[int(i)] for i in best["pred"]]
    out_df.to_csv(OUT_DIR / "best_oof_predictions.csv", index=False)

    info = {
        "models": {k: str(v) for k, v in MODELS.items()},
        "normalization": "row_zscore_logits",
        "search_step": 0.05,
        "macro_f1": best["macro_f1"],
        "accuracy": best["accuracy"],
        "weights": best["weights"],
    }

    with open(OUT_DIR / "oof_ensemble_result.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print("\nSaved:")
    print(" -", OUT_DIR / "oof_ensemble_result.json")
    print(" -", OUT_DIR / "weight_search_results.csv")
    print(" -", OUT_DIR / "best_oof_logits.npy")
    print(" -", OUT_DIR / "best_oof_predictions.csv")


if __name__ == "__main__":
    main()
