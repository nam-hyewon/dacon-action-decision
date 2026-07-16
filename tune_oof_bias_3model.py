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

WEIGHTS = {
    "deberta": 0.29,
    "kluebert": 0.27,
    "tabular": 0.44,
}

OUT_DIR = BASE / "oof_ensemble_bias_3model"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def row_zscore(x, eps=1e-6):
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + eps)


def load_one(name, d):
    logits = np.load(d / "val_logits.npy")
    df = pd.read_csv(d / "val_predictions.csv")
    df["id"] = df["id"].astype(str)

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
            raise ValueError(f"{name}: cannot find true label column: {df.columns.tolist()}")

    if len(df) != logits.shape[0]:
        raise ValueError(f"{name}: df/logits mismatch {len(df)} vs {logits.shape}")

    return logits, df


def load_weighted_logits():
    ref_ids = None
    ref_df = None
    combined = None

    for name, d in MODELS.items():
        logits, df = load_one(name, d)
        ids = df["id"].astype(str).tolist()

        if ref_ids is None:
            ref_ids = ids
            ref_df = df.copy()
        else:
            if ids != ref_ids:
                if set(ids) == set(ref_ids):
                    print(f"[reorder] {name}")
                    id_to_pos = {id_: i for i, id_ in enumerate(ids)}
                    order = [id_to_pos[id_] for id_ in ref_ids]
                    logits = logits[order]
                    df = df.iloc[order].reset_index(drop=True)
                    assert df["id"].astype(str).tolist() == ref_ids
                else:
                    raise ValueError(f"{name}: id set mismatch")

        z = row_zscore(logits)
        part = WEIGHTS[name] * z
        combined = part if combined is None else combined + part

        print(f"{name:8s} loaded {logits.shape}, weight={WEIGHTS[name]}")

    y_true = np.array([label2id[x] for x in ref_df["label"].astype(str).tolist()], dtype=int)
    return combined, y_true, ref_df


def eval_bias(base_logits, y_true, bias):
    logits = base_logits + bias.reshape(1, -1)
    pred = logits.argmax(axis=1)
    f1 = f1_score(y_true, pred, average="macro")
    acc = accuracy_score(y_true, pred)
    return f1, acc, pred


def main():
    base_logits, y_true, ref_df = load_weighted_logits()

    zero_bias = np.zeros(len(ALL_CLASSES), dtype=np.float32)
    base_f1, base_acc, base_pred = eval_bias(base_logits, y_true, zero_bias)

    print("\n===== BASE weighted ensemble =====")
    print("macro_f1:", base_f1)
    print("accuracy:", base_acc)

    # 이전에 좋았던 방향 + 약한 class 위주 후보
    candidate_biases = []

    # old fold0 bias
    b = np.zeros(len(ALL_CLASSES), dtype=np.float32)
    old = {
        "list_directory": 0.15,
        "glob_pattern": 0.15,
        "run_bash": -0.05,
        "lint_or_typecheck": 0.05,
        "web_search": -0.10,
    }
    for k, v in old.items():
        b[label2id[k]] = v
    candidate_biases.append(("old_fold0_bias", b))

    # weak file4 boost variants
    for rf in [0.00, 0.03, 0.05, 0.08, 0.10]:
        for gs in [0.00, 0.03, 0.05, 0.08]:
            for ld in [0.00, 0.05, 0.08, 0.10, 0.12, 0.15]:
                for gp in [0.00, 0.03, 0.05, 0.08, 0.10]:
                    b = np.zeros(len(ALL_CLASSES), dtype=np.float32)
                    b[label2id["read_file"]] = rf
                    b[label2id["grep_search"]] = gs
                    b[label2id["list_directory"]] = ld
                    b[label2id["glob_pattern"]] = gp
                    candidate_biases.append((f"file4_rf{rf}_gs{gs}_ld{ld}_gp{gp}", b))

    best_name = "zero"
    best_bias = zero_bias.copy()
    best_f1, best_acc, best_pred = base_f1, base_acc, base_pred

    rows = []
    for name, bias in candidate_biases:
        f1, acc, pred = eval_bias(base_logits, y_true, bias)
        rows.append({
            "stage": "manual_grid",
            "name": name,
            "macro_f1": float(f1),
            "accuracy": float(acc),
            "gain": float(f1 - base_f1),
            "bias": bias.tolist(),
        })
        if f1 > best_f1:
            best_name, best_bias, best_f1, best_acc, best_pred = name, bias.copy(), f1, acc, pred

    print("\n===== BEST after manual grid =====")
    print("name:", best_name)
    print("macro_f1:", best_f1, "gain:", best_f1 - base_f1)
    print("accuracy:", best_acc)
    print("bias:")
    for c, v in zip(ALL_CLASSES, best_bias):
        if abs(v) > 1e-9:
            print(f"  {c:18s} {v:+.3f}")

    # Greedy coordinate search around current best
    print("\n===== greedy coordinate search =====")
    cur_bias = best_bias.copy()
    cur_f1, cur_acc, cur_pred = best_f1, best_acc, best_pred

    # 너무 과도한 bias 방지
    max_abs = 0.25
    deltas = [0.02, -0.02, 0.01, -0.01]

    improved = True
    iter_no = 0
    while improved and iter_no < 20:
        improved = False
        iter_no += 1
        best_step = None

        for cls_idx, cls_name in enumerate(ALL_CLASSES):
            for delta in deltas:
                nb = cur_bias.copy()
                nb[cls_idx] += delta
                if abs(nb[cls_idx]) > max_abs:
                    continue

                f1, acc, pred = eval_bias(base_logits, y_true, nb)
                if f1 > cur_f1 + 1e-8:
                    best_step = (cls_idx, cls_name, delta, nb, f1, acc, pred)

        if best_step is not None:
            cls_idx, cls_name, delta, nb, f1, acc, pred = best_step
            print(f"iter {iter_no:02d}: {cls_name} {delta:+.3f} -> f1 {cur_f1:.6f} -> {f1:.6f}")
            cur_bias, cur_f1, cur_acc, cur_pred = nb, f1, acc, pred
            improved = True

    final_bias = cur_bias
    final_f1, final_acc, final_pred = cur_f1, cur_acc, cur_pred

    print("\n===== FINAL BIAS RESULT =====")
    print("base_macro_f1:", base_f1)
    print("final_macro_f1:", final_f1)
    print("gain:", final_f1 - base_f1)
    print("final_accuracy:", final_acc)
    print("bias:")
    for c, v in zip(ALL_CLASSES, final_bias):
        if abs(v) > 1e-9:
            print(f"  {c:18s} {v:+.3f}")

    print("\n===== classification report =====")
    print(classification_report(y_true, final_pred, target_names=ALL_CLASSES, digits=4))

    # save
    rows_df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    rows_df.to_csv(OUT_DIR / "manual_bias_search_results.csv", index=False)

    np.save(OUT_DIR / "best_bias.npy", final_bias)
    np.save(OUT_DIR / "biased_oof_logits.npy", base_logits + final_bias.reshape(1, -1))

    out_df = ref_df.copy()
    out_df["prediction"] = [id2label[int(i)] for i in final_pred]
    out_df.to_csv(OUT_DIR / "biased_oof_predictions.csv", index=False)

    info = {
        "base_weights": WEIGHTS,
        "base_macro_f1": float(base_f1),
        "base_accuracy": float(base_acc),
        "final_macro_f1": float(final_f1),
        "final_accuracy": float(final_acc),
        "gain": float(final_f1 - base_f1),
        "bias": {c: float(v) for c, v in zip(ALL_CLASSES, final_bias)},
        "classes": ALL_CLASSES,
        "note": "final logits = weighted row-zscore ensemble logits + class bias",
    }

    with open(OUT_DIR / "bias_result.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print("\nSaved:")
    print(" -", OUT_DIR / "bias_result.json")
    print(" -", OUT_DIR / "best_bias.npy")
    print(" -", OUT_DIR / "manual_bias_search_results.csv")
    print(" -", OUT_DIR / "biased_oof_logits.npy")
    print(" -", OUT_DIR / "biased_oof_predictions.csv")


if __name__ == "__main__":
    main()
