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

MODEL_PATHS = {
    "deberta": BASE / "deberta_base_eda_state4_seed43_oof",
    "kluebert": BASE / "kluebert_mid5_lr15_oof",
    "roberta": BASE / "roberta_eda_state4_oof",
    "tabular": BASE / "tabular_histgb_oof",
    "xlmr": BASE / "xlmr_base_eda_state_seed43_oof",
}

CANDIDATE_SETS = {
    # 기존 SOTA backbone에 XLMR 추가
    "sota3_plus_xlmr": ["deberta", "kluebert", "tabular", "xlmr"],

    # RoBERTa까지 포함한 전체 5-model
    "all5": ["deberta", "kluebert", "roberta", "tabular", "xlmr"],
}

OUT_DIR = BASE / "oof_ensemble_with_xlmr"
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
            raise ValueError(f"{name}: cannot find label column {df.columns.tolist()}")

    return logits, df


def load_all():
    ref_ids = None
    ref_df = None
    logits_dict = {}

    for name, d in MODEL_PATHS.items():
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
                    order = [id_to_pos[i] for i in ref_ids]
                    logits = logits[order]
                    df = df.iloc[order].reset_index(drop=True)
                    assert df["id"].astype(str).tolist() == ref_ids
                else:
                    raise ValueError(f"{name}: id set mismatch")

        logits_dict[name] = row_zscore(logits).astype(np.float32)
        print(f"{name:8s} loaded {logits.shape}")

    y_true = np.array([label2id[x] for x in ref_df["label"].astype(str).tolist()], dtype=int)
    return logits_dict, ref_df, y_true


def grid_weights(names, step):
    units = int(round(1 / step))

    def rec(prefix, left, k):
        if k == 1:
            yield prefix + [left]
        else:
            for v in range(left + 1):
                yield from rec(prefix + [v], left - v, k - 1)

    for counts in rec([], units, len(names)):
        yield dict(zip(names, [c / units for c in counts]))


def local_weights(center, step=0.01, radius=0.08):
    names = list(center.keys())
    ranges = {}

    for name in names:
        c = float(center[name])
        lo = max(0.0, c - radius)
        hi = min(1.0, c + radius)
        vals = []
        x = round(lo, 2)
        while x <= hi + 1e-9:
            vals.append(round(x, 2))
            x += step
        ranges[name] = vals

    for vals in itertools.product(*[ranges[n] for n in names]):
        if abs(sum(vals) - 1.0) > 1e-9:
            continue
        yield dict(zip(names, vals))


def combine(logits_dict, weights):
    out = None
    for name, w in weights.items():
        if w == 0:
            continue
        part = logits_dict[name] * w
        out = part if out is None else out + part
    return out


def eval_logits(logits, y_true, bias=None):
    if bias is not None:
        logits = logits + bias.reshape(1, -1)
    pred = logits.argmax(axis=1)
    f1 = f1_score(y_true, pred, average="macro")
    acc = accuracy_score(y_true, pred)
    return float(f1), float(acc), pred


def main():
    logits_dict, ref_df, y_true = load_all()

    print("\n===== single OOF scores =====")
    for name, logits in logits_dict.items():
        f1, acc, _ = eval_logits(logits, y_true)
        print(f"{name:8s} macro_f1={f1:.6f} acc={acc:.6f}")

    bias_candidates = {
        "zero": np.zeros(len(ALL_CLASSES), dtype=np.float32),
    }

    b3 = BASE / "oof_ensemble_bias_3model" / "best_bias.npy"
    if b3.exists():
        bias_candidates["prev_3model_bias"] = np.load(b3).astype(np.float32)
        print("\nloaded prev_3model_bias:", b3)

    all_summary = []

    for set_name, names in CANDIDATE_SETS.items():
        print("\n" + "=" * 80)
        print("CANDIDATE SET:", set_name, names)
        print("=" * 80)

        # 1) coarse 0.05
        best_coarse = None
        coarse_rows = []

        print("\n--- coarse search step=0.05 ---")
        for weights in grid_weights(names, step=0.05):
            logits = combine(logits_dict, weights)

            for bias_name, bias in bias_candidates.items():
                f1, acc, pred = eval_logits(logits, y_true, bias)
                item = {
                    "set": set_name,
                    "stage": "coarse",
                    "bias_name": bias_name,
                    "macro_f1": f1,
                    "accuracy": acc,
                    "weights": weights,
                }
                coarse_rows.append(item)

                if best_coarse is None or f1 > best_coarse["macro_f1"]:
                    best_coarse = {
                        **item,
                        "logits": logits,
                        "pred": pred,
                    }

        print("BEST COARSE:")
        print(json.dumps({k: v for k, v in best_coarse.items() if k not in ["logits", "pred"]}, indent=2, ensure_ascii=False))

        # 2) local fine around coarse best weights
        best_fine = None
        fine_rows = []

        print("\n--- local fine search step=0.01 radius=0.08 ---")
        for weights in local_weights(best_coarse["weights"], step=0.01, radius=0.08):
            logits = combine(logits_dict, weights)

            for bias_name, bias in bias_candidates.items():
                f1, acc, pred = eval_logits(logits, y_true, bias)
                item = {
                    "set": set_name,
                    "stage": "fine",
                    "bias_name": bias_name,
                    "macro_f1": f1,
                    "accuracy": acc,
                    "weights": weights,
                }
                fine_rows.append(item)

                if best_fine is None or f1 > best_fine["macro_f1"]:
                    best_fine = {
                        **item,
                        "logits": logits,
                        "pred": pred,
                    }

        print("BEST FINE:")
        best_print = {k: v for k, v in best_fine.items() if k not in ["logits", "pred"]}
        print(json.dumps(best_print, indent=2, ensure_ascii=False))

        print("\nclassification report:")
        print(classification_report(y_true, best_fine["pred"], target_names=ALL_CLASSES, digits=4))

        set_dir = OUT_DIR / set_name
        set_dir.mkdir(exist_ok=True)

        pd.DataFrame(coarse_rows).sort_values("macro_f1", ascending=False).to_csv(set_dir / "coarse_results.csv", index=False)
        pd.DataFrame(fine_rows).sort_values("macro_f1", ascending=False).to_csv(set_dir / "fine_results.csv", index=False)

        np.save(set_dir / "best_logits.npy", best_fine["logits"])
        np.save(set_dir / "best_pred_idx.npy", best_fine["pred"])

        out_df = ref_df.copy()
        out_df["prediction"] = [id2label[int(i)] for i in best_fine["pred"]]
        out_df.to_csv(set_dir / "best_predictions.csv", index=False)

        with open(set_dir / "best_result.json", "w", encoding="utf-8") as f:
            json.dump(best_print, f, indent=2, ensure_ascii=False)

        all_summary.append(best_print)

    summary_df = pd.DataFrame(all_summary).sort_values("macro_f1", ascending=False)
    summary_df.to_csv(OUT_DIR / "summary.csv", index=False)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    print("\nSaved:", OUT_DIR)


if __name__ == "__main__":
    main()
