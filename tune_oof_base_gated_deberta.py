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
BASE_DIR = ROOT / "team_4model_ensemble_07626" / "ensemble_07626" / "fold0_team"

PATHS = {
    "deberta": BASE_DIR / "deberta_base_eda_state4_seed43_oof",
    "kluebert": BASE_DIR / "kluebert_mid5_lr15_oof",
    "tabular": BASE_DIR / "tabular_histgb_oof",
}

OUT_DIR = BASE_DIR / "oof_base_gated_deberta"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def row_zscore(x, eps=1e-6):
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + eps)


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def get_margin(x):
    s = np.sort(x, axis=1)
    return s[:, -1] - s[:, -2]


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
            raise ValueError(f"{name}: cannot find label column: {df.columns.tolist()}")

    if len(df) != logits.shape[0]:
        raise ValueError(f"{name}: df/logits mismatch {len(df)} vs {logits.shape}")

    return logits, df


def load_all():
    ref_ids = None
    ref_df = None
    out = {}

    for name, d in PATHS.items():
        logits, df = load_one(name, d)
        ids = df["id"].astype(str).tolist()

        if ref_ids is None:
            ref_ids = ids
            ref_df = df.copy()
        else:
            if ids != ref_ids:
                if set(ids) == set(ref_ids):
                    print(f"[reorder] {name}: same id set but different order")
                    id_to_pos = {id_: i for i, id_ in enumerate(ids)}
                    order = [id_to_pos[i] for i in ref_ids]
                    logits = logits[order]
                    df = df.iloc[order].reset_index(drop=True)
                    assert df["id"].astype(str).tolist() == ref_ids
                else:
                    raise ValueError(
                        f"{name}: id set mismatch. "
                        f"missing={len(set(ref_ids) - set(ids))}, "
                        f"extra={len(set(ids) - set(ref_ids))}"
                    )

        out[name] = row_zscore(logits)
        print(f"{name:8s} loaded {logits.shape} from {d}")

    y_true = np.array([label2id[x] for x in ref_df["label"].astype(str).tolist()], dtype=int)
    return out, ref_df, y_true


def eval_logits(logits, y_true):
    pred = logits.argmax(axis=1)
    f1 = f1_score(y_true, pred, average="macro")
    acc = accuracy_score(y_true, pred)
    return f1, acc, pred


def main():
    z, ref_df, y_true = load_all()

    deb = z["deberta"]
    klue = z["kluebert"]
    tab = z["tabular"]

    # 이전에 찾은 class bias 사용
    bias_path = BASE_DIR / "oof_ensemble_bias_3model" / "best_bias.npy"
    if bias_path.exists():
        prev_bias = np.load(bias_path).astype(np.float32)
        print("loaded previous class bias:", bias_path)
    else:
        prev_bias = np.zeros(len(ALL_CLASSES), dtype=np.float32)
        print("previous class bias not found; use zero bias")

    bias_candidates = {
        "zero": np.zeros(len(ALL_CLASSES), dtype=np.float32),
        "prev_full_bias": prev_bias,
    }

    print("\n===== single OOF scores =====")
    for name, logits in z.items():
        f1, acc, _ = eval_logits(logits, y_true)
        print(f"{name:8s} macro_f1={f1:.6f} acc={acc:.6f}")

    # 1. DeBERTa 없이 KLUEBERT + tabular base weight 탐색
    best_base = None
    for i in range(101):
        wk = i / 100
        wt = 1 - wk
        base_logits = wk * klue + wt * tab
        f1, acc, pred = eval_logits(base_logits, y_true)
        if best_base is None or f1 > best_base["macro_f1"]:
            best_base = {
                "kluebert_weight": wk,
                "tabular_weight": wt,
                "macro_f1": float(f1),
                "accuracy": float(acc),
                "logits": base_logits,
                "pred": pred,
            }

    base = best_base["logits"]

    print("\n===== base model: KLUEBERT + tabular only =====")
    print("kluebert_weight:", best_base["kluebert_weight"])
    print("tabular_weight :", best_base["tabular_weight"])
    print("macro_f1:", best_base["macro_f1"])
    print("accuracy:", best_base["accuracy"])

    base_prob = softmax(base)
    base_conf = base_prob.max(axis=1)
    base_margin = get_margin(base)
    base_pred = base.argmax(axis=1)

    file4_classes = ["read_file", "grep_search", "list_directory", "glob_pattern"]
    weak_classes = [
        "read_file", "grep_search", "list_directory", "glob_pattern",
        "lint_or_typecheck", "web_search", "ask_user", "plan_task",
    ]

    file4_idx = np.array([label2id[c] for c in file4_classes])
    weak_idx = np.array([label2id[c] for c in weak_classes])

    base_pred_file4 = np.isin(base_pred, file4_idx)
    base_pred_weak = np.isin(base_pred, weak_idx)

    # DeBERTa 없이 계산 가능한 gate score만 사용
    # score가 클수록 "불안한 샘플"이 되게 부호를 맞춤
    gate_scores = {
        "low_base_conf": -base_conf,
        "low_base_margin": -base_margin,
    }

    # 실제 DeBERTa 사용 비율 후보
    target_rates = [
        0.02, 0.03, 0.05, 0.07, 0.10,
        0.12, 0.15, 0.20, 0.25, 0.30,
        0.40, 0.50,
    ]

    # 선택된 샘플에서 DeBERTa를 얼마나 섞을지
    alphas = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.65, 0.80, 1.00]

    rows = []
    best = None

    def update_best(item, logits, pred):
        nonlocal best
        if best is None or item["macro_f1"] > best["macro_f1"]:
            best = item.copy()
            best["_logits"] = logits
            best["_pred"] = pred

    # 2. base only baseline 저장
    for bias_name, bias in bias_candidates.items():
        logits = base + bias.reshape(1, -1)
        f1, acc, pred = eval_logits(logits, y_true)
        item = {
            "mode": "base_only",
            "gate": None,
            "target_rate": 0.0,
            "actual_rate": 0.0,
            "use_deberta_n": 0,
            "alpha": 0.0,
            "bias_name": bias_name,
            "macro_f1": float(f1),
            "accuracy": float(acc),
            "base_weights": {
                "kluebert": best_base["kluebert_weight"],
                "tabular": best_base["tabular_weight"],
            },
        }
        rows.append(item)
        update_best(item, logits, pred)

    # 3. full 3-model 참고용: DeBERTa always blend
    for alpha in alphas:
        for bias_name, bias in bias_candidates.items():
            mixed = alpha * deb + (1 - alpha) * base
            logits = mixed + bias.reshape(1, -1)
            f1, acc, pred = eval_logits(logits, y_true)
            item = {
                "mode": "always_blend_deberta_reference",
                "gate": "all",
                "target_rate": 1.0,
                "actual_rate": 1.0,
                "use_deberta_n": len(y_true),
                "alpha": float(alpha),
                "bias_name": bias_name,
                "macro_f1": float(f1),
                "accuracy": float(acc),
                "base_weights": {
                    "kluebert": best_base["kluebert_weight"],
                    "tabular": best_base["tabular_weight"],
                },
            }
            rows.append(item)
            update_best(item, logits, pred)

    # 4. 실제 시간절약용 gated search
    for gate_name, score in gate_scores.items():
        order = np.argsort(-score)  # 불안한 샘플부터

        for rate in target_rates:
            k = int(round(len(y_true) * rate))
            if k <= 0:
                continue

            use = np.zeros(len(y_true), dtype=bool)
            use[order[:k]] = True

            variants = {
                gate_name: use,
                gate_name + "_file4only": use & base_pred_file4,
                gate_name + "_weakonly": use & base_pred_weak,
            }

            for mode, use_mask in variants.items():
                use_n = int(use_mask.sum())
                if use_n == 0:
                    continue

                actual_rate = use_n / len(y_true)

                for alpha in alphas:
                    mixed = base.copy()
                    mixed[use_mask] = alpha * deb[use_mask] + (1 - alpha) * base[use_mask]

                    for bias_name, bias in bias_candidates.items():
                        logits = mixed + bias.reshape(1, -1)
                        f1, acc, pred = eval_logits(logits, y_true)

                        item = {
                            "mode": mode,
                            "gate": gate_name,
                            "target_rate": float(rate),
                            "actual_rate": float(actual_rate),
                            "use_deberta_n": use_n,
                            "alpha": float(alpha),
                            "bias_name": bias_name,
                            "macro_f1": float(f1),
                            "accuracy": float(acc),
                            "base_weights": {
                                "kluebert": best_base["kluebert_weight"],
                                "tabular": best_base["tabular_weight"],
                            },
                        }
                        rows.append(item)
                        update_best(item, logits, pred)

    rows_df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    rows_df.to_csv(OUT_DIR / "base_gated_search_results.csv", index=False)

    print("\n===== TOP 50 RESULTS =====")
    cols = [
        "macro_f1", "accuracy", "mode", "gate",
        "target_rate", "actual_rate", "use_deberta_n",
        "alpha", "bias_name", "base_weights",
    ]
    print(rows_df[cols].head(50).to_string(index=False))

    # 사용률별 최고 결과 요약
    print("\n===== BEST BY DEBERTA USAGE BUCKET =====")
    for max_rate in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00]:
        sub = rows_df[rows_df["actual_rate"] <= max_rate]
        if len(sub) == 0:
            continue
        r = sub.iloc[0]
        print(
            f"<= {max_rate:.2f}: "
            f"f1={r['macro_f1']:.6f}, acc={r['accuracy']:.6f}, "
            f"actual_rate={r['actual_rate']:.4f}, use_n={int(r['use_deberta_n'])}, "
            f"mode={r['mode']}, alpha={r['alpha']}, bias={r['bias_name']}"
        )

    best_print = {k: v for k, v in best.items() if not k.startswith("_")}

    print("\n===== BEST OVERALL =====")
    print(json.dumps(best_print, indent=2, ensure_ascii=False))

    print("\n===== classification report for best overall =====")
    print(classification_report(y_true, best["_pred"], target_names=ALL_CLASSES, digits=4))

    np.save(OUT_DIR / "best_gated_logits.npy", best["_logits"])

    out_df = ref_df.copy()
    out_df["prediction"] = [id2label[int(i)] for i in best["_pred"]]
    out_df.to_csv(OUT_DIR / "best_gated_predictions.csv", index=False)

    with open(OUT_DIR / "best_gated_result.json", "w", encoding="utf-8") as f:
        json.dump(best_print, f, indent=2, ensure_ascii=False)

    # usage bucket별 best도 저장
    bucket_rows = []
    for max_rate in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00]:
        sub = rows_df[rows_df["actual_rate"] <= max_rate]
        if len(sub) == 0:
            continue
        bucket_rows.append(sub.iloc[0].to_dict())
    pd.DataFrame(bucket_rows).to_csv(OUT_DIR / "best_by_usage_bucket.csv", index=False)

    print("\nSaved:")
    print(" -", OUT_DIR / "base_gated_search_results.csv")
    print(" -", OUT_DIR / "best_by_usage_bucket.csv")
    print(" -", OUT_DIR / "best_gated_result.json")
    print(" -", OUT_DIR / "best_gated_logits.npy")
    print(" -", OUT_DIR / "best_gated_predictions.csv")


if __name__ == "__main__":
    main()
