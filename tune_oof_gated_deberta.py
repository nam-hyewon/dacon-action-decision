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

PATHS = {
    "deberta": BASE / "deberta_base_eda_state4_seed43_oof",
    "kluebert": BASE / "kluebert_mid5_lr15_oof",
    "tabular": BASE / "tabular_histgb_oof",
}

OUT_DIR = BASE / "oof_gated_deberta_tune"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def row_zscore(x, eps=1e-6):
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + eps)


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def margin_from_logits(x):
    s = np.sort(x, axis=1)
    return s[:, -1] - s[:, -2]


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
                    print(f"[reorder] {name}")
                    id_to_pos = {id_: i for i, id_ in enumerate(ids)}
                    order = [id_to_pos[i] for i in ref_ids]
                    logits = logits[order]
                    df = df.iloc[order].reset_index(drop=True)
                    assert df["id"].astype(str).tolist() == ref_ids
                else:
                    raise ValueError(
                        f"{name}: id set mismatch "
                        f"missing={len(set(ref_ids)-set(ids))}, extra={len(set(ids)-set(ref_ids))}"
                    )

        out[name] = row_zscore(logits)
        print(f"{name:8s} loaded {logits.shape} from {d}")

    y_true = np.array([label2id[x] for x in ref_df["label"].astype(str).tolist()], dtype=int)
    return out, ref_df, y_true


def eval_logits(logits, y_true):
    pred = logits.argmax(axis=1)
    return f1_score(y_true, pred, average="macro"), accuracy_score(y_true, pred), pred


def combine_kt(k, t, wk):
    wt = 1.0 - wk
    return wk * k + wt * t


def apply_bias(logits, bias):
    return logits + bias.reshape(1, -1)


def grid_kt_weights(step=0.01):
    n = int(round(1 / step))
    for i in range(n + 1):
        wk = i / n
        yield wk


def main():
    z, ref_df, y_true = load_all()

    deb = z["deberta"]
    klue = z["kluebert"]
    tab = z["tabular"]

    # 이전 bias tuning 결과가 있으면 불러오기
    bias_path = BASE / "oof_ensemble_bias_3model" / "best_bias.npy"
    if bias_path.exists():
        full_bias = np.load(bias_path).astype(np.float32)
        print("loaded full 3-model bias:", bias_path)
    else:
        full_bias = np.zeros(len(ALL_CLASSES), dtype=np.float32)
        print("no previous bias found, use zero bias")

    # gated 전용 bias는 일단 동일 bias를 후보로 넣고, zero bias도 같이 비교
    bias_candidates = {
        "zero": np.zeros(len(ALL_CLASSES), dtype=np.float32),
        "prev_full_bias": full_bias,
    }

    print("\n===== single OOF scores =====")
    for name, logits in z.items():
        f1, acc, _ = eval_logits(logits, y_true)
        print(f"{name:8s} macro_f1={f1:.6f} acc={acc:.6f}")

    # 1) DeBERTa 없이 KLUEBERT+tabular best 찾기
    print("\n===== base without DeBERTa: KLUEBERT + tabular =====")
    best_base = None
    for wk in grid_kt_weights(step=0.01):
        base_logits = combine_kt(klue, tab, wk)
        f1, acc, pred = eval_logits(base_logits, y_true)
        if best_base is None or f1 > best_base["macro_f1"]:
            best_base = {
                "wk": wk,
                "wt": 1 - wk,
                "macro_f1": float(f1),
                "accuracy": float(acc),
                "logits": base_logits,
                "pred": pred,
            }

    print("best KT base:")
    print("  kluebert:", best_base["wk"])
    print("  tabular :", best_base["wt"])
    print("  macro_f1:", best_base["macro_f1"])
    print("  accuracy:", best_base["accuracy"])

    base = best_base["logits"]

    # confidence / margin features
    deb_prob = softmax(deb)
    base_prob = softmax(base)

    deb_conf = deb_prob.max(axis=1)
    base_conf = base_prob.max(axis=1)

    deb_margin = margin_from_logits(deb)
    base_margin = margin_from_logits(base)

    deb_pred = deb.argmax(axis=1)
    base_pred = base.argmax(axis=1)

    disagree = deb_pred != base_pred

    # DeBERTa가 base보다 더 자신있는 정도
    conf_diff = deb_conf - base_conf
    margin_diff = deb_margin - base_margin

    # 여러 gate score 후보
    gate_scores = {
        "deb_conf": deb_conf,
        "deb_margin": deb_margin,
        "conf_diff": conf_diff,
        "margin_diff": margin_diff,
    }

    # 개입 비율 후보: 실제 test inference 비용과 직결
    # 0.05 = 전체 샘플 중 5%만 DeBERTa 사용
    target_rates = [
        0.02, 0.03, 0.05, 0.07, 0.10,
        0.15, 0.20, 0.25, 0.30, 0.40,
        0.50, 0.70, 1.00,
    ]

    alphas = [0.25, 0.35, 0.50, 0.65, 0.75, 0.85, 1.00]

    rows = []
    best = None

    def update_best(item, logits, pred):
        nonlocal best
        if best is None or item["macro_f1"] > best["macro_f1"]:
            best = item.copy()
            best["_logits"] = logits
            best["_pred"] = pred

    # 2) base only + bias
    for bias_name, bias in bias_candidates.items():
        logits = apply_bias(base, bias)
        f1, acc, pred = eval_logits(logits, y_true)
        item = {
            "mode": "base_no_deberta",
            "gate_score": None,
            "target_rate": 0.0,
            "actual_rate": 0.0,
            "use_deberta_n": 0,
            "alpha": 0.0,
            "bias_name": bias_name,
            "macro_f1": float(f1),
            "accuracy": float(acc),
            "kt_weights": {"kluebert": best_base["wk"], "tabular": best_base["wt"]},
        }
        rows.append(item)
        update_best(item, logits, pred)

    # 3) full DeBERTa always blend baseline
    # 비교용: DeBERTa 100% 돌렸을 때
    for alpha in alphas:
        for bias_name, bias in bias_candidates.items():
            logits = alpha * deb + (1 - alpha) * base
            logits = apply_bias(logits, bias)
            f1, acc, pred = eval_logits(logits, y_true)
            item = {
                "mode": "always_blend_deberta",
                "gate_score": "all",
                "target_rate": 1.0,
                "actual_rate": 1.0,
                "use_deberta_n": int(len(y_true)),
                "alpha": float(alpha),
                "bias_name": bias_name,
                "macro_f1": float(f1),
                "accuracy": float(acc),
                "kt_weights": {"kluebert": best_base["wk"], "tabular": best_base["wt"]},
            }
            rows.append(item)
            update_best(item, logits, pred)

    # 4) gated: score 상위 rate만 DeBERTa 개입
    for score_name, score in gate_scores.items():
        order_desc = np.argsort(-score)

        for rate in target_rates:
            k = int(round(len(y_true) * rate))
            if k <= 0:
                continue

            use = np.zeros(len(y_true), dtype=bool)
            use[order_desc[:k]] = True

            # disagreement 조건까지 추가한 variant
            use_variants = {
                "top_rate": use,
                "top_rate_and_disagree": use & disagree,
            }

            for variant_name, use_mask in use_variants.items():
                use_n = int(use_mask.sum())
                if use_n == 0:
                    continue

                actual_rate = use_n / len(y_true)

                for alpha in alphas:
                    mixed = base.copy()
                    mixed[use_mask] = alpha * deb[use_mask] + (1 - alpha) * base[use_mask]

                    for bias_name, bias in bias_candidates.items():
                        logits = apply_bias(mixed, bias)
                        f1, acc, pred = eval_logits(logits, y_true)

                        item = {
                            "mode": variant_name,
                            "gate_score": score_name,
                            "target_rate": float(rate),
                            "actual_rate": float(actual_rate),
                            "use_deberta_n": use_n,
                            "alpha": float(alpha),
                            "bias_name": bias_name,
                            "macro_f1": float(f1),
                            "accuracy": float(acc),
                            "kt_weights": {"kluebert": best_base["wk"], "tabular": best_base["wt"]},
                        }
                        rows.append(item)
                        update_best(item, logits, pred)

    # 5) DeBERTa predicted class가 특정 약한 class일 때만 개입하는 variant
    weak_classes = ["read_file", "grep_search", "list_directory", "glob_pattern", "lint_or_typecheck", "web_search"]
    weak_idx = np.array([label2id[c] for c in weak_classes])
    deb_predicts_weak = np.isin(deb_pred, weak_idx)

    for rate in target_rates:
        # weak 중에서도 deb_conf 높은 순서
        candidate = deb_predicts_weak
        idx = np.where(candidate)[0]
        if len(idx) == 0:
            continue
        sorted_idx = idx[np.argsort(-deb_conf[idx])]
        k = min(int(round(len(y_true) * rate)), len(sorted_idx))
        if k <= 0:
            continue

        use = np.zeros(len(y_true), dtype=bool)
        use[sorted_idx[:k]] = True

        for variant_name, use_mask in {
            "weakclass_debconf": use,
            "weakclass_debconf_and_disagree": use & disagree,
        }.items():
            use_n = int(use_mask.sum())
            if use_n == 0:
                continue
            actual_rate = use_n / len(y_true)

            for alpha in alphas:
                mixed = base.copy()
                mixed[use_mask] = alpha * deb[use_mask] + (1 - alpha) * base[use_mask]

                for bias_name, bias in bias_candidates.items():
                    logits = apply_bias(mixed, bias)
                    f1, acc, pred = eval_logits(logits, y_true)
                    item = {
                        "mode": variant_name,
                        "gate_score": "deb_conf_weak_classes",
                        "target_rate": float(rate),
                        "actual_rate": float(actual_rate),
                        "use_deberta_n": use_n,
                        "alpha": float(alpha),
                        "bias_name": bias_name,
                        "macro_f1": float(f1),
                        "accuracy": float(acc),
                        "kt_weights": {"kluebert": best_base["wk"], "tabular": best_base["wt"]},
                    }
                    rows.append(item)
                    update_best(item, logits, pred)

    rows_df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    rows_df.to_csv(OUT_DIR / "gated_search_results.csv", index=False)

    print("\n===== TOP 30 gated results =====")
    cols = [
        "macro_f1", "accuracy", "mode", "gate_score",
        "target_rate", "actual_rate", "use_deberta_n",
        "alpha", "bias_name", "kt_weights"
    ]
    print(rows_df[cols].head(30).to_string(index=False))

    print("\n===== BEST GATED RESULT =====")
    best_print = {k: v for k, v in best.items() if not k.startswith("_")}
    print(json.dumps(best_print, indent=2, ensure_ascii=False))

    print("\n===== classification report =====")
    print(classification_report(y_true, best["_pred"], target_names=ALL_CLASSES, digits=4))

    np.save(OUT_DIR / "best_gated_oof_logits.npy", best["_logits"])

    out_df = ref_df.copy()
    out_df["prediction"] = [id2label[int(i)] for i in best["_pred"]]
    out_df.to_csv(OUT_DIR / "best_gated_oof_predictions.csv", index=False)

    with open(OUT_DIR / "best_gated_result.json", "w", encoding="utf-8") as f:
        json.dump(best_print, f, indent=2, ensure_ascii=False)

    print("\nSaved:")
    print(" -", OUT_DIR / "gated_search_results.csv")
    print(" -", OUT_DIR / "best_gated_result.json")
    print(" -", OUT_DIR / "best_gated_oof_logits.npy")
    print(" -", OUT_DIR / "best_gated_oof_predictions.csv")


if __name__ == "__main__":
    main()
