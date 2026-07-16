from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

MODEL_DIRS = [
    ROOT / "submit/model/deberta_base_eda_state4",
    ROOT / "submit/model/mdeberta_base_eda_state4",
    ROOT / "submit/model/mdeberta_base_4ep_full",
    ROOT / "submit/model/xlmr_base_router4",
    ROOT / "submit/model/deberta_group_full_router_base",
    ROOT / "submit/model/deberta_group_v3_full_router_mdeberta",
    ROOT / "submit/model/deberta_small_eda_ce_full_lr2e5",
    ROOT / "submit/model/deberta_small_eda_kd_soup_a01_a02_a03",
]

def find_cols(df):
    id_col = "id"

    true_col = None
    pred_col = None

    for c in ["true_action", "true", "label", "true_label", "target"]:
        if c in df.columns:
            true_col = c
            break

    for c in ["pred_action", "pred", "prediction", "pred_label"]:
        if c in df.columns:
            pred_col = c
            break

    if true_col is None:
        raise ValueError(f"Cannot find true label column in {df.columns.tolist()}")
    if pred_col is None:
        raise ValueError(f"Cannot find pred column in {df.columns.tolist()}")

    return id_col, true_col, pred_col

def main():
    rows = []
    ref_ids = None
    ref_labels = None

    for d in MODEL_DIRS:
        val_csv = d / "val_predictions.csv"
        val_logits = d / "val_logits.npy"

        if not val_csv.exists() or not val_logits.exists():
            rows.append({
                "model_dir": str(d),
                "exists": False,
                "n_rows": None,
                "same_ids": None,
                "same_order": None,
                "same_labels": None,
                "logits_shape": None,
            })
            continue

        df = pd.read_csv(val_csv)
        logits = np.load(val_logits)

        id_col, true_col, pred_col = find_cols(df)

        ids = df[id_col].astype(str).tolist()
        labels = df[true_col].astype(str).tolist()

        if ref_ids is None:
            ref_ids = ids
            ref_labels = labels
            same_ids = True
            same_order = True
            same_labels = True
        else:
            same_ids = set(ids) == set(ref_ids)
            same_order = ids == ref_ids
            same_labels = labels == ref_labels

        rows.append({
            "model_dir": str(d),
            "exists": True,
            "n_rows": len(df),
            "same_ids": same_ids,
            "same_order": same_order,
            "same_labels": same_labels,
            "logits_shape": str(logits.shape),
            "true_col": true_col,
            "pred_col": pred_col,
        })

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))

    out_dir = ROOT / "stacking_data"
    out_dir.mkdir(exist_ok=True)
    out.to_csv(out_dir / "logits_registry_check.csv", index=False)
    print("[SAVED]", out_dir / "logits_registry_check.csv")

if __name__ == "__main__":
    main()
