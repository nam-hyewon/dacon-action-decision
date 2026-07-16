from pathlib import Path

import numpy as np
import pandas as pd


ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]

FILE4 = [
    "read_file",
    "grep_search",
    "list_directory",
    "glob_pattern",
]

LABEL2ID = {c: i for i, c in enumerate(ALL_CLASSES)}

BASE_DIR = (
    Path("team_4model_ensemble_07626")
    / "ensemble_07626"
    / "fold0_team"
)

KLUE_DIR = BASE_DIR / "kluebert_mid5_lr15_oof"
TAB_DIR = BASE_DIR / "tabular_histgb_oof"

OUT_DIR = BASE_DIR / "file4_hard_negative_ids"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def row_zscore(x, eps=1e-6):
    x = x.astype(np.float32)
    return (
        x - x.mean(axis=1, keepdims=True)
    ) / (
        x.std(axis=1, keepdims=True) + eps
    )


def load_model(path):
    logits = np.load(path / "val_logits.npy")
    df = pd.read_csv(path / "val_predictions.csv")
    df["id"] = df["id"].astype(str)

    if "true_action" in df.columns:
        labels = df["true_action"].astype(str)
    elif "label" in df.columns:
        labels = df["label"].astype(str)
    else:
        raise ValueError(df.columns.tolist())

    return logits, df, labels


klue, ref_df, labels = load_model(KLUE_DIR)
tab, tab_df, _ = load_model(TAB_DIR)

ref_ids = ref_df["id"].astype(str).tolist()
tab_ids = tab_df["id"].astype(str).tolist()

if tab_ids != ref_ids:
    if set(tab_ids) != set(ref_ids):
        raise ValueError("KLUE/TAB ID set mismatch")

    pos = {sid: i for i, sid in enumerate(tab_ids)}
    order = [pos[sid] for sid in ref_ids]
    tab = tab[order]
    tab_df = tab_df.iloc[order].reset_index(drop=True)

klue = row_zscore(klue)
tab = row_zscore(tab)

# all5 OOF에서 이전에 찾은 best KLUE+TAB weights
base = 0.51 * klue + 0.49 * tab

pred_id = base.argmax(axis=1)
pred_action = np.array(
    [ALL_CLASSES[i] for i in pred_id],
    dtype=object,
)

true_action = labels.to_numpy(dtype=object)

true_file4 = np.isin(true_action, FILE4)
pred_file4 = np.isin(pred_action, FILE4)

# 핵심 hard negative:
# true와 pred 모두 FILE4이면서 서로 틀린 경우
hard_mask = (
    true_file4
    & pred_file4
    & (true_action != pred_action)
)

out = pd.DataFrame({
    "id": ref_ids,
    "true_action": true_action,
    "pred_action": pred_action,
})

hard_df = out.loc[hard_mask].copy()

hard_df.to_csv(
    OUT_DIR / "file4_hard_ids.csv",
    index=False,
)

with open(
    OUT_DIR / "file4_hard_ids.txt",
    "w",
    encoding="utf-8",
) as f:
    for sid in hard_df["id"]:
        f.write(str(sid) + "\n")

print("all OOF n:", len(out))
print("true FILE4 n:", int(true_file4.sum()))
print("hard FILE4 n:", len(hard_df))

print("\n===== CONFUSIONS =====")
print(
    hard_df.groupby(
        ["true_action", "pred_action"]
    ).size().sort_values(ascending=False).head(20)
)

print("\n===== SAVED =====")
print(OUT_DIR / "file4_hard_ids.csv")
print(OUT_DIR / "file4_hard_ids.txt")
