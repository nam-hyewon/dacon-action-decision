import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]

CLASS_BIAS = np.array([
    0.00, 0.00, 0.15, 0.15,
    0.00, 0.00, 0.00, -0.05,
    0.00, 0.05, 0.00, 0.00,
    -0.10, 0.00,
], dtype=np.float32)

def row_zscore(x, eps=1e-6):
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + eps)

def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

df = pd.read_csv("roberta_eda_state4/val_predictions.csv")
y = df["true_id"].values.astype(int)

deberta = np.load("deberta_base_eda_state4/val_logits.npy")
tabular = np.load("tabular_histgb_row/val_logits.npy")
kluebert = np.load("kluebert_mid5_lr15_focal1/val_logits.npy")
roberta = np.load("roberta_eda_state4/val_logits.npy")

z_d = row_zscore(deberta)
z_t = row_zscore(tabular)
z_k = row_zscore(kluebert)
z_r = row_zscore(roberta)

# full 4-model
full = 0.30*z_d + 0.30*z_t + 0.15*z_k + 0.25*z_r + CLASS_BIAS[None, :]

# base 3-model: DeBERTa 뺀 나머지 weight를 normalize
# 원래 남은 weight 합 = 0.70
base3 = (
    (0.30/0.70)*z_t +
    (0.15/0.70)*z_k +
    (0.25/0.70)*z_r +
    CLASS_BIAS[None, :]
)

base3_prob = softmax(base3)
base3_conf = base3_prob.max(axis=1)
order = np.argsort(base3_conf)  # 낮은 confidence부터

print("3-model no deberta:", f1_score(y, base3.argmax(1), average="macro"), "acc", (base3.argmax(1)==y).mean())
print("full 4-model     :", f1_score(y, full.argmax(1), average="macro"), "acc", (full.argmax(1)==y).mean())
print()

print("=== gated DeBERTa by low confidence of 3-model ===")
print("ratio | f1 | acc | num_deberta")
for ratio in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80]:
    k = int(len(y) * ratio)
    use_d = np.zeros(len(y), dtype=bool)
    use_d[order[:k]] = True

    gated = base3.copy()
    gated[use_d] = full[use_d]

    pred = gated.argmax(1)
    print(f"{ratio:4.2f} | {f1_score(y, pred, average='macro'):.6f} | {(pred==y).mean():.6f} | {k}")

