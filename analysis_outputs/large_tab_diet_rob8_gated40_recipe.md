# large + tab + diet + rob8 gated40 recipe

Local fold0 Macro-F1: `0.7646781949726907`

Observed LB after submission: `0.7519523375`

Observed runtime: `9m38s`

This score includes model weights, temperature scaling, and class bias.

## Models

| name | path | fp16 path | weight | temperature |
| --- | --- | --- | ---: | ---: |
| large | `submit/model/klue_roberta_large_eda_state_fold0` | `submit/model/klue_roberta_large_eda_state_fold0_fp16` | `0.24114308710234722` | `1.0` |
| tab | `submit/model/tabular_histgb_fold0` |  | `0.4438507372276068` | `1.0` |
| diet | `submit/model/kluebert_diet_fold0` | `submit/model/kluebert_diet_fold0_fp16` | `0.21685297669057865` | `1.0` |
| rob8 | `submit/model/roberta_eda_state8_lr15_fold0` | `submit/model/roberta_eda_state8_lr15_fold0_fp16` | `0.09815319897946724` | `0.85` |

Submission uses large as a gated member:

```text
large_gate_ratio = 0.4
```

## Class Bias

```python
CLASS_BIAS = {
    "read_file": 0.0,
    "grep_search": -0.3000011444091797,
    "list_directory": 0.39999818801879883,
    "glob_pattern": 0.6499979496002197,
    "edit_file": -0.3000011444091797,
    "write_file": 0.0,
    "apply_patch": 0.0,
    "run_bash": -0.000001430511474609375,
    "run_tests": -0.2500011920928955,
    "lint_or_typecheck": -0.35000109672546387,
    "ask_user": -0.2500011920928955,
    "plan_task": -0.10000133514404297,
    "web_search": 0.4999980926513672,
    "respond_only": 1.3499972820281982,
}
```

## Formula

```python
score = (
    0.24114308710234722 * (large_logits / 1.0)
  + 0.4438507372276068  * (tab_logits / 1.0)
  + 0.21685297669057865 * (diet_logits / 1.0)
  + 0.09815319897946724 * (rob8_logits / 0.85)
)

score += class_bias_vector
pred = score.argmax(axis=1)
```

Use the saved `val_logits.npy` values directly. Do not apply an extra softmax before averaging.
