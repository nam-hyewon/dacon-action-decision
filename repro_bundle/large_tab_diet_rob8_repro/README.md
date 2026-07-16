# large + tab + diet + rob8 fold0 eval repro

Run from this directory or after copying files into repo root:

```bash
python scripts/eval_large_tab_diet_rob8_recipe.py
```

Expected:

```text
before_class_bias.macro_f1 = 0.7601265701149389
after_class_bias.macro_f1  = 0.7646781949726907
```

The package contains only validation artifacts required for score reproduction:
`val_logits.npy`, `val_predictions.csv`, `val_info.json`, and labels where available.
It does not contain full model weights for submission inference.
