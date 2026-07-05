# Experiments

## exp00_roberta_baseline

- model: `klue/roberta-base`
- train script: `src/train_roberta.py`
- inference script: `submit/script_roberta.py`
- model dir: `submit/model/roberta`
- feature: current `build_text()`
- max_len: 512
- batch_size: 16
- epochs: 3
- lr: 2e-5
- history: last 8 turns
- prompt_again: yes
- args: yes
- result_summary: yes

Validation Macro-F1:

- epoch 1: 0.5764
- epoch 2: 0.6400
- epoch 3: 0.6550

Notes:

- Local validation score, not DACON leaderboard score.
- Current best baseline for RoBERTa experiments: epoch 3, Macro-F1 0.6550.
