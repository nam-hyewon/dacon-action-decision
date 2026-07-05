#!/bin/bash
set -e

echo "===== DEBERTA SMALL 5EPOCH FULL JOB START ====="
date
hostname
pwd
nvidia-smi || true

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dacon

cd ~/projects/dacon-action-decision

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

EPOCHS=5 \
BATCH_SIZE=8 \
MAX_LEN=512 \
MODEL_NAME=microsoft/deberta-v3-small \
TEXT_MODE=full \
SAVE_NAME=deberta_small_5ep_len512 \
python src/train_deberta.py

date
echo "===== SAVED MODEL ====="
ls -lh submit/model/deberta_small_5ep_len512