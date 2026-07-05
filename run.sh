#!/bin/bash
set -e

echo "===== DEBERTA SMALL 4EPOCH FULL JOB START ====="
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

USE_CLASS_WEIGHT=1 \
EPOCHS=3 \
BATCH_SIZE=8 \
MAX_LEN=512 \
MODEL_NAME=microsoft/deberta-v3-small \
TEXT_MODE=full \
SAVE_NAME=deberta_small_3ep_full_weighted \
python src/train_deberta.py

date
echo "===== SAVED MODEL ====="
ls -lh submit/model/deberta_small_4ep_len512