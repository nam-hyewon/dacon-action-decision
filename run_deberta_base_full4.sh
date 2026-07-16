#!/bin/bash
set -e

echo "===== DEBERTA BASE FULL4 START ====="
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
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

SEED=42 \
MODEL_NAME=microsoft/deberta-v3-base \
TEXT_MODE=full \
SAVE_NAME=deberta_base_4ep_full \
EPOCHS=4 \
BATCH_SIZE=4 \
MAX_LEN=512 \
LR=1e-5 \
WEIGHT_DECAY=0.01 \
WARMUP_RATIO=0.06 \
USE_CLASS_WEIGHT=0 \
python src/train_deberta.py

date
echo "===== SAVED MODEL ====="
ls -lh submit/model/deberta_base_4ep_full
