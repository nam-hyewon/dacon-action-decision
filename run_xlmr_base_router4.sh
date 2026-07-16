#!/bin/bash
set -e

echo "===== XLM-ROBERTA BASE ROUTER4 START ====="
date
hostname
pwd
nvidia-smi || true

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dacon

cd ~/projects/dacon-action-decision

SEED=42 \
MODEL_NAME=xlm-roberta-base \
TEXT_MODE=router \
SAVE_NAME=xlmr_base_router4 \
EPOCHS=4 \
BATCH_SIZE=8 \
MAX_LEN=512 \
LR=1e-5 \
WEIGHT_DECAY=0.01 \
WARMUP_RATIO=0.06 \
USE_CLASS_WEIGHT=0 \
python src/train_deberta.py

date
echo "===== SAVED MODEL ====="
ls -lh submit/model/xlmr_base_router4
