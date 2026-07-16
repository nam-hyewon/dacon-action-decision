#!/bin/bash
set -e

echo "===== MDEBERTA BASE EDA_STATE4 START ====="
date
hostname
pwd
nvidia-smi || true

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dacon

cd ~/projects/dacon-action-decision

SEED=42 \
MODEL_NAME=microsoft/mdeberta-v3-base \
TEXT_MODE=eda_state \
SAVE_NAME=mdeberta_base_eda_state4 \
EPOCHS=4 \
BATCH_SIZE=4 \
MAX_LEN=512 \
LR=1e-5 \
WEIGHT_DECAY=0.01 \
WARMUP_RATIO=0.06 \
python src/train_roberta.py

date
echo "===== SAVED MODEL ====="
ls -lh submit/model/mdeberta_base_eda_state4
