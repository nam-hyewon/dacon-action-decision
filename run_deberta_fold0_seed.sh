#!/bin/bash
set -e

echo "===== DEBERTA BASE EDA_STATE4 FOLD0 START ====="
date
hostname
pwd
nvidia-smi || true

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dacon

cd ~/projects/dacon-action-decision

SEED=${SEED} \
MODEL_NAME=microsoft/deberta-v3-base \
TEXT_MODE=eda_state \
SAVE_NAME=deberta_base_eda_state4_fold0_seed${SEED} \
EPOCHS=4 \
BATCH_SIZE=4 \
MAX_LEN=512 \
LR=1e-5 \
WEIGHT_DECAY=0.01 \
WARMUP_RATIO=0.06 \
python src/train_roberta_fold0.py

date
echo "===== SAVED MODEL ====="
ls -lh submit/model/deberta_base_eda_state4_fold0_seed${SEED}
