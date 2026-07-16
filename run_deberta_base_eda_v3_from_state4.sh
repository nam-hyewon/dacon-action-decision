#!/bin/bash
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dacon

cd ~/projects/dacon-action-decision

SEED=42 \
STUDENT_MODEL=submit/model/deberta_base_eda_state4 \
SAVE_NAME=deberta_base_eda_v3_from_state4_lr5e6 \
EPOCHS=2 \
BATCH_SIZE=8 \
MAX_LEN=512 \
LR=5e-6 \
WEIGHT_DECAY=0.01 \
WARMUP_RATIO=0.03 \
USE_CLASS_WEIGHT=0 \
python src/train_deberta_base_eda_v3.py
