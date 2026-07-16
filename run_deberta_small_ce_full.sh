#!/bin/bash
set -e

echo "===== DEBERTA-SMALL CE FULL START ====="
date
hostname
nvidia-smi || true

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dacon

cd ~/projects/dacon-action-decision

SEED=42 \
STUDENT_MODEL=microsoft/deberta-v3-small \
SAVE_NAME=deberta_small_eda_ce_full_lr2e5 \
EPOCHS=4 \
BATCH_SIZE=16 \
TEACHER_BATCH_SIZE=32 \
MAX_LEN=512 \
LR=2e-5 \
WEIGHT_DECAY=0.01 \
WARMUP_RATIO=0.06 \
KD_ALPHA=0.0 \
KD_T=2.0 \
TEACHER_CACHE=submit/cache/md_teacher_train_logits.npy \
python src/train_distill_deberta_small_eda.py

date
echo "===== DONE ====="
