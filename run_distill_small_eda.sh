#!/bin/bash
set -e

echo "===== DISTILL DEBERTA-SMALL EDA FROM MD TEACHER START ====="
date
hostname
pwd
nvidia-smi || true

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dacon

cd ~/projects/dacon-action-decision

SEED=42 \
STUDENT_MODEL=microsoft/deberta-v3-small \
SAVE_NAME=deberta_small_eda_distill_md_teacher_a07_t2 \
EPOCHS=4 \
BATCH_SIZE=8 \
TEACHER_BATCH_SIZE=32 \
MAX_LEN=512 \
LR=1e-5 \
WEIGHT_DECAY=0.01 \
WARMUP_RATIO=0.06 \
KD_ALPHA=0.7 \
KD_T=2.0 \
python src/train_distill_deberta_small_eda.py

date
echo "===== SAVED MODEL ====="
ls -lh submit/model/deberta_small_eda_distill_md_teacher_a07_t2
