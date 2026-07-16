#!/bin/bash
set -e

echo "===== DEBERTA-SMALL KD FROM CE A01 START ====="
date
hostname
nvidia-smi || true

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dacon

cd ~/projects/dacon-action-decision

SEED=42 \
STUDENT_MODEL=submit/model/deberta_small_eda_ce_full_lr2e5 \
SAVE_NAME=deberta_small_eda_kd_from_ce_a01_t2_lr5e6 \
EPOCHS=2 \
BATCH_SIZE=16 \
TEACHER_BATCH_SIZE=32 \
MAX_LEN=512 \
LR=5e-6 \
WEIGHT_DECAY=0.01 \
WARMUP_RATIO=0.03 \
KD_ALPHA=0.1 \
KD_T=2.0 \
TEACHER_CACHE=submit/cache/md_teacher_train_logits.npy \
python src/train_distill_deberta_small_eda.py

date
echo "===== DONE ====="
