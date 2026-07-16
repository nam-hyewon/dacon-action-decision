#!/bin/bash
#SBATCH --job-name=deb_seed
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=deberta_seed%j.out
#SBATCH --error=deberta_seed%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dacon

cd ~/projects/dacon-action-decision/team_4model_ensemble_07626/ensemble_07626

echo "========================================"
echo "Running DeBERTa seed: ${SEED}"
echo "Output dir: deberta_base_eda_state4_fold0_seed${SEED}"
echo "Start time: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "========================================"

PYTHONUNBUFFERED=1 \
MODEL_NAME=microsoft/deberta-v3-base \
OUT_DIR=deberta_base_eda_state4_fold0_seed${SEED} \
FOLD_FILE=/home/skaodjs/projects/dacon-action-decision/data/folds_seed42.csv \
FOLD_ID=0 \
REF_VAL_PRED=fold0_team/deberta_fold0/val_predictions.csv \
SEED=${SEED} \
MAX_LEN=512 \
LR=3e-6 \
EPOCHS=4 \
BATCH_SIZE=2 \
GRAD_ACCUM=4 \
WEIGHT_DECAY=0.01 \
WARMUP_RATIO=0.10 \
FOCAL_GAMMA=0.0 \
python -u train_deberta_fold0_seed.py

echo "End time: $(date)"
