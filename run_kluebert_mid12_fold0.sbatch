#!/bin/bash
#SBATCH --job-name=klue_mid12_f0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm-klue-mid12-f0-%j.out
#SBATCH --error=slurm-klue-mid12-f0-%j.err

set -e

echo "===== KLUEBERT MID12 FOLD0 START ====="
date
hostname
pwd
nvidia-smi || true

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dacon

cd ~/projects/dacon-action-decision

MODEL_NAME=klue/bert-base \
TEXT_MODE=eda_state_mid \
SAVE_NAME=kluebert_mid12_lr15_fold0 \
EPOCHS=12 \
BATCH_SIZE=16 \
MAX_LEN=512 \
LR=1.5e-5 \
WEIGHT_DECAY=0.01 \
WARMUP_RATIO=0.06 \
REF_VAL_PATH=team_4model_ensemble_07626/ensemble_07626/fold0_team/tabular_folds/tabular_histgb_fold0/val_predictions.csv \
python -u src/train_roberta_fold_ref.py \
2>&1 | tee kluebert_mid12_lr15_fold0.log

echo "===== KLUEBERT MID12 FOLD0 END ====="
date

echo "===== SAVED MODEL ====="
ls -lh submit/model/kluebert_mid12_lr15_fold0
