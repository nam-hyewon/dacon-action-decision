#!/bin/bash
#SBATCH --job-name=deb-f4-hard3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-deb-file4-hard3-f0-%j.out
#SBATCH --error=slurm-deb-file4-hard3-f0-%j.err

set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dacon

cd /home/skaodjs/projects/dacon-action-decision

REF_VAL_PATH=team_4model_ensemble_07626/ensemble_07626/fold0_team/tabular_folds/tabular_histgb_fold0/val_predictions.csv \
MODEL_NAME=microsoft/deberta-v3-base \
SAVE_NAME=deberta_file4_hard3_eda_state_fold0_seed43 \
TEXT_MODE=eda_state \
MAX_LEN=512 \
BATCH_SIZE=4 \
EPOCHS=4 \
LR=1e-5 \
WEIGHT_DECAY=0.01 \
WARMUP_RATIO=0.06 \
SEED=43 \
HARD_REPEAT=3 \
python src/train_deberta_file4_hardneg_fold_ref.py

echo "===== DEBERTA FILE4 HARD3 FOLD0 END ====="
