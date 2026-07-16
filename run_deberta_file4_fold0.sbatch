#!/bin/bash
#SBATCH --job-name=deb_file4_f0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm-deb-file4-f0-%j.out
#SBATCH --error=slurm-deb-file4-f0-%j.err

set -e

echo "===== DEBERTA FILE4 FOLD0 START ====="
date
hostname
nvidia-smi || true

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dacon

cd ~/projects/dacon-action-decision

FOLD_ID=0 \
SEED=43 \
REF_VAL_PATH=team_4model_ensemble_07626/ensemble_07626/fold0_team/tabular_folds/tabular_histgb_fold0/val_predictions.csv \
./run_deberta_file4_fold_ref.sh \
2>&1 | tee deberta_file4_fold0.log

echo "===== DEBERTA FILE4 FOLD0 END ====="
date
