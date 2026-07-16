#!/bin/bash
#SBATCH --job-name=deb-f4-allval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-deb-f4-allval-%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dacon

cd /home/skaodjs/projects/dacon-action-decision

BATCH_SIZE=16 \
python src/infer_deberta_file4_all_fold0.py
