#!/bin/bash
#SBATCH --job-name=raw-file-pairs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=file_raw_pairs_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dacon

cd /home/skaodjs/projects/dacon-action-decision

export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12

python -u mine_file_strict_raw_state_pairs.py
