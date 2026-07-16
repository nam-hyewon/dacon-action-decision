#!/bin/bash
#SBATCH --job-name=file-pooling
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=file_pooling_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dacon

cd /home/skaodjs/projects/dacon-action-decision

python -u \
  probe_file_layer_pooling_geometry.py
