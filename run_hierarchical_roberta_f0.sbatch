#!/bin/bash
#SBATCH --job-name=hier-roberta
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=hier_roberta_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dacon

cd /home/skaodjs/projects/dacon-action-decision

python -u train_hierarchical_roberta_fold0.py
