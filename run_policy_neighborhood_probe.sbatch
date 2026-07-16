#!/bin/bash
#SBATCH --job-name=policy-knn
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=policy_neighborhood_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dacon

cd /home/skaodjs/projects/dacon-action-decision

python -u probe_policy_neighborhood_purity_fold0.py \
  --batch 64 \
  --max-len 512 \
  --query-chunk 256 \
  --train-chunk 8192 \
  --max-k 100 \
  --output-dir policy_neighborhood_probe_fold0
