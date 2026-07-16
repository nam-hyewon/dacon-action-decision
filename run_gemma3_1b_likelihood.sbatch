#!/bin/bash
#SBATCH --job-name=gemma3-1b-probe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=gemma3_1b_likelihood_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dacon

cd /home/skaodjs/projects/dacon-action-decision

python -u probe_decoder_action_likelihood_fold0.py \
  --model-id google/gemma-3-1b-it \
  --subset-size 1000 \
  --batch 16 \
  --max-state-tokens 448 \
  --seed 43 \
  --output-dir decoder_probe_gemma3_1b
