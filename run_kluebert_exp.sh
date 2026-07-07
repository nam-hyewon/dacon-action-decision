#!/usr/bin/env bash
# KLUE-BERT focused experiment runner.
#
# Usage:
#   bash run_kluebert_exp.sh base
#   bash run_kluebert_exp.sh weighted
#   bash run_kluebert_exp.sh focal1
#   bash run_kluebert_exp.sh focal2w
#   bash run_kluebert_exp.sh state
#   bash run_kluebert_exp.sh lr2
#   bash run_kluebert_exp.sh seed7
#   bash run_kluebert_exp.sh seed123
#
# Each run saves validation logits and then tunes specialist boosters.

set -euo pipefail

EXP="${1:-base}"

export MODEL_NAME="klue/bert-base"
export TEXT_MODE="eda_state_mid"
export EPOCHS=5
export LR=1.5e-5
export MAX_LEN=512
export BATCH_SIZE="${BATCH_SIZE:-16}"
export SEED="${SEED:-42}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

unset USE_CLASS_WEIGHT
unset WEIGHT_MODE
unset LOSS_TYPE
unset FOCAL_GAMMA
unset SAVE_EPOCH_CHECKPOINTS

case "$EXP" in
  base)
    export SAVE_NAME="kluebert_mid5_lr15_rerun"
    ;;
  weighted)
    export SAVE_NAME="kluebert_mid5_lr15_weighted"
    export USE_CLASS_WEIGHT=1
    export WEIGHT_MODE=sqrt_inv
    ;;
  focal1)
    export SAVE_NAME="kluebert_mid5_lr15_focal1"
    export LOSS_TYPE=focal
    export FOCAL_GAMMA=1.0
    ;;
  focal2w)
    export SAVE_NAME="kluebert_mid5_lr15_focal2w"
    export LOSS_TYPE=focal
    export FOCAL_GAMMA=2.0
    export USE_CLASS_WEIGHT=1
    export WEIGHT_MODE=sqrt_inv
    ;;
  state)
    export SAVE_NAME="kluebert_state4_lr15"
    export TEXT_MODE=eda_state
    export EPOCHS=4
    ;;
  lr2)
    export SAVE_NAME="kluebert_mid5_lr2"
    export LR=2e-5
    ;;
  lr1)
    export SAVE_NAME="kluebert_mid5_lr1"
    export LR=1e-5
    ;;
  seed7)
    export SAVE_NAME="kluebert_mid5_lr15_seed7"
    export SEED=7
    ;;
  seed123)
    export SAVE_NAME="kluebert_mid5_lr15_seed123"
    export SEED=123
    ;;
  ckpt)
    export SAVE_NAME="kluebert_mid5_lr15_ckpt"
    export SAVE_EPOCH_CHECKPOINTS=3,4,5
    ;;
  *)
    echo "unknown exp: $EXP"
    echo "valid: base weighted focal1 focal2w state lr2 lr1 seed7 seed123 ckpt"
    exit 1
    ;;
esac

mkdir -p logs
LOG="logs/${SAVE_NAME}.log"

echo "===== KLUE-BERT EXP: ${EXP} ====="
echo "SAVE_NAME=${SAVE_NAME}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "TEXT_MODE=${TEXT_MODE} EPOCHS=${EPOCHS} LR=${LR} SEED=${SEED}"
echo "USE_CLASS_WEIGHT=${USE_CLASS_WEIGHT:-0} WEIGHT_MODE=${WEIGHT_MODE:-none}"
echo "LOSS_TYPE=${LOSS_TYPE:-ce} FOCAL_GAMMA=${FOCAL_GAMMA:-none}"
echo "LOG=${LOG}"

python src/train_roberta.py 2>&1 | tee "${LOG}"

python src/tune_specialist_boosters.py \
  --data_dir data \
  --val_csv "submit/model/${SAVE_NAME}/val_predictions.csv" \
  --val_logits "submit/model/${SAVE_NAME}/val_logits.npy" \
  --out_dir "submit/model/${SAVE_NAME}/specialist_boosters_fine" \
  --greedy 2>&1 | tee -a "${LOG}"

echo
echo "===== SUMMARY ${SAVE_NAME} ====="
grep -E "Best Validation Macro-F1|Base Macro-F1|\"macro_f1\"|\"delta_macro_f1\"|\"accuracy\"" "${LOG}" | tail -20
