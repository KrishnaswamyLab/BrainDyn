#!/bin/bash

#SBATCH --job-name=braindyn_tusz_binary_no_lstm
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --qos=qos_nmi --gres=gpu:h200:1
#SBATCH --mem=256G
#SBATCH --output=./logs/slurm/%x_%j.out
#SBATCH --error=./logs/slurm/%x_%j.err

set -euo pipefail

module load uv
source .venv/bin/activate

H5_PATH="${H5_PATH:-data/tusz_binary.h5}"
MANIFEST_CSV="${MANIFEST_CSV:-data/manifest_tusz_binary.csv}"

EPOCHS_FORECAST="${EPOCHS_FORECAST:-100}"
EPOCHS_CLS="${EPOCHS_CLS:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"
CV_FOLDS="${CV_FOLDS:-5}"
SUBSET_TRAIN_WINDOWS="${SUBSET_TRAIN_WINDOWS:-10000}"
SUBSET_EVAL_WINDOWS="${SUBSET_EVAL_WINDOWS:-2500}"

uv run train_tusz_binary.py \
  --h5_path "$H5_PATH" \
  --manifest_csv "$MANIFEST_CSV" \
  --x_len 30 \
  --batch_size "$BATCH_SIZE" \
  --num_workers 4 \
  --zscore \
  --eps 1e-2 \
  --ablation_gat \
  --epochs_forecast "$EPOCHS_FORECAST" \
  --epochs_cls "$EPOCHS_CLS" \
  --cv_folds "$CV_FOLDS" \
  --lr_forecast 3e-4 \
  --lr_cls_head 1e-3 \
  --lr_cls_backbone 1e-5 \
  --freeze_backbone \
  --use_pos_weight \
  --amp \
  --no_pin_memory \
  --subset_train_windows "$SUBSET_TRAIN_WINDOWS" \
  --subset_eval_windows "$SUBSET_EVAL_WINDOWS" \
  --save_forecast_path checkpoints/braindyn_tusz_binary_ablation_gat_forecast_best.pt \
  --save_classifier_path checkpoints/braindyn_tusz_binary_ablation_gat_classifier_best.pt
