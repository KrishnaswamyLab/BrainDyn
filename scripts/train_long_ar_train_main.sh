#!/bin/bash

#SBATCH --job-name=braindyn_long_ar_train_main
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --qos=qos_nmi --gres=gpu:h200:1
#SBATCH --mem=256G
#SBATCH --output=./logs/slurm/%x_%j.out
#SBATCH --error=./logs/slurm/%x_%j.err

set -euo pipefail

module load uv
source .venv/bin/activate

COHORT="${COHORT:-PNC}"
COHORT_LOWER="${COHORT,,}"

uv run main.py \
  --cohort "$COHORT" \
  --cache \
  --x 30 \
  --y 10 \
  --forecast_mode long_ar_train \
  --ar_chunk_size 10 \
  --tbptt_chunks 3 \
  --amp \
  --lr 1e-3 \
  --lr_patience 3 \
  --lr_factor 0.5 \
  --lr_min 1e-6 \
  --precompute_lap_h \
  --run_batch_size 8 \
  --val_every 1 \
  --save_path "checkpoints/braindyn_rbc_${COHORT_LOWER}_long_ar_train_main_best.pt"
