#!/bin/bash

#SBATCH --job-name=braindyn_short_main
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus=1
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
  --x 30 \
  --y 10 \
  --forecast_mode short \
  --amp \
  --no_pin_memory \
  --lr 1e-3 \
  --lr_patience 3 \
  --lr_factor 0.5 \
  --lr_min 1e-6 \
  --save_path "checkpoints/braindyn_rbc_${COHORT_LOWER}_short_main_best.pt"
