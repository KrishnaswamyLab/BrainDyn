#!/bin/bash

#SBATCH --job-name=braindyn_nest
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --qos=qos_nmi --gres=gpu:h200:1
#SBATCH --mem=256G
#SBATCH --output=./logs/slurm/%x_%j.out
#SBATCH --error=./logs/slurm/%x_%j.err

set -euo pipefail

module load uv
source .venv/bin/activate

NPZ_PATH="${NPZ_PATH:-data/simulated_neuron_dataset/dataset.npz}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
X="${X:-30}"
Y="${Y:-10}"
EPOCHS="${EPOCHS:-60}"
CV_FOLDS="${CV_FOLDS:-5}"

uv run train_nest.py \
  --npz_path "$NPZ_PATH" \
  --x "$X" \
  --y "$Y" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --epochs "$EPOCHS" \
  --cv_folds "$CV_FOLDS" \
  --amp \
  --save_path checkpoints/braindyn_nest_unperturbed_best.pt \
  "$@"
