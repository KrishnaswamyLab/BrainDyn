#!/bin/bash

#SBATCH --job-name=braindyn_pnc
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=256G
#SBATCH --output=./logs/slurm/%x_%j.out
#SBATCH --error=./logs/slurm/%x_%j.err

module load uv
source .venv/bin/activate

uv run main.py --cohort PNC --amp --no_pin_memory --lr 1e-3 --lr_patience 3 --lr_factor 0.5 --lr_min 1e-6