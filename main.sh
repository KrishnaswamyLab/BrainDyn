#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=1
#SBATCH --time 6:00:00
#SBATCH --job-name braindyn
#SBATCH --output logs/braindyn-%J.log

cd /gpfs/gibbs/pi/krishnaswamy_smita/sv496/BrainDyn

# Load uv (adjust based on your cluster setup)
module load uv  # or use: export PATH="$HOME/.cargo/bin:$PATH"

source .venv/bin/activate

python main.py