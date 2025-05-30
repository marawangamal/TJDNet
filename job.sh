#!/bin/bash
#SBATCH --output=slurm/slurm-%j.out
#SBATCH --error=slurm/slurm-%j.err   
#SBATCH --partition=short-unkillable
#SBATCH --gres=gpu:a100l:4
#SBATCH --mem=128G
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=12
#SBATCH --nodes=1
source /home/mila/m/marawan.gamal/scratch/tjdnet/.venv/bin/activate

# test hello world
python -c "print('Hello, world!')"
