#!/bin/bash
#SBATCH --job-name=torch_train
#SBATCH --output=slurm/slurm-%j.out
#SBATCH --error=slurm/slurm-%j.err
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --nodes=1

module load python/3.9
source /home/mila/m/marawan.gamal/scratch/TJDNet/.venv/bin/activate

$@  # This will use all arguments passed to the script