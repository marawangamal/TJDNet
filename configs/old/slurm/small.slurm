#!/bin/bash
#SBATCH --output=slurm/slurm-%j.out
#SBATCH --error=slurm/slurm-%j.err
#SBATCH --gres=gpu:2      
#SBATCH --cpus-per-task=4    
#SBATCH --mem=32G           
#SBATCH --nodes=1

module load python/3.9
source /home/mila/m/marawan.gamal/scratch/TJDNet/.venv/bin/activate

$@