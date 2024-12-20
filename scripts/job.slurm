#!/bin/bash
#SBATCH --job-name=torch_train
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --nodes=1

module load python/3.9

# Activate the virtual environment
source /home/mila/m/marawan.gamal/scratch/TJDNet/.venv/bin/activate

torchrun --nproc_per_node=4 train.py --model_type llama --model_head base --horizon 1 --horizon_eval 1 --dataset sharegpt --freeze_base_model --batch_size 64 --seq_len 256 --epochs 1 --init_method random