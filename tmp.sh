#!/bin/bash
#SBATCH --output=slurm/slurm-%j.out
#SBATCH --error=slurm/slurm-error-%j.out
#SBATCH --partition=short-unkillable
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --time=3:00:00
module load python/3.9
source /home/mila/m/marawan.gamal/scratch/TJDNet/.venv/bin/activate
accelerate launch --use_fsdp --config_file configs/fsdp/fsdp_4gpus.yaml train.py --dataset gsm8k --model_type llama7b --epochs 50 --batch_size 8 --seq_len 128 --lr 1e-5 --model_head cp --num_layers 2 --hidden_dim 768  --horizon 2 --horizon_eval 2 --rank 32
