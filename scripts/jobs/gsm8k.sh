#!/bin/bash

# Baseline (random & pretrained)
sbatch scripts/slurm/large-unkillable.slurm accelerate launch --use_fsdp --config_file configs/fsdp/fsdp_4gpus.yaml train.py --dataset gsm8k --model_type llama7b --epochs 50 --batch_size 8 --seq_len 128 --lr 1e-5 --model_head base

# CP (rank=2,4,8 & horizon=2,4)
sbatch scripts/slurm/large-unkillable.slurm accelerate launch --use_fsdp --config_file configs/fsdp/fsdp_4gpus.yaml train.py --dataset gsm8k --model_type llama7b --epochs 50 --batch_size 8 --seq_len 128 --lr 1e-5 --model_head cp --num_layers 3 --hidden_dim 768  --horizon 2 --horizon_eval 2 --rank 2
sbatch scripts/slurm/large-unkillable.slurm accelerate launch --use_fsdp --config_file configs/fsdp/fsdp_4gpus.yaml train.py --dataset gsm8k --model_type llama7b --epochs 50 --batch_size 8 --seq_len 128 --lr 1e-5 --model_head cp --num_layers 3 --hidden_dim 768  --horizon 2 --horizon_eval 2 --rank 4
sbatch scripts/slurm/large-unkillable.slurm accelerate launch --use_fsdp --config_file configs/fsdp/fsdp_4gpus.yaml train.py --dataset gsm8k --model_type llama7b --epochs 50 --batch_size 8 --seq_len 128 --lr 1e-5 --model_head cp --num_layers 3 --hidden_dim 768  --horizon 2 --horizon_eval 2 --rank 8
sbatch scripts/slurm/large-unkillable.slurm accelerate launch --use_fsdp --config_file configs/fsdp/fsdp_4gpus.yaml train.py --dataset gsm8k --model_type llama7b --epochs 50 --batch_size 8 --seq_len 128 --lr 1e-5 --model_head cp --num_layers 3 --hidden_dim 768  --horizon 2 --horizon_eval 2 --rank 16
sbatch scripts/slurm/large-unkillable.slurm accelerate launch --use_fsdp --config_file configs/fsdp/fsdp_4gpus.yaml train.py --dataset gsm8k --model_type llama7b --epochs 50 --batch_size 8 --seq_len 128 --lr 1e-5 --model_head cp --num_layers 3 --hidden_dim 768  --horizon 2 --horizon_eval 2 --rank 32

# horizon=2, num_layers=3, rank=(8,16)
sbatch scripts/slurm/large-unkillable.slurm accelerate launch --use_fsdp --config_file configs/fsdp/fsdp_4gpus.yaml train.py --dataset gsm8k --model_type llama7b --epochs 50 --batch_size 8 --seq_len 128 --lr 1e-5 --model_head cp --num_layers 3 --hidden_dim 768  --horizon 2 --horizon_eval 2 --rank 8 
sbatch scripts/slurm/large-unkillable.slurm accelerate launch --use_fsdp --config_file configs/fsdp/fsdp_4gpus.yaml train.py --dataset gsm8k --model_type llama7b --epochs 50 --batch_size 8 --seq_len 128 --lr 1e-5 --model_head cp --num_layers 3 --hidden_dim 768  --horizon 2 --horizon_eval 2 --rank 16 

