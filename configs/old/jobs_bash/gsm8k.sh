#!/bin/bash

# Baseline (random & pretrained)
sbatch scripts/slurm/large-unkillable.slurm accelerate launch --use_fsdp --config_file configs/fsdp/fsdp_4gpus.yaml train.py --dataset gsm8k --model llama7b --epochs 50 --batch_size 4 --seq_len 128 --lr 1e-5 --model_head base --compute_acc

# CP (h=2, nl=2, rank=2:32)
sbatch scripts/slurm/large-unkillable.slurm accelerate launch --use_fsdp --config_file configs/fsdp/fsdp_4gpus.yaml train.py --dataset gsm8k --model llama7b --epochs 50 --batch_size 8 --seq_len 128 --lr 1e-5 --model_head cp --num_layers 2 --hidden_dim 768  --horizon 2 --horizon_eval 2 --rank 2
sbatch scripts/slurm/large-unkillable.slurm accelerate launch --use_fsdp --config_file configs/fsdp/fsdp_4gpus.yaml train.py --dataset gsm8k --model llama7b --epochs 50 --batch_size 8 --seq_len 128 --lr 1e-5 --model_head cp --num_layers 2 --hidden_dim 768  --horizon 2 --horizon_eval 2 --rank 4
sbatch scripts/slurm/large-unkillable.slurm accelerate launch --use_fsdp --config_file configs/fsdp/fsdp_4gpus.yaml train.py --dataset gsm8k --model llama7b --epochs 50 --batch_size 8 --seq_len 128 --lr 1e-5 --model_head cp --num_layers 2 --hidden_dim 768  --horizon 2 --horizon_eval 2 --rank 8
sbatch scripts/slurm/large-unkillable.slurm accelerate launch --use_fsdp --config_file configs/fsdp/fsdp_4gpus.yaml train.py --dataset gsm8k --model llama7b --epochs 50 --batch_size 8 --seq_len 128 --lr 1e-5 --model_head cp --num_layers 2 --hidden_dim 768  --horizon 2 --horizon_eval 2 --rank 16
sbatch scripts/slurm/large-unkillable.slurm accelerate launch --use_fsdp --config_file configs/fsdp/fsdp_4gpus.yaml train.py --dataset gsm8k --model llama7b --epochs 50 --batch_size 8 --seq_len 128 --lr 1e-5 --model_head cp --num_layers 2 --hidden_dim 768  --horizon 2 --horizon_eval 2 --rank 32

# CP (h=4, nl=2, rank=2:32)
sbatch scripts/slurm/large-unkillable.slurm accelerate launch --use_fsdp --config_file configs/fsdp/fsdp_4gpus.yaml train.py --dataset gsm8k --model llama7b --epochs 50 --batch_size 8 --seq_len 128 --lr 1e-5 --model_head cp --num_layers 2 --hidden_dim 768  --horizon 2 --horizon_eval 2 --rank 2
sbatch scripts/slurm/large-unkillable.slurm accelerate launch --use_fsdp --config_file configs/fsdp/fsdp_4gpus.yaml train.py --dataset gsm8k --model llama7b --epochs 50 --batch_size 8 --seq_len 128 --lr 1e-5 --model_head cp --num_layers 2 --hidden_dim 768  --horizon 2 --horizon_eval 2 --rank 4
sbatch scripts/slurm/large-unkillable.slurm accelerate launch --use_fsdp --config_file configs/fsdp/fsdp_4gpus.yaml train.py --dataset gsm8k --model llama7b --epochs 50 --batch_size 8 --seq_len 128 --lr 1e-5 --model_head cp --num_layers 2 --hidden_dim 768  --horizon 2 --horizon_eval 2 --rank 8
sbatch scripts/slurm/large-unkillable.slurm accelerate launch --use_fsdp --config_file configs/fsdp/fsdp_4gpus.yaml train.py --dataset gsm8k --model llama7b --epochs 50 --batch_size 8 --seq_len 128 --lr 1e-5 --model_head cp --num_layers 2 --hidden_dim 768  --horizon 2 --horizon_eval 2 --rank 16
sbatch scripts/slurm/large-unkillable.slurm accelerate launch --use_fsdp --config_file configs/fsdp/fsdp_4gpus.yaml train.py --dataset gsm8k --model llama7b --epochs 50 --batch_size 8 --seq_len 128 --lr 1e-5 --model_head cp --num_layers 2 --hidden_dim 768  --horizon 2 --horizon_eval 2 --rank 32

# Acc calculations
# sbatch scripts/slurm/large-unkillable.slurm python scripts/eval_acc.py -b 32 -c checkpoints/<ckpt_dir>