#!/bin/bash
# Submit all configurations


# ShareGPT
# --------

# Base
sbatch scripts/llama.slurm torchrun --nproc_per_node=4 train.py --epochs 1 --batch_size 32 --seq_len 256 --dataset sharegpt --model_type llama --model_head base --horizon 1 --horizon_eval 1 --freeze_base_model --init_method random
sbatch scripts/llama.slurm torchrun --nproc_per_node=4 train.py --epochs 1 --batch_size 32 --seq_len 256 --dataset sharegpt --model_type llama --model_head base --horizon 1 --horizon_eval 1 --freeze_base_model --init_method pretrained

# CP (hidden_dim=256-1024) Does increasing num_layers help?
sbatch scripts/llama.slurm torchrun --nproc_per_node=4 train.py --epochs 1 --batch_size 32 --seq_len 256 --dataset sharegpt --model_type llama --model_head cp --horizon 2 --rank 2 --horizon_eval 2 --freeze_base_model --use_memory_efficient_loss --hidden_dim 512 --num_layers 2 --use_layer_norm
sbatch scripts/llama.slurm torchrun --nproc_per_node=4 train.py --epochs 1 --batch_size 32 --seq_len 256 --dataset sharegpt --model_type llama --model_head cp --horizon 2 --rank 2 --horizon_eval 2 --freeze_base_model --use_memory_efficient_loss --hidden_dim 512 --num_layers 4 --use_layer_norm
sbatch scripts/llama.slurm torchrun --nproc_per_node=4 train.py --epochs 1 --batch_size 32 --seq_len 256 --dataset sharegpt --model_type llama --model_head cp --horizon 2 --rank 2 --horizon_eval 2 --freeze_base_model --use_memory_efficient_loss --hidden_dim 512 --num_layers 8 --use_layer_norm


sbatch scripts/llama.slurm torchrun --nproc_per_node=4 train.py --epochs 1 --batch_size 32 --seq_len 256 --dataset sharegpt --model_type llama --model_head cp --horizon 2 --rank 16 --horizon_eval 2 --freeze_base_model --use_memory_efficient_loss --hidden_dim 512 --num_layers 4 --use_layer_norm


# Can we get decent performance with CP horizon=1?
sbatch scripts/llama.slurm torchrun --nproc_per_node=4 train.py --epochs 1 --batch_size 32 --seq_len 256 --dataset sharegpt --model_type llama --model_head base --rank 1 --horizon 1 --horizon_eval 1 --freeze_base_model --init_method random
sbatch scripts/llama.slurm torchrun --nproc_per_node=4 train.py --epochs 1 --batch_size 32 --seq_len 256 --dataset sharegpt --model_type llama --model_head cp --rank 1 --horizon 1 --horizon_eval 1 --freeze_base_model --init_method random
