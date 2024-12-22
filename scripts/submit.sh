#!/bin/bash

# Submit all configurations


# ShareGPT
# --------

Base
sbatch scripts/llama.slurm torchrun --nproc_per_node=4 train.py --epochs 1 --batch_size 32 --seq_len 256 --dataset sharegpt --model_type llama --model_head base --horizon 1 --freeze_base_model --use_memory_efficient_loss --init_method random
sbatch scripts/llama.slurm torchrun --nproc_per_node=4 train.py --epochs 1 --batch_size 32 --seq_len 256 --dataset sharegpt --model_type llama --model_head base --horizon 1 --freeze_base_model --use_memory_efficient_loss --init_method pretrained


# MPS
sbatch scripts/llama.slurm torchrun --nproc_per_node=4 train.py --epochs 1 --batch_size 32 --seq_len 256 --dataset sharegpt --model_type llama --model_head mps --horizon 2 --rank 2 --horizon_eval 2 --freeze_base_model --use_memory_efficient_loss


# CP
sbatch scripts/llama.slurm torchrun --nproc_per_node=4 train.py --epochs 1 --batch_size 32 --seq_len 256 --dataset sharegpt --model_type llama --model_head cp --horizon 2 --rank 2 --horizon_eval 2 --freeze_base_model --use_memory_efficient_loss
sbatch scripts/llama.slurm torchrun --nproc_per_node=4 train.py --epochs 1 --batch_size 32 --seq_len 256 --dataset sharegpt --model_type llama --model_head cp --horizon 2 --rank 4 --horizon_eval 2 --freeze_base_model --use_memory_efficient_loss
sbatch scripts/llama.slurm torchrun --nproc_per_node=4 train.py --epochs 1 --batch_size 32 --seq_len 256 --dataset sharegpt --model_type llama --model_head cp --horizon 2 --rank 8 --horizon_eval 2 --freeze_base_model --use_memory_efficient_loss

# uMPS
sbatch scripts/llama.slurm torchrun --nproc_per_node=4 train.py --epochs 1 --batch_size 32 --seq_len 256 --dataset sharegpt --model_type llama --model_head umps --horizon 2 --rank 2 --horizon_eval 2 --freeze_base_model --use_memory_efficient_loss
sbatch scripts/llama.slurm torchrun --nproc_per_node=4 train.py --epochs 1 --batch_size 32 --seq_len 256 --dataset sharegpt --model_type llama --model_head umps --horizon 2 --rank 4 --horizon_eval 2 --freeze_base_model --use_memory_efficient_loss


# Shakespeare
# -----------

# Base
sbatch scripts/gpt.slurm torchrun --nproc_per_node=4 train.py --epochs 1 --batch_size 32 --seq_len 256 --dataset shakespeare --model_type gpt2 --model_head base --horizon 1 --freeze_base_model --use_memory_efficient_loss --init_method random
sbatch scripts/gpt.slurm torchrun --nproc_per_node=4 train.py --epochs 1 --batch_size 32 --seq_len 256 --dataset shakespeare --model_type gpt2 --model_head base --horizon 1 --freeze_base_model --use_memory_efficient_loss --init_method pretrained

# CP
sbatch scripts/gpt.slurm torchrun --nproc_per_node=4 train.py --epochs 1 --batch_size 32 --seq_len 256 --dataset shakespeare --model_type gpt2 --model_head cp --horizon 2 --rank 2 --horizon_eval 2 --freeze_base_model --use_memory_efficient_loss
sbatch scripts/gpt.slurm torchrun --nproc_per_node=4 train.py --epochs 1 --batch_size 32 --seq_len 256 --dataset shakespeare --model_type gpt2 --model_head cp --horizon 2 --rank 4 --horizon_eval 2 --freeze_base_model --use_memory_efficient_loss
sbatch scripts/gpt.slurm torchrun --nproc_per_node=4 train.py --epochs 1 --batch_size 32 --seq_len 256 --dataset shakespeare --model_type gpt2 --model_head cp --horizon 2 --rank 8 --horizon_eval 2 --freeze_base_model --use_memory_efficient_loss
sbatch scripts/gpt.slurm torchrun --nproc_per_node=4 train.py --epochs 1 --batch_size 32 --seq_len 256 --dataset shakespeare --model_type gpt2 --model_head cp --horizon 2 --rank 16 --horizon_eval 2 --freeze_base_model --use_memory_efficient_loss
sbatch scripts/gpt.slurm torchrun --nproc_per_node=4 train.py --epochs 1 --batch_size 32 --seq_len 256 --dataset shakespeare --model_type gpt2 --model_head cp --horizon 2 --rank 32 --horizon_eval 2 --freeze_base_model --use_memory_efficient_loss