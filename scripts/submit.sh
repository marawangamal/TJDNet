#!/bin/bash

# Submit all configurations

# Base
sbatch job.sh torchrun --nproc_per_node=4 train.py --model_type llama --model_head base --horizon 1 --horizon_eval 1 --dataset sharegpt --freeze_base_model --batch_size 64 --seq_len 256 --epochs 1 --init_method random
sbatch job.sh torchrun --nproc_per_node=4 train.py --model_type llama --model_head base --horizon 1 --horizon_eval 1 --dataset sharegpt --freeze_base_model --batch_size 64 --seq_len 256 --epochs 1 --init_method pretrained

# MPS
sbatch job.sh torchrun --nproc_per_node=4 train.py --model_type llama --model_head mps --horizon 2 --rank 2 --horizon_eval 2 --dataset sharegpt --freeze_base_model --batch_size 32 --seq_len 256 --epochs 1
sbatch job.sh torchrun --nproc_per_node=4 train.py --model_type llama --model_head mps --horizon 2 --rank 4 --horizon_eval 2 --dataset sharegpt --freeze_base_model --batch_size 4 --seq_len 128 --epochs 1 --use_memory_efficient_loss

# CP

# uMPS