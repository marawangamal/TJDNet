#!/bin/bash
# Submit all configurations


# ShareGPT
# --------

# Base
sbatch scripts/llama.slurm torchrun --nproc_per_node=4 train.py --epochs 1 --batch_size 32 --seq_len 256 --dataset sharegpt --model_type llama --model_head base --horizon 1 --horizon_eval 1 --freeze_base_model --init_method random
sbatch scripts/llama.slurm torchrun --nproc_per_node=4 train.py --epochs 1 --batch_size 32 --seq_len 256 --dataset sharegpt --model_type llama --model_head base --horizon 1 --horizon_eval 1 --freeze_base_model --init_method pretrained

# CP (R=2-8)  Does increasing rank help? Yes.
sbatch scripts/llama.slurm torchrun --nproc_per_node=4 train.py --epochs 1 --batch_size 32 --seq_len 256 --dataset sharegpt --model_type llama --model_head cp --horizon 2 --rank 2 --horizon_eval 2 --freeze_base_model --use_memory_efficient_loss
sbatch scripts/llama.slurm torchrun --nproc_per_node=4 train.py --epochs 1 --batch_size 32 --seq_len 256 --dataset sharegpt --model_type llama --model_head cp --horizon 2 --rank 4 --horizon_eval 2 --freeze_base_model --use_memory_efficient_loss
sbatch scripts/llama.slurm torchrun --nproc_per_node=4 train.py --epochs 1 --batch_size 32 --seq_len 256 --dataset sharegpt --model_type llama --model_head cp --horizon 2 --rank 8 --horizon_eval 2 --freeze_base_model --use_memory_efficient_loss

# CP (hidden_dim=256-1024) Does increasing hidden_dim help?
# sbatch scripts/llama.slurm torchrun --nproc_per_node=4 train.py --epochs 1 --batch_size 32 --seq_len 256 --dataset sharegpt --model_type llama --model_head cp --horizon 2 --rank 2 --horizon_eval 2 --freeze_base_model --use_memory_efficient_loss --hidden_dim 512
sbatch scripts/llama.slurm torchrun --nproc_per_node=4 train.py --epochs 1 --batch_size 32 --seq_len 256 --dataset sharegpt --model_type llama --model_head cp --horizon 2 --rank 2 --horizon_eval 2 --freeze_base_model --use_memory_efficient_loss --hidden_dim 512 
sbatch scripts/llama.slurm torchrun --nproc_per_node=4 train.py --epochs 1 --batch_size 32 --seq_len 256 --dataset sharegpt --model_type llama --model_head cp --horizon 2 --rank 2 --horizon_eval 2 --freeze_base_model --use_memory_efficient_loss --hidden_dim 1024


# CP (use_nonlinearity) Does nonlinearity help?
# sbatch scripts/llama.slurm torchrun --nproc_per_node=4 train.py --epochs 1 --batch_size 32 --seq_len 256 --dataset sharegpt --model_type llama --model_head cp --horizon 2 --rank 2 --horizon_eval 2 --freeze_base_model --use_memory_efficient_loss
sbatch scripts/llama.slurm torchrun --nproc_per_node=4 train.py --epochs 1 --batch_size 32 --seq_len 256 --dataset sharegpt --model_type llama --model_head cp --horizon 2 --rank 2 --horizon_eval 2 --freeze_base_model --use_memory_efficient_loss --use_nonlinearity


# MPS
sbatch scripts/llama.slurm torchrun --nproc_per_node=4 train.py --epochs 1 --batch_size 32 --seq_len 256 --dataset sharegpt --model_type llama --model_head mps --horizon 2 --rank 2 --horizon_eval 2 --freeze_base_model --use_memory_efficient_loss


# uMPS
sbatch scripts/llama.slurm torchrun --nproc_per_node=4 train.py --epochs 1 --batch_size 32 --seq_len 256 --dataset sharegpt --model_type llama --model_head umps --horizon 2 --rank 2 --horizon_eval 2 --freeze_base_model --use_memory_efficient_loss
sbatch scripts/llama.slurm torchrun --nproc_per_node=4 train.py --epochs 1 --batch_size 32 --seq_len 256 --dataset sharegpt --model_type llama --model_head umps --horizon 2 --rank 4 --horizon_eval 2 --freeze_base_model --use_memory_efficient_loss


# Shakespeare
# -----------

# Base
sbatch scripts/gpt.slurm torchrun --nproc_per_node=2 train.py --eval_strategy epoch --eval_steps 1 --epochs 20 --batch_size 32 --seq_len 256 --dataset shakespeare --model_type gpt2 --model_head base --horizon 1 --horizon_eval 1 --freeze_base_model --use_memory_efficient_loss --init_method random
sbatch scripts/gpt.slurm torchrun --nproc_per_node=2 train.py --eval_strategy epoch --eval_steps 1 --epochs 20 --batch_size 32 --seq_len 256 --dataset shakespeare --model_type gpt2 --model_head base --horizon 1 --horizon_eval 1 --freeze_base_model --use_memory_efficient_loss --init_method pretrained

# CP
sbatch scripts/gpt.slurm torchrun --nproc_per_node=2 train.py --eval_strategy epoch --eval_steps 1 --epochs 20 --batch_size 32 --seq_len 256 --dataset shakespeare --model_type gpt2 --model_head cp --horizon 2 --rank 2 --horizon_eval 2 --freeze_base_model --use_memory_efficient_loss
sbatch scripts/gpt.slurm torchrun --nproc_per_node=2 train.py --eval_strategy epoch --eval_steps 1 --epochs 20 --batch_size 32 --seq_len 256 --dataset shakespeare --model_type gpt2 --model_head cp --horizon 2 --rank 4 --horizon_eval 2 --freeze_base_model --use_memory_efficient_loss
sbatch scripts/llama.slurm torchrun --nproc_per_node=4 train.py --eval_strategy epoch --eval_steps 1 --epochs 20 --batch_size 32 --seq_len 256 --dataset shakespeare --model_type gpt2 --model_head cp --horizon 2 --rank 8 --horizon_eval 2 --freeze_base_model --use_memory_efficient_loss
sbatch scripts/llama.slurm torchrun --nproc_per_node=4 train.py --eval_strategy epoch --eval_steps 1 --epochs 20 --batch_size 32 --seq_len 256 --dataset shakespeare --model_type gpt2 --model_head cp --horizon 2 --rank 16 --horizon_eval 2 --freeze_base_model --use_memory_efficient_loss
sbatch scripts/llama.slurm torchrun --nproc_per_node=4 train.py --eval_strategy epoch --eval_steps 1 --epochs 20 --batch_size 32 --seq_len 256 --dataset shakespeare --model_type gpt2 --model_head cp --horizon 2 --rank 32 --horizon_eval 2 --freeze_base_model --use_memory_efficient_loss