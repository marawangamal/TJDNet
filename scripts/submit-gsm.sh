#!/bin/bash
# Submit all configurations


# -------------------
# Dataset: GSM
# Model: GPT2 (char)
# -------------------
# Base
sbatch scripts/slurm/large.slurm torchrun --nproc_per_node=4 train.py --epochs 50 --logging_strategy epoch --logging_steps 1 --eval_strategy epoch --eval_steps 1 --generate_strategy epoch --tokenizer_type char --num_layers 1 --dropout 0 --top_k 200 --num_beams 1 --batch_size 32 --seq_len 512 --dataset gsm8k --model_type gpt2 --model_head base --horizon 1 --horizon_eval 1 --rank 1 --init_method random --lr 1e-3

# CP (H=2)
sbatch scripts/slurm/large.slurm torchrun --nproc_per_node=4 train.py --epochs 50 --logging_strategy epoch --logging_steps 1 --eval_strategy epoch --eval_steps 1 --generate_strategy epoch --tokenizer_type char --num_layers 1 --dropout 0 --top_k 200 --num_beams 1 --batch_size 32 --seq_len 512 --dataset gsm8k --model_type gpt2 --model_head cp --horizon 2 --horizon_eval 2 --rank 2 --init_method random --lr 1e-3
sbatch scripts/slurm/large.slurm torchrun --nproc_per_node=4 train.py --epochs 50 --logging_strategy epoch --logging_steps 1 --eval_strategy epoch --eval_steps 1 --generate_strategy epoch --tokenizer_type char --num_layers 1 --dropout 0 --top_k 200 --num_beams 1 --batch_size 32 --seq_len 512 --dataset gsm8k --model_type gpt2 --model_head cp --horizon 2 --horizon_eval 2 --rank 4 --init_method random --lr 1e-3

# CP (H=4)
sbatch scripts/slurm/large.slurm torchrun --nproc_per_node=4 train.py --epochs 50 --logging_strategy epoch --logging_steps 1 --eval_strategy epoch --eval_steps 1 --generate_strategy epoch --tokenizer_type char --num_layers 1 --dropout 0 --top_k 200 --num_beams 1 --batch_size 32 --seq_len 512 --dataset gsm8k --model_type gpt2 --model_head cp --horizon 4 --horizon_eval 4 --rank 4 --init_method random --lr 1e-3
sbatch scripts/slurm/large.slurm torchrun --nproc_per_node=4 train.py --epochs 50 --logging_strategy epoch --logging_steps 1 --eval_strategy epoch --eval_steps 1 --generate_strategy epoch --tokenizer_type char --num_layers 1 --dropout 0 --top_k 200 --num_beams 1 --batch_size 32 --seq_len 512 --dataset gsm8k --model_type gpt2 --model_head cp --horizon 4 --horizon_eval 4 --rank 4 --init_method random --lr 1e-3


# -------------------
# Dataset: GSM
# Model: GPT2 (word)
# -------------------

# Base
# sbatch scripts/slurm/large.slurm torchrun --nproc_per_node=4 train.py --epochs 50 --logging_strategy epoch --logging_steps 1 --eval_strategy epoch --eval_steps 1 --generate_strategy epoch --tokenizer_type word --num_layers 1 --dropout 0 --top_k 200 --num_beams 1 --batch_size 8 --seq_len 32 --dataset gsm8k --model_type gpt2 --model_head base --horizon 1 --horizon_eval 1 --rank 1 --init_method random --lr 1e-4

# CP (H=2)
# sbatch scripts/slurm/large.slurm torchrun --nproc_per_node=4 train.py --epochs 50 --logging_strategy epoch --logging_steps 1 --eval_strategy epoch --eval_steps 1 --generate_strategy epoch --tokenizer_type word --num_layers 1 --dropout 0 --top_k 200 --num_beams 1 --batch_size 32 --seq_len 512 --dataset gsm8k --model_type gpt2 --model_head cp --horizon 2 --horizon_eval 2 --rank 2 --init_method random --lr 1e-3
sbatch scripts/slurm/large.slurm torchrun --nproc_per_node=4 train.py --epochs 50 --logging_strategy epoch --logging_steps 1 --eval_strategy epoch --eval_steps 1 --generate_strategy epoch --tokenizer_type word --num_layers 1 --dropout 0 --top_k 200 --num_beams 1 --batch_size 32 --seq_len 512 --dataset gsm8k --model_type gpt2 --model_head cp --horizon 2 --horizon_eval 2 --rank 4 --init_method random --lr 1e-3

# CP (H=4)
sbatch scripts/slurm/large.slurm torchrun --nproc_per_node=4 train.py --epochs 50 --logging_strategy epoch --logging_steps 1 --eval_strategy epoch --eval_steps 1 --generate_strategy epoch --tokenizer_type word --num_layers 1 --dropout 0 --top_k 200 --num_beams 1 --batch_size 32 --seq_len 512 --dataset gsm8k --model_type gpt2 --model_head base --horizon 2 --horizon_eval 4 --rank 4 --init_method random --lr 1e-3
sbatch scripts/slurm/large.slurm torchrun --nproc_per_node=4 train.py --epochs 50 --logging_strategy epoch --logging_steps 1 --eval_strategy epoch --eval_steps 1 --generate_strategy epoch --tokenizer_type word --num_layers 1 --dropout 0 --top_k 200 --num_beams 1 --batch_size 32 --seq_len 512 --dataset gsm8k --model_type gpt2 --model_head base --horizon 2 --horizon_eval 4 --rank 4 --init_method random --lr 1e-3



# -------------------
# Dataset: GSM
# Model: LLAMA (word)
# -------------------
sbatch scripts/slurm/large.slurm torchrun --nproc_per_node=4 train.py --epochs 5 --logging_strategy epoch --logging_steps 1 --eval_strategy epoch --eval_steps 1 --generate_strategy epoch --tokenizer_type word --num_layers 1 --dropout 0 --top_k 200 --num_beams 1 --batch_size 8 --seq_len 128 --dataset gsm8k --model_type llama --model_head base --horizon 1 --horizon_eval 1 --rank 1 --init_method random --hidden_dim 768 --num_layers 2 --use_layer_norm --lr 1e-4