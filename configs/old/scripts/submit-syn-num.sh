# -------------------
# Dataset: Synthetic Templates
# Model: GPT2 (word)
# -------------------
# Base
# sbatch scripts/slurm/large.slurm torchrun --nproc_per_node=4 

# Base
python train.py --epochs 5 --logging_strategy epoch --logging_steps 1 --eval_strategy epoch --eval_steps 1 --generate_strategy epoch --tokenizer_type word --num_layers 1 --dropout 0 --top_k 200 --num_beams 1 --batch_size 32 --seq_len 256 --dataset snum --model_type gpt2 --model_head base --horizon 1 --horizon_eval 1 --rank 1 --init_method random --lr 1e-3

# CP (H=2)
python train.py --epochs 5 --logging_strategy epoch --logging_steps 1 --eval_strategy epoch --eval_steps 1 --generate_strategy epoch --tokenizer_type word --num_layers 1 --dropout 0 --top_k 200 --num_beams 1 --batch_size 32 --seq_len 256 --dataset snum --model_type gpt2 --model_head cp --horizon 2 --horizon_eval 2 --rank 2 --init_method random --lr 1e-3
python train.py --epochs 5 --logging_strategy epoch --logging_steps 1 --eval_strategy epoch --eval_steps 1 --generate_strategy epoch --tokenizer_type word --num_layers 1 --dropout 0 --top_k 200 --num_beams 1 --batch_size 32 --seq_len 256 --dataset snum --model_type gpt2 --model_head cp --horizon 2 --horizon_eval 2 --rank 4 --init_method random --lr 1e-3