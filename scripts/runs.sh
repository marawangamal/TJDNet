# Debugging train modes

# (stemp, full, 1-layer head) Acc: 1
python train.py --epochs 5 --logging_strategy epoch --logging_steps 1 --eval_strategy epoch --eval_steps 1 --generate_strategy epoch --tokenizer_type word --num_layers 1 --dropout 0 --top_k 200 --num_beams 1 --batch_size 32 --seq_len 256 --dataset stemp --model_type gpt2 --model_head base --horizon 1 --horizon_eval 1 --rank 1 --init_method random --lr 1e-4 --train_mode full

# (stemp, full, 2-layer head) Acc: 
python train.py --epochs 5 --logging_strategy epoch --logging_steps 1 --eval_strategy epoch --eval_steps 1 --generate_strategy epoch --tokenizer_type word --num_layers 2 --hidden_size 768 --dropout 0 --top_k 200 --num_beams 1 --batch_size 32 --seq_len 256 --dataset stemp --model_type gpt2 --model_head base --horizon 1 --horizon_eval 1 --rank 1 --init_method random --lr 1e-4 --train_mode full

# (stemp, last, 1-layer head) Acc: 
python train.py --epochs 5 --logging_strategy epoch --logging_steps 1 --eval_strategy epoch --eval_steps 1 --generate_strategy epoch --tokenizer_type word --num_layers 1 --dropout 0 --top_k 200 --num_beams 1 --batch_size 32 --seq_len 256 --dataset stemp --model_type gpt2 --model_head base --horizon 1 --horizon_eval 1 --rank 1 --init_method random --lr 1e-4 --train_mode lora --lora_rank 8

# (stemp, lora8, 1-layer head) Acc: 
python train.py --epochs 5 --logging_strategy epoch --logging_steps 1 --eval_strategy epoch --eval_steps 1 --generate_strategy epoch --tokenizer_type word --num_layers 1 --dropout 0 --top_k 200 --num_beams 1 --batch_size 32 --seq_len 256 --dataset stemp --model_type gpt2 --model_head base --horizon 1 --horizon_eval 1 --rank 1 --init_method random --lr 1e-4 --train_mode lora --lora_rank 32
