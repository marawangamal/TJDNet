python train.py --epochs 5 --logging_strategy epoch --logging_steps 1 --eval_strategy epoch --eval_steps 1 --generate_strategy epoch --tokenizer_type char --num_layers 1 --dropout 0 --top_k 200 --num_beams 1 --batch_size 64 --seq_len 256 --dataset gsm8k --model_type gpt2 --model_head base --horizon 1 --horizon_eval 1 --rank 1 --init_method random --lr 1e-3



# --epochs 5 --logging_strategy epoch --logging_steps 1 --eval_strategy epoch --eval_steps 1 --generate_strategy epoch --tokenizer_type char --num_layers 1 --dropout 0 --top_k 200 --num_beams 1 --batch_size 8 --seq_len 8 --dataset gsm8k --model_type gpt2 --model_head base --horizon 1 --horizon_eval 1 --rank 1 --init_method random --lr 1e-3
