# Base
python train.py --dataset stemp --model_type gpt2 --epochs 10 --batch_size 32 --seq_len 256 --lr 1e-4 --model_head base --compute_acc


# CP
python train.py --dataset sharegpt --model_type gpt2 --epochs 10 --batch_size 32 --seq_len 256 --lr 1e-4 --model_head cp --num_layers 2 --hidden_dim 768  --horizon 2 --horizon_eval 2 --rank 2
