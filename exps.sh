# python train.py --dataset stemp --model_type gpt2 --epochs 10 --batch_size 8 --seq_len 128 --lr 1e-5 --model_head cp --hidden_dim 768 --horizon 4 --horizon_eval 4 --rank 2 --compute_acc
# python train.py --dataset stemp --model_type gpt2 --epochs 10 --batch_size 8 --seq_len 128 --lr 1e-5 --model_head mps --hidden_dim 128 --horizon 2 --horizon_eval 2 --rank 2 --compute_acc


python train.py --dataset stemp --model_type gpt2 --epochs 10 --batch_size 8 --seq_len 128 --lr 1e-5 --model_head cp --hidden_dim 768 --horizon 3 --horizon_eval 3 --rank 2 --compute_acc
python train.py --dataset stemp --model_type gpt2 --epochs 10 --batch_size 8 --seq_len 128 --lr 1e-5 --model_head mps --hidden_dim 768 --horizon 2 --horizon_eval 2 --rank 2 --compute_acc
python train.py --dataset stemp --model_type gpt2 --epochs 10 --batch_size 8 --seq_len 128 --lr 1e-5 --model_head mps --hidden_dim 768 --horizon 3 --horizon_eval 3 --rank 2 --compute_acc