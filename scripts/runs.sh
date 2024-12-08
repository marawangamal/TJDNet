# Baseline
python train_pll.py --model base --rank 1 --horizon 1 --horizon_eval 1

# Horizon 2
python train_pll.py --model cp --rank 4 --horizon 2 --horizon_eval 2
# python train_pll.py --model cp --rank 16 --horizon 2 --horizon_eval 2

python train_pll.py --model mps --rank 2 --horizon 2 --horizon_eval 2
# python train_pll.py --model mps --rank 4 --horizon 2 --horizon_eval 2


# Horizon 4
# python train.py --model cp --rank 4 --horizon 4 --horizon_eval 4
# python train.py --model cp --rank 16 --horizon 4 --horizon_eval 4
# python train.py --model cp --rank 64 --horizon 4 --horizon_eval 4

# python train.py --model mps --rank 2 --horizon 4 --horizon_eval 4
# python train.py --model mps --rank 4 --horizon 4 --horizon_eval 4
# python train.py --model mps --rank 8 --horizon 4 --horizon_eval 4