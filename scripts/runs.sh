# Baseline
# python train.py --model base --rank 1 --horizon 1 --horizon_eval 1

# Vary Rank
# python train.py --model cp --rank 2 --horizon 2 --horizon_eval 2
# python train.py --model cp --rank 8 --horizon 2 --horizon_eval 2
# python train.py --model cp --rank 16 --horizon 2 --horizon_eval 2

# python train.py --model mps --rank 2 --horizon 2 --horizon_eval 2
# python train.py --model mps --rank 8 --horizon 2 --horizon_eval 2
# python train.py --model mps --rank 16 --horizon 2 --horizon_eval 2


# Vary Horizon
# python train.py --model cp --rank 2 --horizon 2 --horizon_eval 2
python train.py --model cp --rank 2 --horizon 4 --horizon_eval 4
python train.py --model cp --rank 2 --horizon 8 --horizon_eval 8

# python train.py --model mps --rank 2 --horizon 2 --horizon_eval 2
python train.py --model mps --rank 2 --horizon 4 --horizon_eval 4
python train.py --model mps --rank 2 --horizon 8 --horizon_eval 8