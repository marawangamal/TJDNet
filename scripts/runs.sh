# Baseline
# python train_pll.py --model base --rank 1 --horizon 1 --horizon_eval 1

# Horizon 2
# python train_pll.py --batch_size 8 --model cp --rank 2 --horizon 2 --horizon_eval 2
# python train_pll.py --batch_size 8 --model mps --rank 2 --horizon 2 --horizon_eval 2
# python train_pll.py --batch_size 8 --model umps --rank 2 --horizon 2 --horizon_eval 2

# Horizon 4
torchrun --nproc_per_node=4 train_pll.py --batch_size 2 --model cp --rank 16 --horizon 2 --horizon_eval 2
torchrun --nproc_per_node=4 train_pll.py --batch_size 2 --model mps --rank 4 --horizon 2 --horizon_eval 2
torchrun --nproc_per_node=4 train_pll.py --batch_size 2 --model umps --rank 4 --horizon 2 --horizon_eval 2