# python train.py --model base --rank 1 --horizon 1 --horizon_eval 1

python train.py --model cp --rank 2 --horizon 2 --horizon_eval 2
python train.py --model cp --rank 16 --horizon 2 --horizon_eval 2
python train.py --model cp --rank 64 --horizon 2 --horizon_eval 2

# python train.py --model cp --rank 2 --horizon 2 --horizon_eval 1
# python train.py --model cp --rank 16 --horizon 2 --horizon_eval 1
# python train.py --model cp --rank 64 --horizon 2 --horizon_eval 1




# python train.py --model cp --rank 2 --horizon 2 --horizon_eval 2
# python train.py --scale_loss --model cp --rank 2 --horizon 2 --horizon_eval 2

# python train.py --model cp --rank 64 --horizon 2 --horizon_eval 2
# python train.py --scale_loss --model cp --rank 64 --horizon 2 --horizon_eval 2