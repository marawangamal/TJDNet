# python train.py --batch_size 2
# python train.py --batch_size 8
# python train.py --batch_size 2 --max_seq_len 3 
# python train.py --batch_size 8 --max_seq_len 3

python train.py --batch_size 2  --lr 1e-2
python train.py --batch_size 8 --lr 1e-2
python train.py --batch_size 2 --max_seq_len 3  --lr 1e-2
python train.py --batch_size 8 --max_seq_len 3 --lr 1e-2

python train.py --batch_size 2  --tjd_mode tjd-bounded
python train.py --batch_size 8  --tjd_mode tjd-bounded
python train.py --batch_size 2 --max_seq_len 3  --tjd_mode tjd-bounded
python train.py --batch_size 8 --max_seq_len 3  --tjd_mode tjd-bounded
