# python train_ttdist_synthetic.py --vocab_size 16 --true_rank 18 --rank 2 --loss_type entropy --output_size 4
# python train_ttdist_synthetic.py --vocab_size 16 --true_rank 18 --rank 4 --loss_type entropy --output_size 4
# python train_ttdist_synthetic.py --vocab_size 16 --true_rank 18 --rank 6 --loss_type entropy --output_size 4
# python train_ttdist_synthetic.py --vocab_size 16 --true_rank 18 --rank 8 --loss_type entropy --output_size 4
# python train_ttdist_synthetic.py --vocab_size 16 --true_rank 18 --rank 10 --loss_type entropy --output_size 4


python train_ttdist_synthetic.py --true_dist uniform --vocab_size 32 --true_rank 32 --rank 2 --loss_type entropy --output_size 3