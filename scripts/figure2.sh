python train_ttdist_synthetic.py --loss_type entropy --output_size 2
python train_ttdist_synthetic.py --loss_type entropy --output_size 4
python train_ttdist_synthetic.py --loss_type entropy --output_size 6
python train_ttdist_synthetic.py --loss_type entropy --output_size 8
python train_ttdist_synthetic.py --loss_type entropy --output_size 10

python train_ttdist_synthetic.py --loss_type preference --output_size 2 --init_method uniform_positive  
python train_ttdist_synthetic.py --loss_type preference --output_size 4 --init_method uniform_positive
python train_ttdist_synthetic.py --loss_type preference --output_size 6 --init_method uniform_positive
python train_ttdist_synthetic.py --loss_type preference --output_size 8 --init_method uniform_positive
python train_ttdist_synthetic.py --loss_type preference --output_size 10 --init_method uniform_positive