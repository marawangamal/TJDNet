python train_ttdist_synthetic.py --output_size 3
python train_ttdist_synthetic.py --output_size 4
python train_ttdist_synthetic.py --output_size 5
python train_ttdist_synthetic.py --output_size 6
python train_ttdist_synthetic.py --output_size 7
python train_ttdist_synthetic.py --output_size 8
python train_ttdist_synthetic.py --output_size 9
python train_ttdist_synthetic.py --output_size 10
python train_ttdist_synthetic.py --output_size 11

python train_ttdist_synthetic.py --init_method preference --output_size 3 --batch_size 32 --lr 1e-5
python train_ttdist_synthetic.py --init_method preference --output_size 3 --batch_size 32 --lr 1e-4
python train_ttdist_synthetic.py --init_method preference --output_size 3 --batch_size 32 --lr 1e-3 

python train_ttdist_synthetic.py --init_method preference --output_size 3 --batch_size 64 --lr 1e-5
python train_ttdist_synthetic.py --init_method preference --output_size 3 --batch_size 64 --lr 1e-4
python train_ttdist_synthetic.py --init_method preference --output_size 3 --batch_size 64 --lr 1e-3 


