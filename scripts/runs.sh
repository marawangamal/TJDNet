# For character level models these ranks get same eval/nll as baseline
python train.py --tokenizer_type char --model base --rank 1 --horizon 1 --horizon_eval 1
python train.py --tokenizer_type char --model cp --rank 4 --horizon 2 --horizon_eval 2
python train.py --tokenizer_type char --model mps --rank 4 --horizon 2 --horizon_eval 2
python train.py --tokenizer_type char --model umps --rank 4 --horizon 2 --horizon_eval 2