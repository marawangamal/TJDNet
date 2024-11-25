# python train_clm_shakespeare_char_tjd.py --model tgpt2 --horizon 2 --horizon_eval 2 --rank 2
# python train_clm_shakespeare_char_tjd.py --model tgpt2 --horizon 2 --horizon_eval 2 --rank 4
# python train_clm_shakespeare_char_tjd.py --model tgpt2 --horizon 2 --horizon_eval 2 --rank 6
# python train_clm_shakespeare_char_tjd.py --model tgpt2 --horizon 2 --horizon_eval 2 --rank 8
# python train_clm_shakespeare_char_tjd.py --model tgpt2 --horizon 2 --horizon_eval 2 --rank 10
# python train_clm_shakespeare_char_tjd.py --model tgpt2 --horizon 2 --horizon_eval 2 --rank 12
# python train_clm_shakespeare_char_tjd.py --model tgpt2 --horizon 2 --horizon_eval 2 --rank 16


# python train_clm_shakespeare_char_tjd.py --model tgpt2 --horizon 2 --horizon_eval 1 --rank 2  --lr 1e-3
# python train_clm_shakespeare_char_tjd.py --model tgpt2 --horizon 2 --horizon_eval 1 --rank 2  --lr 1e-4
# python train_clm_shakespeare_char_tjd.py --model tgpt2 --horizon 2 --horizon_eval 1 --rank 2  --lr 1e-5
# python train_clm_shakespeare_char_tjd.py --model tgpt2 --horizon 2 --horizon_eval 1 --rank 2  --lr 1e-6
# python train_clm_shakespeare_char_tjd.py --model tgpt2 --horizon 2 --horizon_eval 1 --rank 4  --lr 1e-5
# python train_clm_shakespeare_char_tjd.py --model tgpt2 --horizon 2 --horizon_eval 1 --rank 4
# python train_clm_shakespeare_char_tjd.py --model tgpt2 --horizon 2 --horizon_eval 1 --rank 6
# python train_clm_shakespeare_char_tjd.py --model tgpt2 --horizon 2 --horizon_eval 1 --rank 8
# python train_clm_shakespeare_char_tjd.py --model tgpt2 --horizon 2 --horizon_eval 1 --rank 10
# python train_clm_shakespeare_char_tjd.py --model tgpt2 --horizon 2 --horizon_eval 1 --rank 12
# python train_clm_shakespeare_char_tjd.py --model tgpt2 --horizon 2 --horizon_eval 1 --rank 16



# Sanity check (these should be roughly the same)
# python train_clm_shakespeare_char_tjd.py --model cp --horizon 1 --horizon_eval 1 --rank 1  --lr 1e-3
# python train_clm_shakespeare_char_tjd.py --model gpt2 --lr 1e-3

# Comparison of baseline, materialized and tjd
# python train_clm_shakespeare_char_tjd.py --model gpt2 1e-3  # baseline
# python train_clm_shakespeare_char_tjd.py --model mgpt2 --horizon 2 --horizon_eval 1 --lr 1e-3   # materialized
# python train_clm_shakespeare_char_tjd.py --model tgpt2 --horizon 2 --horizon_eval 1 --rank 1  --lr 1e-3 #tjd

python train.py --model cp --rank 2
python train.py --model cp --rank 4
python train.py --model cp --rank 8
python train.py --model cp --rank 16
python train.py --model cp --rank 32
python train.py --model cp --rank 64