# python train.py --model base --horizon 1

# python train.py --model cp --rank 2 --horizon 2
# python train.py --model cp --rank 4 --horizon 2
# python train.py --model cp --rank 8 --horizon 2
# python train.py --model cp --rank 16 --horizon 2
# python train.py --model cp --rank 32 --horizon 2
# python train.py --model cp --rank 64 --horizon 2

# python train.py --model cp --rank 4 --horizon 3
# python train.py --model cp --rank 4 --horizon 4
# python train.py --model cp --rank 4 --horizon 6
# python train.py --model cp --rank 4 --horizon 8
# python train.py --model cp --rank 4 --horizon 16
# python train.py --model cp --rank 4 --horizon 32

# python scripts/model_latency_benchmark.py --model cp --max_new_tokens 128 --horizon 2
# python scripts/model_latency_benchmark.py --model cp --max_new_tokens 128 --horizon 4
# python scripts/model_latency_benchmark.py --model cp --max_new_tokens 128 --horizon 8
# python scripts/model_latency_benchmark.py --model cp --max_new_tokens 128 --horizon 16
# python scripts/model_latency_benchmark.py --model cp --max_new_tokens 128 --horizon 32
# python scripts/model_latency_benchmark.py --model cp --max_new_tokens 128 --horizon 64

# python scripts/model_latency_benchmark.py --model cp --horizon 8 --max_new_tokens 128
# python scripts/model_latency_benchmark.py --model cp --horizon 8 --max_new_tokens 256
# python scripts/model_latency_benchmark.py --model cp --horizon 8 --max_new_tokens 512
# python scripts/model_latency_benchmark.py --model cp --horizon 8 --max_new_tokens 1024
# python scripts/model_latency_benchmark.py --model cp --horizon 8 --max_new_tokens 2048


# python train.py --model mps --rank 2 --horizon 2
python train.py --model mps --rank 4 --horizon 2
python train.py --model mps --rank 8 --horizon 2
python train.py --model mps --rank 16 --horizon 2
python train.py --model mps --rank 32 --horizon 2
python train.py --model mps --rank 64 --horizon 2