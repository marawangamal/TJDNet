# python scripts/datasets/create_hf_tjdnet_ds.py --horizon 2 --model meta-llama/Llama-2-7b-chat-hf
# python scripts/datasets/create_hf_tjdnet_ds.py --horizon 4 --model meta-llama/Llama-2-7b-chat-hf
# python scripts/datasets/create_hf_tjdnet_ds.py --horizon 8 --model meta-llama/Llama-2-7b-chat-hf
# python scripts/datasets/create_hf_tjdnet_ds.py --horizon 16 --model meta-llama/Llama-2-7b-chat-hf
# python scripts/datasets/create_hf_tjdnet_ds.py --horizon 32 --model meta-llama/Llama-2-7b-chat-hf
# python scripts/datasets/create_hf_tjdnet_ds.py --horizon 2 --model gpt2
# python scripts/datasets/create_hf_tjdnet_ds.py --horizon 4 --model gpt2

# python scripts/plots/plot_output_dist_recons_error.py --init_method normal
# python scripts/plots/plot_output_dist_recons_error.py --init_method zeros

# python scripts/plots/plot_output_dist_recons_error.py --init_method normal --use_log
# python scripts/plots/plot_output_dist_recons_error.py --init_method zeros  --use_log

python scripts/eval_acc_bl.py --model meta-llama/Llama-3.2-3B-Instruct --use_few_shot --dataset gsm8k --max_new_tokens 512
python scripts/eval_acc_bl.py --model meta-llama/Llama-3.2-3B-Instruct --use_few_shot --dataset gsm8k --max_new_tokens 2048