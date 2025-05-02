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

# python scripts/eval_acc_bl.py --model meta-llama/Llama-3.2-3B-Instruct --use_few_shot --dataset gsm8k --max_new_tokens 512
# python scripts/eval_acc_bl.py --model meta-llama/Llama-3.2-3B-Instruct --use_few_shot --dataset gsm8k --max_new_tokens 2048


# jobrunner --dev --job "python scripts/eval_acc.py -c checkpoints/e5_bs8_sl128_l5e_05_ws100_gcv1_0_mmeta_llama_Llama_3_2_3B_Instruct_mhcp_hd8192_r8_h2_pfexp_imrandom_tmlora_lr32_umelTrue_he2_mnt128_tk200_ussFalse_dgsm8k_lsepoch_lo1_esepoch_ev1_gsepoch_ge1000_mns10000_eoFalse_caFalse_abs1_wi950729ce" -p configs/slurm/dev-main-g1.txt
# jobrunner --dev --job "python scripts/eval_acc.py -c checkpoints/e5_bs8_sl128_l5e_05_ws100_gcv1_0_mmeta_llama_Llama_3_2_3B_Instruct_mhcp_hd5120_r1_h3_pfexp_imrandom_tmlora_lr32_umelTrue_he3_mnt128_tk200_ussFalse_dgsm8k_lsepoch_lo1_esepoch_ev1_gsepoch_ge1000_mns10000_eoFalse_caFalse_abs1_wi57cee77b" -p configs/slurm/dev-main-g1.txt
# jobrunner --dev --job "python scripts/eval_acc.py -c checkpoints/e5_bs8_sl128_l5e_05_ws100_gcv1_0_mmeta_llama_Llama_3_2_3B_Instruct_mhcp_hd5120_r1_h2_pfexp_imrandom_tmlora_lr32_umelTrue_he2_mnt128_tk200_ussFalse_dgsm8k_lsepoch_lo1_esepoch_ev1_gsepoch_ge1000_mns10000_eoFalse_caFalse_abs1_wi7259d4b4" -p configs/slurm/dev-main-g1.txt
# jobrunner --dev --job "python scripts/eval_acc.py -c checkpoints/e5_bs8_sl128_l5e_05_ws100_gcv1_0_mmeta_llama_Llama_3_2_3B_Instruct_mhcpo_hd5120_r1_h3_pfexp_imrandom_tmlora_lr32_umelTrue_he3_mnt128_tk200_ussFalse_dgsm8k_lsepoch_lo1_esepoch_ev1_gsepoch_ge1000_mns10000_eoFalse_caFalse_abs1_wi5c72a680" -p configs/slurm/dev-main-g1.txt
# jobrunner --dev --job "python scripts/eval_acc.py -c checkpoints/e5_bs8_sl128_l5e_05_ws100_gcv1_0_mmeta_llama_Llama_3_2_3B_Instruct_mhcpo_hd5120_r1_h2_pfexp_imrandom_tmlora_lr32_umelTrue_he2_mnt128_tk200_ussFalse_dgsm8k_lsepoch_lo1_esepoch_ev1_gsepoch_ge1000_mns10000_eoFalse_caFalse_abs1_wie119d715" -p configs/slurm/dev-main-g1.txt


# jobrunner --dev --job "accelerate launch --use_fsdp --config_file configs/fsdp/fsdp_4gpus.yaml train.py --dataset gsm8k --model meta-llama/Llama-3.2-3B-Instruct --epochs 5 --batch_size 8 --seq_len 128 --lr 5e-5 --model_head cp --hidden_dim 8192 --horizon 2 --horizon_eval 2 --rank 8" -p configs/slurm/dev-short-unkillable-g4.txt