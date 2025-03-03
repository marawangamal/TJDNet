#!/usr/bin/env python3
"""
Checkpoint Evaluation Script (eval.py)

This script evaluates all checkpoints in a specified directory for a trained model,
computing accuracy metrics for each checkpoint and saving the results to a CSV file.

Usage:
    python eval.py --checkpoint_dir [checkpoint_dir]

Example:
    python eval.py --checkpoint_dir checkpoints

Hardware Requirements:
    Same as train.py requirements for inference
"""

import json
import os
import argparse
import torch
from tqdm import tqdm


from callbacks.eval_gsm8k import compute_accuracy
from helpers import (
    get_model_and_tokenizer,
    get_chat_template,
)
from data.gsm8k import load_gsm8k_data
from data.shakespeare import load_shakespeare_data
from data.sharegptv2 import load_sharegptv2_data
from data.syn_number_bases import load_syn_num_base_data
from data.syn_numbers import load_syn_num_data
from data.syn_temp import load_syn_temp_data
from data.wikitext import load_wikitext_data


def load_weights(model, checkpoint_path):
    checkpoint_file = os.path.join(checkpoint_path, "pytorch_model.bin")
    checkpoint = torch.load(checkpoint_file, map_location="cpu")

    # If checkpoint is a state_dict wrapper (from Trainer)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    return model


def main():
    # Parse just our evaluation-specific arguments
    parser = argparse.ArgumentParser(description="Evaluate model checkpoints")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory containing the model checkpoint and args.json",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for evaluation",
    )
    args = parser.parse_args()

    checkpoints = os.listdir(args.checkpoint_dir)
    results = []
    for checkpoint in tqdm(checkpoints, desc="Evaluating checkpoints"):

        # 1. Setup
        exp_args_dict = json.load(
            open(os.path.join(args.checkpoint_dir, checkpoint, "args.json"))
        )
        exp_args = argparse.Namespace(**exp_args_dict)
        model, tokenizer = get_model_and_tokenizer(exp_args)
        chat_template = get_chat_template(exp_args)
        lm_dataset = {
            "shakespeare": load_shakespeare_data,
            "wikitext": load_wikitext_data,
            "sharegpt": load_sharegptv2_data,
            "gsm8k": load_gsm8k_data,
            "stemp": load_syn_temp_data,
            "snum": load_syn_num_data,
            "sbase": load_syn_num_base_data,
        }[args.dataset](tokenizer, args.seq_len, max_num_samples=args.max_num_samples)

        # 2. Load weights
        model = load_weights(model, checkpoint)
        model.to(args.device)
        acc = compute_accuracy(
            model,
            tokenizer=tokenizer,
            test_dataset=lm_dataset,
            eos_token=tokenizer.eos_token,
            chat_template=chat_template,
            horizon=exp_args.horizon,
            top_k=exp_args.top_k,
            num_beams=exp_args.num_beams,
        )
        print(f"Eval accuracy: {acc} for checkpoint: {checkpoint}")
        results.append(acc)


if __name__ == "__main__":
    main()
