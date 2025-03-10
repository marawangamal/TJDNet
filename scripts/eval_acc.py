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

import os
import os.path as osp
import json
import argparse
from typing import List
import torch
from tqdm import tqdm

from utils.accuracy import compute_accuracy
from utils.helpers import (
    get_model_and_tokenizer,
    get_chat_template,
)
from data.gsm8k import load_gsm8k_data
from data.shakespeare import load_shakespeare_data
from data.sharegpt import load_sharegpt
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


def save_results_checkpoint(results, file_path):
    with open(file_path, "w") as f:
        json.dump(results, f)
    print(f"Results saved to {file_path}")


def main():
    # Parse just our evaluation-specific arguments
    parser = argparse.ArgumentParser(description="Evaluate model checkpoints")
    parser.add_argument(
        "-c",
        "--checkpoint",
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
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "-s",
        "--max_num_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate",
    )
    parser.add_argument(
        "-t",
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=200,
        help="Top-k value for sampling",
    )
    args = parser.parse_args()

    checkpoints: List[str] = [
        osp.join(args.checkpoint, c)
        for c in os.listdir(args.checkpoint)
        if c.startswith("checkpoint")
    ]
    exp_args_dict = json.load(open(os.path.join(args.checkpoint, "args.json")))

    # 1. Setup
    exp_args = argparse.Namespace(**exp_args_dict)
    model, tokenizer = get_model_and_tokenizer(exp_args)
    chat_template = get_chat_template(exp_args)
    lm_dataset = {
        "shakespeare": load_shakespeare_data,
        "wikitext": load_wikitext_data,
        "sharegpt": load_sharegpt,
        "gsm8k": load_gsm8k_data,
        "stemp": load_syn_temp_data,
        "snum": load_syn_num_data,
        "sbase": load_syn_num_base_data,
    }[exp_args.dataset](
        tokenizer, exp_args.seq_len, max_num_samples=exp_args.max_num_samples
    )

    results = {}
    results_file = osp.join(
        args.checkpoint,
        f"eval_results_b{args.batch_size}_s{args.max_num_samples}_t{args.max_new_tokens}.json",
    )
    if osp.exists(results_file):
        print(f"Initalizing results from {results_file}")
        with open(results_file) as f:
            results = json.load(f)

    for checkpoint in tqdm(checkpoints, desc="Evaluating checkpoints"):
        model = load_weights(model, checkpoint)
        model.to(args.device)
        average_meter_kwargs = results.get(checkpoint, {"sum": 0, "count": 0})
        acc, avg_meter_kwargs = compute_accuracy(
            model,
            tokenizer=tokenizer,
            test_dataset=lm_dataset["test"],
            chat_template=chat_template,
            horizon=exp_args.horizon,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            max_num_samples=args.max_num_samples,
            avg_meter_kwargs=average_meter_kwargs,
            on_batch_end=lambda avg_meter_kwargs: save_results_checkpoint(
                {**results, checkpoint: avg_meter_kwargs}, results_file
            ),
        )
        results[checkpoint] = avg_meter_kwargs
        print(f"Eval accuracy: {acc} for checkpoint: {checkpoint}")

    save_results_checkpoint(results, results_file)
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
