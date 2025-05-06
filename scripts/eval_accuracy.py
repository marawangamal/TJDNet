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

from dataloaders import CHAT_TEMPLATES, DATASET_LOADERS
from utils.accpetance_rates import compute_acceptance_rate
from utils.accuracy import compute_accuracy
from utils.helpers import get_model_and_tokenizer


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


def parse_args():
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
        "--metric",
        type=str,
        choices=["accuracy", "acceptance_rate"],
        default="accuracy",
        help="Metric to compute during evaluation",
    )
    # ===== Generation kwargs =====
    parser.add_argument(
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
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p value for sampling",
    )
    # === Early stopping ===
    parser.add_argument(
        "-m",
        "--max_num_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    metric_fn = {
        "accuracy": compute_accuracy,
        "acceptance_rate": compute_acceptance_rate,
    }[args.metric]

    checkpoints: List[str] = [
        osp.join(args.checkpoint, c)
        for c in os.listdir(args.checkpoint)
        if c.startswith("checkpoint")
    ]
    if len(checkpoints) == 0:
        print(f"No checkpoints found in {args.checkpoint}.")
        return

    exp_args_dict = json.load(open(os.path.join(args.checkpoint, "args.json")))

    # 1. Setup
    exp_args = argparse.Namespace(**exp_args_dict)
    if args.metric == "acceptance_rate":
        exp_args.use_speculative_sampling = True
    model, tokenizer = get_model_and_tokenizer(exp_args)
    chat_template = CHAT_TEMPLATES[exp_args.dataset]
    lm_dataset = DATASET_LOADERS[exp_args.dataset](tokenizer, exp_args.seq_len)

    results = {}
    results_file = osp.join(
        args.checkpoint,
        f"eval_results_{args.metric}.json",
    )
    if osp.exists(results_file):
        print(f"Initalizing results from {results_file}")
        with open(results_file) as f:
            results = json.load(f)

    print(f"Evaluating metric: {args.metric}")
    print("Using device:", args.device)
    for checkpoint in tqdm(checkpoints, desc="Evaluating checkpoints"):
        model = load_weights(model, checkpoint)
        model.to(args.device)
        average_meter_kwargs = results.get(checkpoint, {"sum": 0, "count": 0})
        metric_ds_val, metric_avg_meter_kwargs = metric_fn(
            model,
            tokenizer=tokenizer,
            test_dataset=lm_dataset["test"],
            chat_template=chat_template,
            batch_size=args.batch_size,
            avg_meter_kwargs=average_meter_kwargs,
            on_batch_end=lambda avg_meter_kwargs: save_results_checkpoint(
                {**results, checkpoint: avg_meter_kwargs}, results_file
            ),
            log_samples=True,
            # max_num_samples=args.max_num_samples,
            # horizon=exp_args.horizon,
            # top_k=args.top_k,
            # max_new_tokens=args.max_new_tokens,
            generate_kwargs=dict(
                do_sample=True,
                max_new_tokens=args.max_new_tokens,
                top_k=args.top_k,
                horizon=exp_args.horizon,
            ),
            max_num_samples=args.max_num_samples,
        )
        results[checkpoint] = metric_avg_meter_kwargs
        print(f"Eval {args.metric}: {metric_ds_val} for checkpoint: {checkpoint}")

    save_results_checkpoint(results, results_file)
    print(f"Results saved to {results_file}")
    # Print best eval metric achieved
    best_acc = max([r["avg"] for r in results.values()])
    print(f"Eval {args.metric} (best): {best_acc} for exp: {args.checkpoint}")


if __name__ == "__main__":
    main()
