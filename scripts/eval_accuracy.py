#!/usr/bin/env python3
"""
Experiment evaluation script.

Usage:
    python scripts/eval_accuracy.py -e [experiments_dir]

Example:
    python scripts/eval_accuracy.py -e experiments

"""

import os
import os.path as osp
import json
import argparse
from typing import List
import torch
from tqdm import tqdm
import wandb

from dataloaders import CHAT_TEMPLATES, DATASET_LOADERS
from tjdnet.models.tjd import TJDGenerationConfig
from utils.accuracy import compute_accuracy
from utils.helpers import get_git_info, get_model_and_tokenizer_v2


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
        "-e",
        "--experiment",
        type=str,
        required=True,
        help="Directory containing experiment checkpoints",
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
    checkpoints: List[str] = [
        osp.join(args.experiment, c)
        for c in os.listdir(args.experiment)
        if c.startswith("checkpoint")
    ]
    if len(checkpoints) == 0:
        print(f"No checkpoints found in {args.experiment}.")
        return

    exp_args_dict = json.load(open(os.path.join(args.experiment, "args.json")))

    # 1. Setup
    exp_args = argparse.Namespace(**exp_args_dict)
    if args.metric == "acceptance_rate":
        exp_args.use_speculative_sampling = True
    model, tokenizer = get_model_and_tokenizer_v2(exp_args)
    chat_template = CHAT_TEMPLATES[exp_args.dataset]
    lm_dataset = DATASET_LOADERS[exp_args.dataset](tokenizer, exp_args.seq_len)

    results = {}
    results_file = osp.join(
        args.experiment,
        f"eval_results_{args.metric}.json",
    )
    if osp.exists(results_file):
        print(f"Initalizing results from {results_file}")
        with open(results_file) as f:
            results = json.load(f)

    print("Using device:", args.device)
    for checkpoint in tqdm(checkpoints, desc="Evaluating checkpoints"):
        model = load_weights(model, checkpoint)
        model.to(args.device)
        metric_dict = compute_accuracy(
            model=model,
            tokenizer=tokenizer,
            dataset=lm_dataset["test"],
            chat_template=chat_template,
            batch_size=args.batch_size,
            generation_config=TJDGenerationConfig(
                do_sample=True,
                max_new_tokens=args.max_new_tokens,
                top_k=args.top_k,
                horizon=exp_args.horizon,
            ),
            ckpt_dir=checkpoint,
        )
        results[checkpoint] = metric_dict

    save_results_checkpoint(results, results_file)
    print(f"Results saved to {results_file}")
    best_acc = max([(k, v["accuracy"]) for k, v in results.items()], key=lambda x: x[1])
    print(f"Eval accuracy (best): {best_acc} for exp: {args.experiment}")

    # ==== Log to wandb
    git_info = get_git_info()
    suffix = "main" if git_info.get("branch") == "main" else "dev"
    project_name = f"{args.wandb_project}-{suffix}"
    wandb.init(
        project=project_name,
        name=os.path.basename(args.experiment),
        id=exp_args.wandb_id,
        resume="allow",
    )
    wandb.log(
        {
            f"eval/{args.metric}": best_acc,
        },
        step=int(best_acc[0].replace("checkpoint-", "")),
    )


if __name__ == "__main__":
    main()
