"""
Generates code completions for HumanEval problems and saves them to samples.jsonl.

Usage:
1. Generate: python eval/generate_completions.py --ckpt /path/to/checkpoint
2. Evaluate: python eval/human-eval/human_eval/evaluate_functional_correctness.py samples.jsonl
"""

import os
import os.path as osp
import argparse
from tqdm import tqdm

import torch

from human_eval.data import read_problems, write_jsonl
from helpers import (
    get_model_and_tokenizer,
    get_test_samples,
    load_args,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Path to checkpoint directory"
    )
    parser.add_argument(
        "--dev", action="store_true", help="Run in quick development mode"
    )
    parser.add_argument(
        "-m",
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate for each completion",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Number of tokens to generate at each step",
    )
    return parser.parse_args()


def find_latest_checkpoint(ckpt_dir):
    """Find the latest checkpoint subdirectory in the given directory."""
    try:
        subdirs = [d for d in os.listdir(ckpt_dir) if d.startswith("checkpoint-")]
        if not subdirs:
            raise FileNotFoundError(f"No checkpoint directories found in {ckpt_dir}.")
        # Sort subdirectories by the step number (e.g., "checkpoint-5000")
        subdirs.sort(key=lambda x: int(x.split("-")[-1]), reverse=True)
        return osp.join(ckpt_dir, subdirs[0])
    except FileNotFoundError as e:
        return None


def load_model(ckpt_dir):
    saved_args = load_args(ckpt_dir)
    model, tokenizer, chat_template = get_model_and_tokenizer(
        argparse.Namespace(**saved_args)
    )
    latest_ckpt_dir = find_latest_checkpoint(ckpt_dir)

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if latest_ckpt_dir is not None:
        ckpt_path = osp.join(latest_ckpt_dir, "pytorch_model.bin")

        if not osp.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found at {ckpt_path}")

        # Load model state dict and move to device
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        print(f"Loaded model from {ckpt_path}")
    else:
        print(f"No checkpoint found in {ckpt_dir}. Using a fresh model.")
    model = model.to(device)
    model.eval()

    return model, tokenizer, chat_template


def generate_one_completion(prompt, model, tokenizer, max_new_tokens=256, horizon=1):
    completetion = get_test_samples(
        model,
        tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        horizon=horizon,
    )
    return completetion


def main():
    args = parse_args()
    model, tokenizer, chat_template = load_model(ckpt_dir=args.ckpt)
    problems = read_problems()

    # Limit problems and samples for development mode
    if args.dev:
        print("\nRunning in development mode: limiting problems and samples...")
        problems = dict(list(problems.items())[:2])  # Limit to 2 problems
        num_samples_per_task = 2  # Limit to 2 samples per task
    else:
        num_samples_per_task = 200

    samples = []

    # Use tqdm for the outer loop to track progress across problems
    for task_id in tqdm(problems, desc="Processing problems"):
        # Use tqdm for the inner loop to track samples per problem
        for _ in tqdm(
            range(num_samples_per_task), desc=f"Samples for {task_id}", leave=False
        ):
            prompt = chat_template.format_prompt(problems[task_id]["prompt"])
            samples.append(
                dict(
                    task_id=task_id,
                    prompt=prompt,
                    completion=generate_one_completion(
                        prompt,
                        model,
                        tokenizer,
                        max_new_tokens=args.max_new_tokens,
                        horizon=args.horizon,
                    ),
                )
            )

    write_jsonl("samples.jsonl", samples)
    print(f"\nGenerated {len(samples)} completions and saved to samples.jsonl")


if __name__ == "__main__":
    main()
