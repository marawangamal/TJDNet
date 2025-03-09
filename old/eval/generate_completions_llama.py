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
from transformers import AutoModelForCausalLM, AutoTokenizer
from human_eval.data import read_problems, write_jsonl

from utils.helpers import (
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


def generate_one_completion(prompt, model, tokenizer, eval_horizon=1):
    # UNCOMMENT THIS LINE TO GENERATE COMPLETIONS
    return model.generate(
        tokenizer(prompt, return_tensors="pt").input_ids,
        max_length=len(tokenizer(prompt)["input_ids"]) + eval_horizon,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=1,
    )[0]


def main():
    args = parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
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
            samples.append(
                dict(
                    task_id=task_id,
                    prompt=problems[task_id]["prompt"],
                    completion=generate_one_completion(
                        problems[task_id]["prompt"], model, tokenizer
                    ),
                )
            )

    write_jsonl("samples.jsonl", samples)
    print(f"\nGenerated {len(samples)} completions and saved to samples.jsonl")


if __name__ == "__main__":
    main()
