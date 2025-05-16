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

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from dataloaders import CHAT_TEMPLATES, DATASET_LOADERS
from utils.accuracy import compute_accuracy
from utils.helpers import get_auto_tokenizer, get_model_and_tokenizer
from dataloaders.gsm8k import load_gsm8k_data
from dataloaders.shakespeare import load_shakespeare_data
from dataloaders.sharegpt import load_sharegpt
from dataloaders.syn_number_bases import load_syn_num_base_data
from dataloaders.syn_numbers import load_syn_num_data
from dataloaders.syn_temp import load_syn_temp_data
from dataloaders.wikitext import load_wikitext_data
from utils.experiment_naming import get_experiment_name


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
        "-m",
        "--model",
        type=str,
        default=None,
        help="Hugging Face model identifier (default: gpt2)",
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
        "--use_few_shot",
        action="store_true",
        default=False,
        help="Use few-shot examples for evaluation",
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_LOADERS.keys(),
        default=None,
    )

    # ===== Generation kwargs =====
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
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
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # 1. Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        low_cpu_mem_usage=True,
    )
    tokenizer = get_auto_tokenizer(args.model)
    # NOTE: do not do this for training, otherwise the model cannot learn to end sents
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2. Load ds and chat template
    lm_dataset = DATASET_LOADERS[args.dataset](
        tokenizer=tokenizer, use_few_shot=args.use_few_shot
    )
    chat_template = CHAT_TEMPLATES.get(args.dataset)

    # 3. Create results file path
    results_dir = "results/eval_acc"
    os.makedirs(results_dir, exist_ok=True)
    exp_name = get_experiment_name(vars(args))
    results_file = osp.join(results_dir, f"{exp_name}.json")

    assert chat_template is not None, f"Chat template not found for {args.dataset }"

    average_meter_kwargs = {"sum": 0, "count": 0}
    if osp.exists(results_file):
        print(f"Initalizing results from {results_file}")
        with open(results_file) as f:
            average_meter_kwargs = json.load(f)

    model.to(args.device)
    acc, avg_meter_kwargs = compute_accuracy(
        model,
        tokenizer=tokenizer,
        test_dataset=lm_dataset["test"],
        chat_template=chat_template,
        batch_size=args.batch_size,
        avg_meter_kwargs=average_meter_kwargs,
        on_batch_end=lambda new_avg_meter: save_results_checkpoint(
            new_avg_meter, results_file
        ),
        generate_kwargs=dict(
            do_sample=True,
            max_new_tokens=args.max_new_tokens,
            top_k=args.top_k,
            top_p=args.top_p,
        ),
        # log_samples=True, # used for debugging
    )
    save_results_checkpoint(avg_meter_kwargs, results_file)
    print(f"Eval accuracy: {acc} for model: {args.model}")
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
