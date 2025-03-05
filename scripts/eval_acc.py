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
from git import Optional
import torch
from tqdm import tqdm

from torch.utils.data import DataLoader
from transformers import (
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


from utils.callbacks.eval_gsm8k import compute_accuracy
from utils.train_helpers import (
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


# def compute_accuracy_v2(
#     model,
#     tokenizer,
#     test_dataset,
#     eos_token,
#     chat_template,
#     batch_size=32,
#     device="cuda" if torch.cuda.is_available() else "cpu",
# ):
#     model.eval()
#     # will set attention mask = 0 for tokens matching tokenizer.pad_token_id
#     data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
#     dataloader = DataLoader(
#         test_dataset,
#         shuffle=True,
#         collate_fn=data_collator,
#         batch_size=batch_size,
#     )

#     for batch in dataloader:
#         batch = {k: v.to(device) for k, v in batch.items()}
#         input_ids = batch["input_ids"]


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
    args = parser.parse_args()

    checkpoints = [
        osp.join(args.checkpoint, c)
        for c in os.listdir(args.checkpoint)
        if c != "args.json"
    ]
    exp_args_dict = json.load(open(os.path.join(args.checkpoint, "args.json")))

    # 1. Setup
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
    }[exp_args.dataset](
        tokenizer, exp_args.seq_len, max_num_samples=exp_args.max_num_samples
    )

    results = []
    for checkpoint in tqdm(checkpoints, desc="Evaluating checkpoints"):
        model = load_weights(model, checkpoint)
        model.to(args.device)
        acc = compute_accuracy(
            model,
            tokenizer=tokenizer,
            test_dataset=lm_dataset["test"],
            eos_token=tokenizer.eos_token,
            chat_template=chat_template,
            horizon=exp_args.horizon,
            top_k=exp_args.top_k,
            num_beams=exp_args.num_beams,
            max_num_samples=None,
        )
        print(f"Eval accuracy: {acc} for checkpoint: {checkpoint}")
        results.append(acc)

    # Save results to CSV
    results_file = osp.join(args.checkpoint, "eval_results.csv")
    with open(results_file, "w") as f:
        f.write("checkpoint,accuracy\n")
        for checkpoint, acc in zip(checkpoints, results):
            f.write(f"{checkpoint},{acc}\n")


if __name__ == "__main__":
    main()
