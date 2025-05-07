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

import gc
import os
import os.path as osp
import json
import argparse
import traceback
from typing import List
import torch
from tqdm import tqdm

from dataloaders import CHAT_TEMPLATES, DATASET_LOADERS
from utils.accpetance_rates import compute_acceptance_rate
from utils.accuracy import compute_accuracy
from utils.helpers import get_model_and_tokenizer
from utils.latency import benchmark_model_v2, get_params
from utils.models import train_forward


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
        help="Directory containing the experiment checkpoints and config args.json",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for evaluation",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval"],
        default="eval",
        help="Mode to run the script in",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "-i",
        "--inp_seq_len",
        type=int,
        default=256,
    )
    parser.add_argument(
        "-o",
        "--out_seq_len",
        type=int,
        default=128,
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
    exp_args_dict = json.load(open(os.path.join(args.experiment, "args.json")))

    # 1. Load model
    exp_args = argparse.Namespace(**exp_args_dict)
    if hasattr(exp_args, "use_speculative_sampling"):
        exp_args.use_speculative_sampling = True
    else:
        exp_args.use_speculative_sampling = False
    model, _ = get_model_and_tokenizer(exp_args)

    gen_kwargs = {
        "max_new_tokens": args.out_seq_len,
        "top_k": args.top_k,
        "do_sample": False,
    }
    exps = [
        # {
        #     "name": "train",
        #     "model_fn": lambda: model,
        #     "benchmark_fn": train_forward,
        # },
        {
            "name": "eval",
            "model_fn": lambda: model,
            "benchmark_fn": lambda model, input_ids: model.generate(
                input_ids, **gen_kwargs
            ),
        },
    ]

    input_ids = torch.randint(0, 100, (args.batch_size, args.inp_seq_len)).to(
        args.device
    )

    results = {
        "config": vars(args),
        # "compute": {
        #     # desc of current compute
        #     "name": os.uname()[1],
        #     "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        #     "gpu_mem": torch.cuda.get_device_properties(0).total_memory
        #     if torch.cuda.is_available()
        #     else "N/A",
        #     "cpu_mem": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
        #     "cpu_cores": os.cpu_count(),
        #     "cpu_freq": os.sysconf("SC_CLK_TCK"),
        #     "cpu_model": os.uname()[4],
        #     "cpu_model_name": os.uname()[0],
        #     "cpu_model_cores": os.sysconf("SC_NPROCESSORS_ONLN"),
        # },
        "modes": {},
    }
    results_file = osp.join(
        args.experiment,
        f"eval_results_latency.json",
    )

    try:
        for exp in exps:
            print(f"\nBenchmarking {exp['name']}...")

            # Build model
            model = exp["model_fn"]().to(args.device)
            # Run benchmark
            benchmark_fn = exp["benchmark_fn"]
            benchmark_results = benchmark_model_v2(
                model, benchmark_fn, benchmark_fn_kwargs={"input_ids": input_ids}
            )

            # Save results
            results["modes"][exp["name"]] = benchmark_results
            results["modes"][exp["name"]]["Params [M]"] = get_params(model)

            # Clean up
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    except Exception as e:
        print(f"Error benchmarking {exp['name']}: {str(e)}")
        traceback.print_exc()  # This will print the full stack trace

    save_results_checkpoint(results, results_file)
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
