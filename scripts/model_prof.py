"""
Simple PyTorch model profiler with CUDA support.
Profiles model performance and memory usage on either CPU or GPU.

Usage:
    python profiler.py --device cuda  # For GPU profiling
    python profiler.py --device cpu   # For CPU profiling
"""

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import argparse
import torch
import torch.autograd.profiler as profiler
from models.tjdgpt2.tjdgpt2 import TJDGPT2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="cp",
        choices=["cp", "mps"],
        help="Model type (cp or mps)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to run on",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Initialize model
    model_config = {
        "model": args.model,
        "vocab_size": 128,
        "n_embd": 64,
        "n_layer": 2,
        "n_head": 2,
        "dropout": 0.1,
        "rank": 2,
        "horizon": 2,
    }

    # Create model and move to device
    model = TJDGPT2(**model_config).to(args.device)

    # Create inputs
    seq_len = 32
    inputs = torch.randint(0, model_config["vocab_size"], (1, seq_len)).to(args.device)
    labels = torch.randint(0, model_config["vocab_size"], (1, seq_len)).to(args.device)

    # Warmup
    model(input_ids=inputs, labels=labels)

    # Profile using compatible syntax
    with profiler.profile(
        use_cuda=(args.device == "cuda"), profile_memory=True, with_stack=True
    ) as prof:
        loss = model(inputs, labels)
        if args.device == "cuda":
            torch.cuda.synchronize()

    # Print results
    print("\nProfiling Results:")
    print(
        prof.key_averages().table(
            sort_by="self_cpu_time_total" if args.device == "cpu" else "cuda_time_total"
        )
    )
