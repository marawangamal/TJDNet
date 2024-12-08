"""
Simple PyTorch model latency benchmark.
Compares inference time between tensor model and baseline.
Works on both CPU and CUDA devices.

Usage:
    python scripts/model_latency_benchmark.py --rank 2 --horizon 8 --vocab_size 50000
    python scripts/model_latency_benchmark.py --model mps  # For MPS model
"""

import os
import sys
import argparse
import torch
import time


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from models.tjdgpt2.tjdgpt2 import TJDGPT2


def parse_args():
    parser = argparse.ArgumentParser()
    # Existing arguments
    parser.add_argument(
        "--num_runs",
        type=int,
        default=100,
        help="Number of inference runs for averaging",
    )
    parser.add_argument(
        "--warmup_runs",
        type=int,
        default=10,
        help="Number of warmup runs",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=2,
        help="Rank of the MPS/CP model",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=2,
        help="Horizon of the MPS/CP model",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=128,
        help="Sequence length for input tensor",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Sequence length for input tensor",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="generate",
        help="Mode to benchmark (generate or train)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to run on (cuda/cpu)",
    )

    # Model configuration arguments
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=128,
        help="Vocabulary size for the model",
    )
    parser.add_argument(
        "--n_embd",
        type=int,
        default=64,
        help="Embedding dimension",
    )
    parser.add_argument(
        "--n_layer",
        type=int,
        default=2,
        help="Number of transformer layers",
    )
    parser.add_argument(
        "--n_head",
        type=int,
        default=2,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability",
    )
    return parser.parse_args()


def get_test_sample(
    model: torch.nn.Module,
    inp: torch.Tensor,
    max_new_tokens: int = 8,
    horizon: int = 1,
):
    model.eval()
    outputs = model.generate(
        inp,
        max_new_tokens=max_new_tokens,
        horizon=horizon,
    )
    return outputs


def measure_latency(
    model,
    inputs,
    labels,
    num_runs,
    warmup_runs,
    max_new_tokens=8,
    device="cuda",
    mode: str = "generate",  # "generate" or "train"
):
    # Warmup runs
    for _ in range(warmup_runs):
        _ = model(inputs, labels=labels)

    # Measure latency
    latencies = []
    for _ in range(num_runs):
        if device == "cuda":
            # Use CUDA events for GPU timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            if mode == "train":
                _ = model(inputs, labels=labels)
            else:
                _ = model.generate(inputs, max_new_tokens=max_new_tokens)
            end_event.record()

            torch.cuda.synchronize()
            latencies.append(start_event.elapsed_time(end_event))
        else:
            # Use time.perf_counter for CPU timing
            start_time = time.perf_counter()
            if mode == "train":
                _ = model(inputs, labels=labels)
            else:
                _ = model.generate(inputs, max_new_tokens=max_new_tokens)
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to milliseconds

    avg_latency = sum(latencies) / len(latencies)
    std_latency = torch.tensor(latencies).std().item()
    return avg_latency, std_latency


def print_latency_results(latency_results):
    # Get the width of the longest model name
    name_width = max(len(name) for name in latency_results)

    # Print header
    print("\nLatency Results")
    print(f"{'Model':<{name_width}} | {'Mean (ms)':>10} ± {'Std (ms)':>10}")
    print("-" * (name_width + 25))  # Adjust line length

    # Print each result
    for model_name, (mean, std) in latency_results.items():
        print(f"{model_name:<{name_width}} | {mean:>10.2f} ± {std:>10.2f}")


def main():
    args = parse_args()

    print(f"\nUsing device: {args.device}")

    # Model configuration
    model_config = {
        "vocab_size": args.vocab_size,
        "n_embd": args.n_embd,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "dropout": args.dropout,
        "rank": args.rank,
    }

    # Initialize models
    models = {
        k: TJDGPT2(**model_config, model=k, horizon=h).to(args.device)
        for k, h in [("base", 1), ("cp", args.horizon), ("mps", args.horizon)]
    }
    latency_results = {}

    # Prepare inputs
    inputs = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len)).to(
        args.device
    )
    labels = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len)).to(
        args.device
    )

    for model_name, model in models.items():
        print(f"Measuring latency for {model_name} model...")
        latency = measure_latency(
            model,
            inputs,
            labels,
            args.num_runs,
            args.warmup_runs,
            args.max_new_tokens,
            args.device,
            mode=args.mode,
        )
        latency_results[model_name] = latency

    # Print results in tabular format
    print_latency_results(latency_results)


if __name__ == "__main__":
    main()
