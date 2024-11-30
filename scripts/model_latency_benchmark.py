"""
Simple PyTorch model latency benchmark.
Compares inference time between tensor model and baseline.
Works on both CPU and CUDA devices.

Usage:
    python scripts/model_latency_benchmark.py --model cp  # For CP model
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
    parser.add_argument(
        "--model",
        type=str,
        default="cp",
        choices=["cp", "mps"],
        help="Model type (cp or mps)",
    )
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
        "--horizon",
        type=int,
        default=2,
        help="Number of warmup runs",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2,
        help="Number of warmup runs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to run on (cuda/cpu)",
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
            _ = model.generate(inputs, max_new_tokens=max_new_tokens)
            end_event.record()

            torch.cuda.synchronize()
            latencies.append(start_event.elapsed_time(end_event))
        else:
            # Use time.perf_counter for CPU timing
            start_time = time.perf_counter()
            _ = model.generate(inputs, max_new_tokens=max_new_tokens)
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to milliseconds

    avg_latency = sum(latencies) / len(latencies)
    return avg_latency


def main():
    args = parse_args()

    print(f"\nUsing device: {args.device}")

    # Model configuration
    seq_len = 1
    shared_config = {
        "vocab_size": 128,
        "n_embd": 64,
        "n_layer": 2,
        "n_head": 2,
        "dropout": 0.1,
        "rank": 2,
    }

    # Initialize models
    tensor_model = TJDGPT2(**shared_config, model=args.model, horizon=args.horizon).to(
        args.device
    )
    baseline_model = TJDGPT2(**shared_config, model="base", horizon=1).to(args.device)

    # Prepare inputs
    inputs = torch.randint(0, shared_config["vocab_size"], (1, seq_len)).to(args.device)
    labels = torch.randint(0, shared_config["vocab_size"], (1, seq_len)).to(args.device)

    print(f"\nMeasuring latency...")
    print(f"Model type: {args.model}")
    print(f"Number of runs: {args.num_runs}")
    print(f"Warmup runs: {args.warmup_runs}\n")

    # Measure latencies
    tensor_latency = measure_latency(
        tensor_model,
        inputs,
        labels,
        args.num_runs,
        args.warmup_runs,
        args.max_new_tokens,
        args.device,
    )
    baseline_latency = measure_latency(
        baseline_model,
        inputs,
        labels,
        args.num_runs,
        args.warmup_runs,
        args.max_new_tokens,
        args.device,
    )

    # Print results
    print("\nResults:")
    print("-" * 50)
    print(f"Tensor Model latency: {tensor_latency:.2f} ms")
    print(f"Baseline Model latency: {baseline_latency:.2f} ms")
    print(f"Speedup (baseline/tensor): {baseline_latency/tensor_latency:.2f}x")


if __name__ == "__main__":
    main()
