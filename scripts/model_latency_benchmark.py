"""
Simple PyTorch model latency benchmark using CUDA events.
Compares inference time between tensor model and baseline.

Usage:
    python model_latency_benchmark.py --model cp  # For CP model
    python model_latency_benchmark.py --model mps  # For MPS model
"""

import os
import sys
import argparse
import torch


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
    return parser.parse_args()


def measure_latency(model, inputs, labels, num_runs, warmup_runs):
    # Warmup runs
    for _ in range(warmup_runs):
        _ = model(inputs, labels=labels)

    # Create CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Measure latency
    latencies = []
    for _ in range(num_runs):
        start_event.record()
        _ = model(inputs, labels=labels)
        end_event.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()

        # Gets time in milliseconds
        latencies.append(start_event.elapsed_time(end_event))

    avg_latency = sum(latencies) / len(latencies)
    return avg_latency


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available. This script requires GPU.")
        return

    # Model configuration
    seq_len = 32
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
        "cuda"
    )
    baseline_model = TJDGPT2(**shared_config, model="base", horizon=1).to("cuda")

    # Prepare inputs
    inputs = torch.randint(0, shared_config["vocab_size"], (1, seq_len)).cuda()
    labels = torch.randint(0, shared_config["vocab_size"], (1, seq_len)).cuda()

    print(f"\nMeasuring latency...")
    print(f"Model type: {args.model}")
    print(f"Number of runs: {args.num_runs}")
    print(f"Warmup runs: {args.warmup_runs}\n")

    # Measure latencies
    tensor_latency = measure_latency(
        tensor_model, inputs, labels, args.num_runs, args.warmup_runs
    )
    baseline_latency = measure_latency(
        baseline_model, inputs, labels, args.num_runs, args.warmup_runs
    )

    # Print results
    print("\nResults:")
    print("-" * 50)
    print(f"Tensor Model latency: {tensor_latency:.2f} ms")
    print(f"Baseline Model latency: {baseline_latency:.2f} ms")
    print(
        f"Speedup (baseline/tensor): {baseline_latency/tensor_latency * args.horizon:.2f}x"
    )


if __name__ == "__main__":
    main()
