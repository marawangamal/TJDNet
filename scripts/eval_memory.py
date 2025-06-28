#!/usr/bin/env python3
"""
Simple memory comparison script for CP tensor functions.
"""

import sys
import os

from mtllama.modeling_mtllama import MultiTokenLlama, MultiTokenLlamaConfig
from tjdnet.models.tjdsimple import TJDSimple, TJDSimpleConfig
from tjdnet.types import ModelHeadType

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import gc
import psutil


def forward_backward_tjdnet(
    horizon,
    rank,
    model="distilbert/distilgpt2",
    model_head: ModelHeadType = "cp",
    device="cuda" if torch.cuda.is_available() else "cpu",
    batch_size=1,
    seq_len=10,
):
    tjd_model = TJDSimple(
        config=TJDSimpleConfig(
            model_name=model,
            model_head=model_head,
            horizon=horizon,
            rank=rank,
        )
    )
    tjd_model.to(device)

    input_ids = torch.randint(
        0, tjd_model.vocab_size, (batch_size, seq_len), device=device
    )
    # Use input_ids as labels for loss computation
    output = tjd_model(input_ids, labels=input_ids)

    # Run backward pass
    output.loss.backward()


def forward_backward(
    horizon,
    model="distilbert/distilgpt2",
    device="cuda" if torch.cuda.is_available() else "cpu",
    batch_size=1,
    seq_len=10,
):
    mtllama_model = MultiTokenLlama(
        config=MultiTokenLlamaConfig(
            model_name=model,
            horizon=horizon,
        )
    )
    mtllama_model.to(device)  # type: ignore

    input_ids = torch.randint(
        0, mtllama_model.vocab_size, (batch_size, seq_len), device=device
    )
    # Use input_ids as labels for loss computation
    output = mtllama_model(input_ids, labels=input_ids)

    # Run backward pass
    output.loss.backward()


def measure_memory(func, *args, **kwargs):
    """Measure peak memory usage during function execution."""
    # Extract device from function arguments
    device = "cpu"
    for i, arg in enumerate(args):
        if isinstance(arg, str) and arg in ["cuda", "cpu"]:
            device = arg
            break

    if device == "cuda" and torch.cuda.is_available():
        # CUDA memory measurement
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()

        initial_memory = torch.cuda.memory_allocated()
        result = func(*args, **kwargs)
        peak_memory = torch.cuda.max_memory_allocated()

        torch.cuda.empty_cache()
        gc.collect()

        return result, peak_memory
    else:
        # CPU memory measurement
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        result = func(*args, **kwargs)
        peak_memory = process.memory_info().rss

        # Force garbage collection
        gc.collect()

        return result, peak_memory


def main():
    horizon = 4
    rank = 1
    model = "distilbert/distilgpt2"
    model_head: ModelHeadType = "cp"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8
    seq_len = 32

    print(f"Running memory evaluation on {device.upper()}")
    print("=" * 50)

    exps = {
        "tjdnet": {
            "func": forward_backward_tjdnet,
            "args": (horizon, rank, model, model_head, device, batch_size, seq_len),
        },
        "mtllama": {
            "func": forward_backward,
            "args": (horizon, model, device, batch_size, seq_len),
        },
    }

    for framework, exp in exps.items():
        print(f"Running {framework}...")
        _, peak_mem = measure_memory(exp["func"], *exp["args"])

        if device == "cuda":
            print(f"{framework} peak mem: {peak_mem/1024**2:.2f} MB")
        else:
            print(f"{framework} peak mem: {peak_mem/1024**2:.2f} MB (RAM)")


if __name__ == "__main__":
    main()
