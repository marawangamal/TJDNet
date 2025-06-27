#!/usr/bin/env python3
"""
Simple memory comparison script for CP tensor functions.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import gc
from tjdnet.tensorops.cp import (
    select_margin_cp_tensor_batched,
    select_margin_cp_tensor_batched_w_decoder,
)


def measure_memory(func, *args, **kwargs):
    """Measure peak memory usage during function execution."""
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    initial_memory = torch.cuda.memory_allocated()
    result = func(*args, **kwargs)
    peak_memory = torch.cuda.max_memory_allocated()

    torch.cuda.empty_cache()
    gc.collect()

    return result, peak_memory - initial_memory


def main():
    """Run simple memory comparison."""
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping memory tests.")
        return

    print(
        "Memory Comparison: select_margin_cp_tensor_batched vs select_margin_cp_tensor_batched_w_decoder"
    )
    print("=" * 70)

    # Random parameters
    batch_size, rank, horizon, vocab_size = 16, 8, 6, 16000
    hidden_dim = 256

    device = torch.device("cuda")

    # Random test data
    cp_params_regular = torch.randn(
        batch_size, rank, horizon, vocab_size, device=device
    )
    cp_params_decoder = torch.randn(
        batch_size, rank, horizon, hidden_dim, device=device
    )
    decoder = torch.randn(hidden_dim, vocab_size, device=device)
    ops = torch.randint(-2, vocab_size, (batch_size, horizon), device=device)

    # Measure memory for both functions
    (_, _), mem_regular = measure_memory(
        select_margin_cp_tensor_batched, cp_params_regular, ops
    )
    (_, _), mem_decoder = measure_memory(
        select_margin_cp_tensor_batched_w_decoder, cp_params_decoder, ops, decoder
    )

    # Print results
    print(f"\nMemory Usage Comparison:")
    print(f"Regular function: {mem_regular/1024**2:.2f} MB")
    print(f"Decoder function: {mem_decoder/1024**2:.2f} MB")
    print(f"Memory savings: {(mem_regular - mem_decoder)/1024**2:.2f} MB")
    print(f"Memory reduction: {((mem_regular - mem_decoder)/mem_regular*100):.1f}%")


if __name__ == "__main__":
    main()
