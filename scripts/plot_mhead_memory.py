#!/usr/bin/env python3
"""Simple memory usage comparison for CP tensor functions."""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import gc
from tjdnet.distributions._base import BaseDistConfig
from tjdnet.distributions.cp import CPDist


def measure_memory_usage(
    batch_size,
    horizon,
    rank,
    embedding_dim=256,
    vocab_size=30000,
    backward=False,
):
    """Measure peak memory usage for a single configuration."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Create model and run forward pass
    config = BaseDistConfig(
        vocab_size=vocab_size,
        horizon=horizon,
        rank=rank,
        embedding_dim=embedding_dim,
        positivity_func="safe_exp",
    )
    model = CPDist(config).to(device)
    x = torch.randn(batch_size, embedding_dim, device=device)
    y = torch.randint(0, vocab_size, (batch_size, horizon), device=device)
    loss = model(x, y)
    if backward:
        loss.backward()

    # Measure memory
    if device == "cuda":
        memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
        torch.cuda.empty_cache()
    else:
        import psutil

        memory_mb = psutil.Process().memory_info().rss / (1024**2)

    # reset peak memory stats
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    else:
        import psutil

        psutil.Process().memory_info().rss

    gc.collect()
    return memory_mb


def generate_data():
    """Generate memory data for different parameters."""
    data = []

    defaults = {
        "batch_size": 2,
        "horizon": 2,
        "rank": 2,
        "embedding_dim": 256,
        "vocab_size": 30000,
    }

    variables = {
        "batch_size": [2, 4, 8, 16, 32],
        "horizon": [2, 4, 8, 16, 32],
        "rank": [2, 4, 8, 16, 32],
    }

    for param, values in variables.items():
        for value in values:
            kwargs = defaults.copy()
            kwargs[param] = value
            memory = measure_memory_usage(**kwargs)
            data.append({**kwargs, "memory_mb": memory})

    return pd.DataFrame(data)


def plot_memory_usage(df, save_path="results/plots/memory_usage_facetgrid.png"):
    """Create seaborn visualization of memory usage."""
    sns.set_theme(style="whitegrid")

    # Create subplots for each parameter
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, param in enumerate(["batch_size", "horizon", "rank"]):
        sns.lineplot(
            data=df.groupby(param)["memory_mb"].mean().reset_index(),
            x=param,
            y="memory_mb",
            marker="o",
            ax=axes[i],
        )
        axes[i].set_title(f'Memory vs {param.replace("_", " ").title()}')
        axes[i].set_ylabel("Memory (MB)")

    plt.suptitle("Peak Memory Usage vs Model Parameters", y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    df = generate_data()
    plot_memory_usage(df)
