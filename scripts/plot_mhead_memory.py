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
from utils.perf import get_peak_memory_usage


sns.set_theme(style="whitegrid")


def cp_forward(
    batch_size: int,
    horizon: int,
    rank: int,
    embedding_dim=256,
    vocab_size=30000,
    backward: bool = False,
):
    """Measure peak memory usage for a single configuration."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
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


if __name__ == "__main__":
    # Set up parameters
    defaults = {
        "batch_size": 2,
        "horizon": 2,
        "rank": 2,
        "embedding_dim": 256,
        "vocab_size": 30000,
        "backward": False,
    }
    variables = {
        "batch_size": [2, 4, 8, 16],
        "horizon": [2, 4, 8, 16],
        "rank": [2, 4, 8, 16],
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Compute memory usage
    data = []
    for param, values in variables.items():
        for value in values:
            kwargs = defaults.copy()
            kwargs[param] = kwargs[param] * value
            model = cp_forward(**kwargs)
            memory = get_peak_memory_usage(cp_forward, device=device, **kwargs)
            data.append({**kwargs, "memory_mb": memory})
    df = pd.DataFrame(data)

    # Plot memory usage
    save_path = "results/plots/memory_usage_facetgrid.png"
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, param in enumerate(["batch_size", "horizon", "rank"]):
        # Filter out rows where other two parameters are not at defaults
        fixed_params = {k: v for k, v in defaults.items() if k != param}
        df_filtered = df.copy()
        for fixed_param, fixed_value in fixed_params.items():
            df_filtered = df_filtered[df_filtered[fixed_param] == fixed_value]

        sns.lineplot(
            data=df_filtered,
            x=param,
            y="memory_mb",
            ax=axes[i],
        )
        axes[i].set_title(f'Memory vs {param.replace("_", " ").title()}')
        axes[i].set_ylabel("Memory (MB)")
        axes[i].set_xlabel(param.replace("_", " ").title())
        axes[i].set_xticks(sorted(df_filtered[param].unique()))

    plt.suptitle("Peak Memory Usage vs Model Parameters", y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
