#!/usr/bin/env python3
"""Simple memory usage comparison for CP tensor functions."""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tjdnet.distributions._base import BaseDistConfig
from tjdnet.distributions.cp import CPDist
from utils.perf import get_peak_memory_usage


# sns.set_theme(style="whitegrid")
sns.set_theme()


def forward_pass(
    batch_size: int,
    horizon: int,
    rank: int,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    embedding_dim=256,
    vocab_size=30000,
    backward: bool = False,
):
    config = BaseDistConfig(
        vocab_size=vocab_size,
        horizon=horizon,
        rank=rank,
        embedding_dim=embedding_dim,
        positivity_func="sigmoid",
    )
    model = CPDist(config).to(device)
    x = torch.randn(batch_size, embedding_dim, device=device)
    y = torch.randint(0, vocab_size, (batch_size, horizon), device=device)
    loss = model(x, y)

    if backward:
        loss = loss.mean()
        loss.backward()

    return loss


def main():

    # Default values
    defaults = {
        "batch_size": 2,
        "horizon": 2,
        "rank": 2,
        "embedding_dim": 256,
        "vocab_size": 30000,
        "backward": False,
    }
    max_exp = 10
    max_exp_embedding_dim = 3
    configs = (
        [
            {**defaults, "batch_size": 2**i * defaults["batch_size"]}
            for i in range(max_exp)
        ]
        + [{**defaults, "horizon": 2**i * defaults["horizon"]} for i in range(max_exp)]
        + [{**defaults, "rank": 2**i * defaults["rank"]} for i in range(max_exp)]
        + [
            {**defaults, "vocab_size": 2**i * defaults["vocab_size"]}
            for i in range(max_exp)
        ]
        + [
            {**defaults, "embedding_dim": 2**i * defaults["embedding_dim"]}
            for i in range(max_exp_embedding_dim)
        ]
    )

    # Add backward pass
    configs = configs + [
        {**conf, "backward": True} for conf in configs if "backward" not in conf
    ]

    # Collect data
    data = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for conf in configs:
        memory = get_peak_memory_usage(forward_pass, device=device, **conf)
        data.append({**conf, "memory_mb": memory})
    df = pd.DataFrame(data)

    # Create plot
    variable_params = ["batch_size", "horizon", "rank", "embedding_dim"]
    fig, axes = plt.subplots(1, len(variable_params), figsize=(15, 5))
    for i, param in enumerate(variable_params):
        df_filtered = df.copy()
        fixed_params = {k: v for k, v in defaults.items() if k != param}
        for fixed_param in fixed_params.keys():
            df_filtered = df_filtered[
                df_filtered[fixed_param] == fixed_params[fixed_param]
            ]

        sns.lineplot(
            data=df_filtered[df_filtered["backward"] == False],  # type: ignore
            x=param,
            y="memory_mb",
            ax=axes[i],
            marker="o",
            label="Forward",
        )
        sns.lineplot(
            data=df_filtered[df_filtered["backward"] == True],  # type: ignore
            x=param,
            y="memory_mb",
            ax=axes[i],
            marker="x",
            label="Forward + Backward",
        )
        axes[i].set_title(f'Memory vs {param.replace("_", " ").title()}')
        axes[i].set_ylabel("Memory (MB)")
        axes[i].set_xlabel("Param Multiplier")
        axes[i].set_xscale("log", base=2)
        axes[i].set_ylim(0, 4000)

    # save
    plt.savefig("results/plots/memory_usage_facetgrid.png")
    print(f"Plot saved to results/plots/memory_usage_facetgrid.png")


if __name__ == "__main__":
    main()
