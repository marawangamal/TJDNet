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


sns.set_theme(style="whitegrid")


def forward_pass(
    batch_size: int,
    horizon: int,
    rank: int,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    embedding_dim=256,
    vocab_size=30000,
    backward_pass: bool = False,
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

    if backward_pass:
        loss = loss.mean()
        loss.backward()

    return loss


def parse_args():
    parser = argparse.ArgumentParser(
        description="Memory usage comparison for CP tensor functions"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Parameters to test
    mult_factors = [2**i for i in range(10)]

    params = {
        "batch_size": mult_factors,
        "horizon": mult_factors,
        "rank": mult_factors,
        "embedding_dim": [2, 4, 8],
    }

    # Default values
    defaults = {
        "batch_size": 2,
        "horizon": 2,
        "rank": 2,
        "embedding_dim": 256,
        "vocab_size": 30000,
    }

    # Collect data for both forward-only and forward+backward
    data = []
    for backward_pass in [False, True]:
        for param, values in params.items():
            for value in values:
                kwargs = defaults.copy()
                kwargs[param] = value * defaults[param]
                kwargs["backward_pass"] = backward_pass
                device = "cuda" if torch.cuda.is_available() else "cpu"
                memory = get_peak_memory_usage(forward_pass, device=device, **kwargs)
                data.append(
                    {**kwargs, "memory_mb": memory, "backward_pass": backward_pass}
                )

    df = pd.DataFrame(data)

    # Create plot
    fig, axes = plt.subplots(1, len(params), figsize=(15, 5))

    for i, param in enumerate(params.keys()):
        # Filter data for this parameter
        df_filtered = df.copy()
        for other_param in params.keys():
            if other_param != param:
                df_filtered = df_filtered[
                    df_filtered[other_param] == defaults[other_param]
                ]

        # Ensure df_filtered is a DataFrame, not a Series
        if isinstance(df_filtered, pd.Series):
            df_filtered = df_filtered.to_frame().T

        # Plot both forward-only and forward+backward
        sns.lineplot(
            data=df_filtered[df_filtered["backward_pass"] == False],
            x=param,
            y="memory_mb",
            ax=axes[i],
            marker="o",
            label="Forward Only",
        )
        sns.lineplot(
            data=df_filtered[df_filtered["backward_pass"] == True],
            x=param,
            y="memory_mb",
            ax=axes[i],
            marker="s",
            label="Forward + Backward",
        )

        axes[i].set_title(f'Memory vs {param.replace("_", " ").title()}')
        axes[i].set_ylabel("Memory (MB)")
        axes[i].set_xlabel("Param Multiplier")
        axes[i].legend()

        # axis
        axes[i].set_xscale("log", base=2)
        axes[i].set_ylim(0, 4000)

    plt.suptitle("Peak Memory Usage vs Model Parameters", y=1.02)
    plt.tight_layout()

    # Save plot
    save_path = "results/plots/memory_usage_facetgrid.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    main()
