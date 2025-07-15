#!/usr/bin/env python3
"""Simple memory usage comparison for CP tensor functions."""

from typing import Literal
from tqdm import tqdm

import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tjdnet.distributions import TJD_DISTS
from tjdnet.distributions._base import BaseDistConfig
from tjdnet.distributions.cp import CPDist
from utils.perf import get_peak_memory_usage


sns.set_theme()


def train_fn(
    batch_size: int,
    horizon: int,
    rank: int,
    model_head: str = "cp",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    embedding_dim=256,
    vocab_size=30000,
    mode: Literal["init", "forward", "backward"] = "init",
    **kwargs,
):
    config = BaseDistConfig(
        vocab_size=vocab_size,
        horizon=horizon,
        rank=rank,
        embedding_dim=embedding_dim,
        positivity_func="sigmoid",
    )
    model = TJD_DISTS[model_head](config).to(device)
    x = torch.randn(batch_size, embedding_dim, device=device)
    y = torch.randint(0, vocab_size, (batch_size, horizon), device=device)

    if mode in ["forward", "backward"]:
        loss = model(x, y)
        if mode in ["backward"]:
            loss = loss.mean()
            loss.backward()


def main():

    # Default values
    defaults = {
        "batch_size": 2,
        "horizon": 2,
        "rank": 2,
        "embedding_dim": 128,
        "vocab_size": 30000,
    }
    max_exp = 10
    max_exp_embedding_dim = 5

    # Attrs:
    # col: hparam (batch_size, horizon, rank, embedding_dim)
    # x: multiplier
    # y: memory_mb
    # hue: mode (init, init + forward, init + forward + backward)
    # style: model_head (cp, cp_drop)

    kwargs = []
    for mode in ["init", "forward", "backward"]:
        for head in ["cp", "cp_drop"]:
            for hparam in ["batch_size", "horizon", "rank", "embedding_dim"]:
                T = max_exp_embedding_dim if hparam == "embedding_dim" else max_exp
                for i in range(T):
                    kwargs.append(
                        {
                            **defaults,
                            "hparam": hparam,
                            "multiplier": 2**i,
                            "mode": mode,
                            "model_head": head,
                            hparam: 2**i * defaults[hparam],
                        }
                    )

    # Compute memory
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for conf in tqdm(kwargs, desc="Running forward pass"):
        memory = get_peak_memory_usage(train_fn, device=device, **conf)
        conf["memory_mb"] = memory

    df = pd.DataFrame(kwargs)
    sns.relplot(
        data=df,
        x="multiplier",
        y="memory_mb",
        col="hparam",
        style="model_head",
        hue="mode",
        kind="line",
        markers=True,
        alpha=0.6,
    )
    plt.xscale("log", base=2)
    plt.savefig("results/plots/memory_usage_facetgrid.png")
    print(f"Plot saved to results/plots/memory_usage_facetgrid.png")


if __name__ == "__main__":
    main()
