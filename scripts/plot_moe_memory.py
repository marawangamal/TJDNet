#!/usr/bin/env python3
"""Minimal memory usage comparison for MOE layers with different numbers of experts."""

import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tjdnet.layers import MOELayer
from utils.perf import get_latency, get_peak_memory_usage


sns.set_theme()


def train_fn(
    batch_size: int,
    seq_len: int,
    input_dim: int,
    hidden_dim: int,
    num_experts: int,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    mode: str = "forward",
    **kwargs,
):
    """Training function for memory comparison."""

    model = MOELayer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        num_experts_per_token=2,
    ).to(device)

    x = torch.randn(batch_size, seq_len, input_dim, device=device)

    if mode in ["forward", "backward"]:
        output, _ = model(x)
        if mode == "backward":
            loss = output.sum()
            loss.backward()


def main():
    # Default values
    defaults = {
        "batch_size": 32,
        "seq_len": 128,
        "input_dim": 512,
        "hidden_dim": 1024,
        "num_experts": 4,
    }

    max_exps = {"num_experts": 5}

    # Create configs first
    kwargs = []
    for mode in ["init", "forward", "backward"]:
        for hparam, T in max_exps.items():
            for i in range(T):
                # Vary the parameter
                param_value = 2**i * defaults[hparam]

                # For each parameter value, test different numbers of experts
                for num_experts_exp in range(max_exps["num_experts"]):
                    num_experts = 2**num_experts_exp
                    kwargs.append(
                        {
                            **defaults,
                            "hparam": hparam,
                            "multiplier": 2**i,
                            "mode": mode,
                            "num_experts": num_experts,
                            hparam: param_value,
                        }
                    )

    # Compute memory
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for conf in kwargs:
        memory_mb = get_peak_memory_usage(train_fn, device=device, **conf)
        latency_sec = get_latency(train_fn, device=device, **conf)
        conf["memory_mb"] = memory_mb
        conf["latency_sec"] = latency_sec

    df = pd.DataFrame(kwargs)

    # Plot memory usage
    sns.relplot(
        data=df,
        x="num_experts",
        y="memory_mb",
        col="hparam",
        hue="mode",
        kind="line",
        markers=True,
        alpha=0.6,
    )
    plt.xscale("log", base=2)
    plt.savefig("results/plots/moe_memory_usage_facetgrid.png")
    print(f"Plot saved to results/plots/moe_memory_usage_facetgrid.png")

    # # Plot latency
    # sns.relplot(
    #     data=df,
    #     x="num_experts",
    #     y="latency_sec",
    #     col="hparam",
    #     hue="mode",
    #     kind="line",
    #     markers=True,
    #     alpha=0.6,
    # )
    # plt.xscale("log", base=2)
    # plt.savefig("results/plots/moe_latency_facetgrid.png")
    # print(f"Plot saved to results/plots/moe_latency_facetgrid.png")


if __name__ == "__main__":
    main()
