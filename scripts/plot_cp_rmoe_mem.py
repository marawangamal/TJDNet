import gc
import itertools
import os
from matplotlib import pyplot as plt
import psutil
from typing import Callable

import seaborn as sns
import torch
import pandas as pd
import tqdm

from tjdnet.distributions._base import BaseDistConfig
from tjdnet.distributions.cp_rmoe import CPRMoEDist

sns.set_theme()


def get_peak_memory_usage(fn: Callable, **kwargs) -> float:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        fn(**kwargs)

        memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
        torch.cuda.empty_cache()
    else:
        # For CPU: measure memory difference
        gc.collect()
        memory_before = psutil.Process().memory_info().rss / (1024**2)

        fn(**kwargs)

        gc.collect()
        memory_after = psutil.Process().memory_info().rss / (1024**2)
        memory_mb = max(0, memory_after - memory_before)  # Ensure non-negative

    return memory_mb


def fn(batch_size=10, mode="init", **kwargs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(batch_size, kwargs["config"].embedding_dim, device=device)
    y = torch.randint(
        0,
        kwargs["config"].vocab_size,
        (batch_size, kwargs["config"].horizon),
        device=device,
    )
    model = CPRMoEDist(**kwargs)
    model.to(device)

    # fw pass
    if mode in ["forward", "backward"]:
        loss = model(x, y)
        if mode in ["backward"]:
            # bw pass
            loss = loss.mean()
            loss.backward()


def main():

    common_kwargs = {
        "vocab_size": 1000,
        "horizon": 8,
        "rank": 32,
        "rank_active": 2,
        "embedding_dim": 512,
        "positivity_func": "exp",
    }
    configs = []

    for conf in itertools.product(
        [2**t for t in range(1, 8)], ["init", "forward", "backward"]
    ):
        bconf = BaseDistConfig(**common_kwargs)
        # update
        bconf.rank = conf[0]
        bconf.rank_active = conf[0]
        configs.append(
            {
                "config": bconf,
                "rank": conf[0],
                "mode": conf[1],
                "col": "rank == rank_active",
            }
        )

    for conf in itertools.product(
        [2**t for t in range(1, 8)], ["init", "forward", "backward"]
    ):
        bconf = BaseDistConfig(**common_kwargs)
        # update
        bconf.rank = conf[0]
        configs.append(
            {
                "config": bconf,  # type: ignore
                "rank": conf[0],
                "mode": conf[1],
                "col": "rank",
            }
        )

    for conf in tqdm.tqdm(configs):
        conf["mem_mb"] = get_peak_memory_usage(fn, **conf)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # plot
    df = pd.DataFrame(configs)
    sns.relplot(
        data=df,
        x="rank",
        y="mem_mb",
        kind="line",
        style="mode",
        col="col",
        markers=True,
        alpha=0.6,
    )
    plt.xscale("log", base=2)

    save_path = "results/plots/cp_rmoe_memory_plot.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    print(df)


if __name__ == "__main__":
    main()
