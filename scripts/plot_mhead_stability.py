import argparse
import itertools
from tqdm import tqdm

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tjdnet.distributions import TJD_DISTS
from tjdnet.distributions._base import BaseDistConfig

sns.set_theme()


def range_exp(start, stop, step=1):
    return [2**t for t in range(start, stop, step)]


def run_train(model, train_dl, device, lr, epochs):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    stats = []
    try:
        for _ in tqdm(range(epochs), desc="Training", leave=False):
            for batch in train_dl:
                batch = [b.to(device) for b in batch]
                loss = model(*batch)
                loss = loss.mean()
                loss.backward()

                # save stats
                stats.append(
                    {
                        "train_losses": loss.item(),
                        "grad_norms": [
                            p.grad.norm(2).item() for p in model.parameters()
                        ],
                        "param_norms": [p.norm(2).item() for p in model.parameters()],
                    }
                )

                # update params
                optimizer.step()
                optimizer.zero_grad()
    except Exception as e:
        print(f"Training failed: {e}")

    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--n_samples", type=int, default=1024)
    args = p.parse_args()

    model_hparams = {
        "horizon": 8,
        "rank": 8,
        "embedding_dim": 512,
        "vocab_size": 1000,
        "positivity_func": "sigmoid",
    }

    # create experiment configs
    configs = []

    # add cp configs
    for model_head, rank in itertools.product(["cp", "cp_cond", "cp_condl"], [1]):
        configs.append(
            {
                "model_head": model_head,
                "model_head_kwargs": {
                    **model_hparams,
                    "rank": rank,
                },
                "rank": rank,
            }
        )

    # add multihead config
    configs.append(
        {
            "model_head": "multihead",
            "model_head_kwargs": {**model_hparams},
            "rank": 1,
        }
    )

    # add dummy dist
    configs.append(
        {
            "model_head": "dummy",
            "model_head_kwargs": {**model_hparams},
            "rank": 1,
        }
    )

    # create synthetic data
    train_ds = torch.utils.data.TensorDataset(
        torch.randn(args.n_samples, model_hparams["embedding_dim"]),
        torch.randint(
            0, model_hparams["vocab_size"], (args.n_samples, model_hparams["horizon"])
        ),
    )
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size)

    for config in tqdm(configs, desc="Computing stats"):
        train_stats = run_train(
            model=TJD_DISTS[config["model_head"]](
                BaseDistConfig(**config["model_head_kwargs"])
            ),
            train_dl=train_dl,
            lr=args.lr,
            epochs=args.epochs,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        config["stats"] = train_stats

    # create a dataframe of the results
    results_df = pd.DataFrame(configs)
    results_df = results_df.explode("stats")
    results_df["timestep"] = results_df.groupby(level=0).cumcount()  # 0, 1, 2…
    results_df = pd.concat(
        [results_df.drop("stats", axis=1), results_df["stats"].apply(pd.Series)], axis=1
    )
    results_df["grad_norms"] = results_df["grad_norms"].apply(np.mean)
    results_df["param_norms"] = results_df["param_norms"].apply(np.mean)

    # create long format dataframe
    long = results_df.melt(
        id_vars=["model_head", "rank", "timestep"],  # keep these
        value_vars=["train_losses", "grad_norms", "param_norms"],
        var_name="metric",  # new col → facet
        value_name="value",  # y‑axis
    )
    g = sns.relplot(
        data=long,
        x="timestep",
        y="value",
        hue="model_head",
        style="rank",
        col="metric",  # facet by metric (use row='metric' if you prefer)
        kind="line",
        facet_kws=dict(sharey=False),  # separate y‑scales (optional)
        alpha=0.5,
    )
    for ax in g.axes.flat:
        ax.set_yscale("log")  # or "log", "symlog", etc.
    plt.savefig("results/plots/mhead_stability.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
