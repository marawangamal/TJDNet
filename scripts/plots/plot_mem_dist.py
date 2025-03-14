import argparse
import gc
import torch
from tqdm import tqdm
import itertools
import numpy as np

from tjdnet.distributions._base import BaseDistConfig
from tjdnet.distributions.tpnet import TensorParamNetConfig
from tjdnet.models._tjd import TJDConfig
from tjdnet.models.tjdgpt2 import TJDGPT2

import matplotlib.pyplot as plt

from tjdnet.models.tjdllama import TJDLLAMA


def get_params(model):
    params = {}
    params_detailed = {}
    for name, param in model.named_parameters():
        key = name.split(".")[0]
        if key not in params:
            params[key] = 0
        params[key] += param.numel()
        params_detailed[name] = param.numel()

    return params, params_detailed


def plot_params_dist(results, path="cp_params.png"):
    width = 0.1  # the width of the bars
    multiplier = 0
    for name, params_dict in results.items():
        offset = width * multiplier
        rects = plt.bar(
            np.arange(len(params_dict)) + offset,
            list(params_dict.values()),
            width,
            label=name,
        )
        # plt.bar_label(rects, padding=3)
        multiplier += 1

    plt.xlabel("Index")
    plt.ylabel("Number of Parameters")
    plt.xticks(
        # np.arange(len(params_dict)) + width * (multiplier - 1) / 2,
        np.arange(len(params_dict)) + width,
        params_dict.keys(),
    )
    plt.legend(loc="upper left", ncols=3)
    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    plt.close()

    # # Extract ranks and horizons
    # ranks = sorted(
    #     list(set([int(k.split("::r")[1].split("::")[0]) for k in results.keys()]))
    # )
    # horizons = sorted(list(set([int(k.split("::h")[1]) for k in results.keys()])))

    # # Create plot
    # fig, ax = plt.subplots(figsize=(10, 6))

    # # Prepare data points
    # for h in horizons:
    #     x_values = []
    #     y_values = []
    #     for r in ranks:
    #         model_name = f"gpt2::r{r}::h{h}"
    #         if model_name in results:
    #             x_values.append(r)
    #             y_values.append(results[model_name])

    #     # Plot line for this horizon
    #     ax.plot(x_values, y_values, marker="o", label=f"h={h}")

    # # Set axis to log scale (better for visualizing parameter scaling)
    # ax.set_xscale("log", base=2)
    # ax.set_yscale("log")

    # # Add grid, labels and legend
    # ax.grid(True, which="both", ls="-", alpha=0.2)
    # ax.set_xlabel("Rank")
    # ax.set_ylabel("Number of Parameters")
    # ax.set_title("Parameter Count by Rank and Horizon")
    # ax.legend()

    # plt.tight_layout()
    # plt.savefig("parameter_scaling.png", dpi=300)


def create_model_gpt_fn(rank, horizon, model_head="cp"):
    return lambda: TJDGPT2(
        TJDConfig(
            base_dist=BaseDistConfig(
                vocab_size=768,
                rank=rank,
                horizon=horizon,
                param_net=TensorParamNetConfig(),
            ),
            model_head=model_head,
        )
    )


def create_model_llama_fn(
    rank,
    horizon,
    model_head="cp",
    model_kwargs={"hf_model_name": "meta-llama/Llama-2-7b-chat-hf"},
):
    return lambda: TJDLLAMA(
        TJDConfig(
            base_dist=BaseDistConfig(
                vocab_size=32000,
                horizon=horizon,
                rank=rank,
                param_net=TensorParamNetConfig(
                    num_layers=2,
                    hidden_dim=768,
                ),
            ),
            model_head=model_head,
            model_kwargs=model_kwargs,
        ),
    )


def main(args):
    model_fn = {
        "gpt2": create_model_gpt_fn,
        "llama": create_model_llama_fn,
    }[args.model_family]
    exps = [
        {
            "name": f"{args.model_family}::r{r}::h{h}",
            "model_fn": model_fn(r, h),  # Pass current r, h values
        }
        for (r, h) in itertools.product([1, 16, 32], [2])
        # for (r, h) in zip([2], [2])
    ] + [
        {
            "name": f"{args.model_family}::base",
            "model_fn": model_fn(1, 1, model_head="base"),
        }
    ]

    results = {}
    pbar = tqdm(exps, desc="Computing parameters")
    for exp in pbar:
        # Create model
        model = exp["model_fn"]()
        results[exp["name"]], params_detailed = get_params(model)
        # Clean up to avoid memory accumulation between experiments
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        pbar.set_postfix({"exp": exp["name"]})

    # Plot
    plot_params_dist(results, path=f"results/plots/{args.model_family}_params_dist.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_family",
        type=str,
        choices=["gpt2", "llama"],
        default="gpt2",
    )
    args = parser.parse_args()
    main(args)
