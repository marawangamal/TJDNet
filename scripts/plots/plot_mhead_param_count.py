"""Plot memory usage layer-wise distribution.

Usage:
    python scripts/plots/plot_mem_dist.py --model_family [gpt2/llama]

Example:
     python scripts/plots/plot_mem_dist.py --model_family gpt2

"""

import argparse
import gc
import torch
from tqdm import tqdm
import itertools
import numpy as np

import matplotlib.pyplot as plt

from utils.models import create_model_gpt_fn, create_model_llama_fn


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


def main(args):
    model_fn = {
        "gpt2": lambda r, h, m: create_model_gpt_fn(
            r, h, m, param_net_config={"hidden_dim": 768, "use_decoder": True}
        ),
        "llama": lambda r, h, m: create_model_llama_fn(
            r, h, m, param_net_config={"hidden_dim": 32000, "use_decoder": True}
        ),
    }[args.model_family]
    exps = (
        [
            {
                "name": f"{args.model_family}::mhcp::r{r}::h{h}",
                "model_fn": model_fn(r, h, "cp"),  # Pass current r, h values
            }
            for (r, h) in itertools.product([2], [2, 4, 6])
        ]
        + [
            {
                "name": f"{args.model_family}::mhucp::r{r}::h{h}",
                "model_fn": model_fn(r, h, "ucp"),  # Pass current r, h values
            }
            for (r, h) in itertools.product([2], [2, 4, 6])
        ]
        + [
            {
                "name": f"{args.model_family}::mhbase",
                "model_fn": model_fn(1, 1, "base"),
            }
        ]
    )

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
