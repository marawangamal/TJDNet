import argparse
import math
import os
from typing import Dict, List, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from tqdm import tqdm


def ckpt_friendly_name(ckpt_name: str):
    # results/spectrum_progress_meta-llama_Llama-2-7b-chat-hf_topk5000_rand.pt' => 'meta-llama_Llama-2-7b-chat-hf (rand)'
    return (
        ckpt_name.replace("results/spectrum_progress_", "")
        .split("_topk")[0]
        .replace(".pt", "")
        .replace("_", "/")
        + f"{'_rand' if 'rand' in ckpt_name else 'pretrained'}"
    )


def matrix_rank_from_spectrum(
    spectrum,
    matrix_shape: Optional[Tuple[int, int]] = None,
    eps=torch.finfo(torch.float64).eps,
):
    """Compute matrix rank from spectrum.

    Args:
        spectrum (torch.Tensor): Spectrum of the matrix.
        matrix_shape (tuple): Shape of the matrix.
        eps (float): Tolerance.
    """
    m_shape = (len(spectrum), len(spectrum)) if matrix_shape is None else matrix_shape
    thresh = spectrum.max() * max(m_shape) * eps
    rank = (spectrum > thresh).sum().item()
    return rank


def plot_dataset_rank(ckpt_paths: List[str], save_path=None):
    ckpts = {ckpt: torch.load(ckpt, weights_only=False) for ckpt in ckpt_paths}
    data = []
    for ckpt_name in ckpts:
        for category, stats in ckpts[ckpt_name].items():
            ranks = [
                matrix_rank_from_spectrum(stats["spectra"][i])
                for i in range(len(stats["spectra"]))
            ]
            mu_rank = torch.mean(torch.tensor(ranks, dtype=torch.float32)).item()
            data.append(
                {
                    "Model": ckpt_friendly_name(ckpt_name),
                    "Category": category,
                    "Rank": mu_rank,
                }
            )

    df = pd.DataFrame(data)

    # Set seaborn style
    sns.set_theme(style="whitegrid")

    # Get unique models and categories for ordering
    models = df["Model"].unique()
    categories = df["Category"].unique()

    # Create color mapping
    colors = plt.cm.get_cmap("Set3")(np.linspace(0, 1, len(models)))
    color_map = {model: colors[i] for i, model in enumerate(models)}

    plt.figure(figsize=(12, 8))
    ax = sns.barplot(
        data=df,
        x="Category",
        y="Rank",
        hue="Model",
        palette=color_map,
        edgecolor="black",
    )

    plt.ylabel("Matrix Rank (Mean)")
    plt.xlabel("Dataset")
    plt.title("Matrix Rank Comparison Across Datasets and Models")
    plt.legend(title="Model", loc="upper left")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")


def get_model_name(ckpt_path: str):
    return "".join(ckpt_path.split("spectrum_progress_")[-1].split("_")[:-1])


def plot_histograms(
    # max_ys: Dict[str, List[torch.Tensor]],
    # max_xs: Dict[str, List[torch.Tensor]],
    # save_path=None,
    ckpt_path: str,
    save_path=None,
):
    """Plot histograms of max_ys and max_xs in a single figure with subplots for each category."""

    ckpt = torch.load(ckpt_path, weights_only=False)
    max_ys = {k: v["max_ys"] for k, v in ckpt.items()}
    max_xs = {k: v["max_xs"] for k, v in ckpt.items()}

    categories = list(max_ys.keys())
    n = len(categories)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False
    )
    for idx, category in tqdm(
        enumerate(categories), leave=False, desc="Creating histograms"
    ):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        ys = (
            torch.cat(max_ys[category]).numpy()
            if len(max_ys[category]) > 0
            else np.array([])
        )
        xs = (
            torch.cat(max_xs[category]).numpy()
            if len(max_xs[category]) > 0
            else np.array([])
        )
        zs = np.concatenate([ys, xs])
        # print(f"Max zs: {zs.max()} | Min zs: {zs.min()}")
        if len(zs) > 0:
            sns.histplot(
                data=zs,
                bins=10,
                alpha=0.5,
                label="Max Xs",
                log_scale=True,
                ax=ax,
                # binrange=(10e-16, 10e-1),  # set range here
            )
        ax.set_title(category)
        ax.set_xlim(10e-16, 10e-1)
        ax.legend()
    # Hide any unused subplots
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        fig.delaxes(axes[row][col])
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpts", type=str, nargs="+", default=None)
    args = parser.parse_args()

    save_path_rank = os.path.join(
        "results",
        "plots",
        f"model_dataset_ranks.png",
    )

    plot_dataset_rank(args.ckpts, save_path=save_path_rank)
    for ckpt_path in args.ckpts:
        save_path_hist = os.path.join(
            "results",
            "plots",
            f"model_dataset_histogram_{get_model_name(ckpt_path)}.png",
        )
        plot_histograms(ckpt_path, save_path=save_path_hist)

    print("Done")


if __name__ == "__main__":
    main()

# --ckpts results/spectrum_progress_meta-llama_Llama-2-7b-chat-hf_topk5000.pt results/spectrum_progress_deepseek-ai_DeepSeek-R1-0528-Qwen3-8B_topk5000.pt
