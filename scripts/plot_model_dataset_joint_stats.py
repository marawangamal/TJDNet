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

    # sort by rank
    df = df.sort_values(by="Rank", ascending=True)

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
        # palette=color_map,
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


def plot_histograms(ckpt_path: str, save_path=None):
    """Plot histograms using seaborn FacetGrid for cleaner visualization."""

    ckpt = torch.load(ckpt_path, weights_only=False)

    # Prepare data for seaborn
    data = []
    zero_thresholds = {}  # Store zero thresholds for each category

    for category, stats in ckpt.items():
        ys = (
            torch.cat(stats["max_ys"]).numpy()
            if len(stats["max_ys"]) > 0
            else np.array([])
        )
        xs = (
            torch.cat(stats["max_xs"]).numpy()
            if len(stats["max_xs"]) > 0
            else np.array([])
        )
        zs = np.concatenate([ys, xs])

        if len(zs) > 0:
            ranks = [
                matrix_rank_from_spectrum(stats["spectra"][i])
                for i in range(len(stats["spectra"]))
            ]
            mu_rank = torch.mean(torch.tensor(ranks, dtype=torch.float32)).item()
            for value in zs:
                data.append({"Category": category, "Value": value, "Rank": mu_rank})

    if not data:
        print("No data to plot")
        return

    df = pd.DataFrame(data)
    df = df.sort_values(by="Rank", ascending=True)

    # Create seaborn FacetGrid with percentage y-axis
    g = sns.FacetGrid(df, col="Category", col_wrap=3, height=4, aspect=1.2)
    g.map_dataframe(sns.histplot, x="Value", bins=50, log_scale=True, stat="percent")
    g.set_titles(col_template="{col_name}")
    g.set_xlabels("Value")
    g.set_ylabels("Percentage")

    # Set consistent x-axis limits
    g.set(xlim=(1e-16, 1e-1))
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved histogram to {save_path}")


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
