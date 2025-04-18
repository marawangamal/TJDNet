"""Plots reconstruction error of language model output distributions using tensor completion.

Example:
    python plot_output_dist_recons_error.py --model meta-llama/Llama-2-7b-chat-hf

"""

import os
from argparse import Namespace
import argparse
from typing import Tuple

import torch
from datasets import load_dataset
import tensorly as tl
import tqdm


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_errors(subsets, output_path: str = "tensor_completion_errors.png"):
    """Plots reconstruction error as a function of tensor rank

    Args:
        subsets (list): List of dictionaries containing subset names, errors, and ranks.

    """

    # mkdir if not exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(10, 6))

    # Set a nice color palette
    colors = sns.color_palette("viridis", len(subsets))

    # Plot each subset's errors
    for i, subset in enumerate(subsets):
        name = subset["name"]
        errors = subset["errors"]
        ranks = subset["ranks"]

        # Convert to numpy for easier manipulation
        if not isinstance(errors, np.ndarray):
            errors = np.array(errors)

        # Normalize errors to make comparison easier
        normalized_errors = errors / np.max(errors)

        # Plot the error curve
        plt.plot(
            ranks,
            normalized_errors,
            "o-",
            label=name,
            color=colors[i],
            linewidth=2,
            markersize=8,
        )

    # Add gridlines
    plt.grid(True, linestyle="--", alpha=0.7)

    # Set labels and title
    plt.xlabel("Tensor Rank", fontsize=14)
    plt.ylabel("Normalized Reconstruction Error", fontsize=14)
    plt.title("Tensor Completion Reconstruction Error vs Rank", fontsize=16)

    # Add legend
    plt.legend(fontsize=12)

    # Tight layout to avoid clipping
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_path}")


def train_tc(
    y_train: torch.Tensor,
    x_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    tensor_model: str = "mps",
) -> Tuple[list, list]:
    """Train a tensor completion model on the dataset and compute reconstruction error.

    Args:
        y: torch.Tensor: Input tensor (e.g., model output). Shape: (B, D).
        x: torch.Tensor: Target tensor (e.g., ground truth). Shape: (B,)

    Returns:
        - errors (list): Errors achieved at different tensor ranks.
        - ranks (list): Rank values.
    """

    errors = []
    ranks = []

    for rank in [1, 2, 4, 8, 16]:
        # cp_regressor = tl.regression.CPRegressor(weight_rank=4)  # did not work
        cp_regressor = tl.regression.CP_PLSR(n_components=rank)
        cp_regressor.fit(x_train.double().numpy(), y_train.numpy())

        # Compute the reconstruction error
        preds = cp_regressor.predict(x_test.double().numpy())
        errors.append(
            torch.linalg.norm(torch.from_numpy(preds).squeeze(-1) - y_test).item()
        )
        ranks.append(rank)
    return errors, ranks


def main(args: Namespace):

    subsets = [
        {"name": "gpt2_gsm8k", "errors": [], "ranks": []},
        {"name": "gpt2_poem", "errors": [], "ranks": []},
        {"name": "gpt2_newline", "errors": [], "ranks": []},
        {"name": "gpt2_space", "errors": [], "ranks": []},
    ]

    for subset in tqdm.tqdm(subsets, desc="Processing subsets"):
        dataset = load_dataset("mremila/tjdnet", name=subset["name"])
        dataset = dataset.select_columns(["x", "y", "py|x"])
        errors, ranks = train_tc(
            x_train=torch.tensor(dataset["train"]["y"]),
            y_train=torch.tensor(dataset["train"]["py|x"]),
            x_test=torch.tensor(dataset["test"]["y"]),
            y_test=torch.tensor(dataset["test"]["py|x"]),
            tensor_model="mps",
        )

        # Store errors and ranks
        subset["errors"] = errors
        subset["ranks"] = ranks

    # Plot errors for all subsets
    plot_errors(subsets, output_path="results/plots/tensor_completion_errors.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze the output distribution of a language model.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-n",
        "--num_samples",
        type=int,
        default=1000,
    )
    args = parser.parse_args()
    main(args)
