"""Plots reconstruction error of language model output distributions using tensor completion.

Example:
    python plot_output_dist_recons_error.py --model meta-llama/Llama-2-7b-chat-hf

"""

import os
from argparse import Namespace
import argparse
from typing import Optional, Tuple

import torch
from datasets import load_dataset
import tensorly as tl
import tqdm


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.sparse import coo_matrix

from tjdnet.models.cp_regressor import CPRegressor

# set tl backend to pytorch
tl.set_backend("pytorch")


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
    y_val: Optional[torch.Tensor] = None,
    x_val: Optional[torch.Tensor] = None,
    vocab_size: int = 4,
) -> Tuple[list, list]:
    """Train a tensor completion model on the dataset and compute reconstruction error.

    Args:
        x: torch.Tensor: Input tensor (e.g., model output). Shape: (B, H)
        y: torch.Tensor: Target tensor (e.g., ground truth). Shape: (B,)

    Returns:
        - errors (list): Errors achieved at different tensor ranks.
        - ranks (list): Rank values.
    """
    # coords tensors are already (N, H)
    errors, ranks = [], []
    for rank in tqdm.tqdm([1, 2, 4, 8, 16], desc="Training CPRegressor", leave=False):
        reg = CPRegressor(
            vocab_size,
            horizon=x_train.shape[1],
            rank=rank,
            device=x_train.device,
        )
        reg.fit(
            x_train,
            y_train,
            x_val=x_val,
            y_val=y_val,
            lr=1e-3,
            epochs=1000,
            batch_size=512,
            rtol=1e-3,  # ≥ 0.1% relative improvement
            atol=1e-5,  # ≥ absolute improvement
            patience=10,
        )
        preds = reg.predict(x_test)
        error = torch.linalg.norm(preds - y_test).item()
        errors.append(error)
        ranks.append(rank)
    return errors, ranks


def test(seed: int = 0) -> None:
    """Quick self-test for :pyfunc:`train_tc`.

    Creates a synthetic CP tensor of known rank, samples (x, y) pairs,
    and checks that CP regression recovers it with low error when the
    correct rank is allowed.

    Args:
        seed: Random seed for repeatability.

    Raises:
        AssertionError if the best error is not achieved at or below the
        ground-truth rank (indicating something is wrong with the pipeline).
    """
    torch.manual_seed(seed)
    horizon, rank_true, vocab_size = 3, 4, 4
    num_pts = 2000

    # 1. Generate a synthetic CP tensor of known rank
    cp_cores = [torch.randn(vocab_size, rank_true) for _ in range(horizon)]
    # full tensor T(i,j,k) = sum_r A[i,r] * B[j,r] * C[k,r]
    tensor_gt = tl.cp_to_tensor((None, cp_cores))  # shape (I, J, K)

    # 2. Sample from the tensor
    # x = torch.randint(0, vocab_size, (num_pts, horizon))
    x = torch.randint(0, vocab_size, (num_pts, horizon))
    idx_tuple = tuple(x[:, d] for d in range(x.shape[1]))  # H tensors, each (num_pts,)
    y = tensor_gt[idx_tuple]

    # 80/20 split
    n_train = int(0.8 * num_pts)
    x_train, y_train = x[:n_train], y[:n_train]
    x_test, y_test = x[n_train:], y[n_train:]

    # 3. run the training routine for several candidate ranks
    errors, ranks = train_tc(
        y_train=y_train,
        x_train=x_train,
        x_test=x_test,
        y_test=y_test,
    )

    # 4. print a tiny report
    for r, e in zip(ranks, errors):
        print(f"rank={r:<2d}  RMSE={e:.4f}")

    # 5. sanity assertion — best error should occur at or below R_true
    best_rank = ranks[int(torch.argmin(torch.tensor(errors)))]
    assert (
        best_rank >= rank_true
    ), f"Expected lowest error at rank >= {rank_true}, got {best_rank}. Check CP fitting pipeline."
    print(
        f"[✓] self-test passed — pipeline recovers the true rank (true_rank={rank_true})"
    )


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

        # Split train to train and validation sets
        dataset["train"], dataset["val"] = (
            dataset["train"].train_test_split(test_size=0.2, seed=42).values()
        )

        errors, ranks = train_tc(
            x_train=torch.tensor(dataset["train"]["y"]),
            y_train=torch.tensor(dataset["train"]["py|x"]),
            x_test=torch.tensor(dataset["test"]["y"]),
            y_test=torch.tensor(dataset["test"]["py|x"]),
            x_val=torch.tensor(dataset["val"]["y"]),
            y_val=torch.tensor(dataset["val"]["py|x"]),
            vocab_size=torch.max(torch.tensor(dataset["train"]["y"])) + 1,
        )

        # Print a tiny report
        print("-" * 80 + f"\nSubset: {subset['name']}")
        for r, e in zip(ranks, errors):
            print(f"rank={r:<2d}  RMSE={e:.4f}")

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
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run the test function.",
    )
    args = parser.parse_args()
    if args.test:
        test()
    else:
        main(args)
