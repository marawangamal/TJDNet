"""Plots reconstruction error of language model output distributions using tensor completion.

Example:
    python plot_output_dist_recons_error.py --model meta-llama/Llama-2-7b-chat-hf

"""

import os
from argparse import Namespace
import argparse
from typing import List, Literal, Optional, Tuple

import torch
from datasets import load_dataset
import tensorly as tl
import tqdm
import tntorch as tn


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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


def train_tnt(
    y_train: torch.Tensor,
    x_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    y_val: Optional[torch.Tensor] = None,
    x_val: Optional[torch.Tensor] = None,
    vocab_size: int = 4,
):
    """Train a tensor completion model on the dataset and compute reconstruction error.

    Args:
        x: torch.Tensor: Input tensor (e.g., model output). Shape: (B, H)
        y: torch.Tensor: Target tensor (e.g., ground truth). Shape: (B,)

    Returns:
        - errors (list): Errors achieved at different tensor ranks.
        - ranks (list): Rank values.
    """
    errors, ranks = [], []
    _, horizon = x_train.shape

    def loss(t_hat):
        return tn.relative_error(y_train, t_hat[x_train])

    for rank in tqdm.tqdm([1, 2, 4, 8, 16], desc="Training CPRegressor", leave=False):
        t_hat = tn.rand((vocab_size,) * horizon, ranks_tt=rank, requires_grad=True)
        tn.optimize(t_hat, loss)

        error = tn.relative_error(y_test, t_hat[x_test])
        errors.append(error.item())
        ranks.append(rank)

    return errors, ranks


def train_tc(
    y_train: torch.Tensor,
    x_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    y_val: Optional[torch.Tensor] = None,
    x_val: Optional[torch.Tensor] = None,
    vocab_size: int = 4,
    ranks: List = [1, 2, 4, 8, 16],
    # Optimization args
    **kwargs,
) -> Tuple[list, list, float]:
    """Train a tensor completion model on the dataset and compute reconstruction error.

    Args:
        x: torch.Tensor: Input tensor (e.g., model output). Shape: (B, H)
        y: torch.Tensor: Target tensor (e.g., ground truth). Shape: (B,)

    Returns:
        - errors (list): Errors achieved at different tensor ranks.
        - ranks (list): Rank values.
    """
    # coords tensors are already (N, H)
    errors = []
    for rank in tqdm.tqdm([1, 2, 4, 8, 16], desc="Training CPRegressor", leave=False):
        reg = CPRegressor(
            vocab_size,
            horizon=x_train.shape[1],
            rank=rank,
            device=x_train.device,
        )
        reg = reg.fit(  # return self with best state
            x_train,
            y_train,
            x_val=x_val,
            y_val=y_val,
            **kwargs,
        )

        preds = reg.predict(x_test)
        error = torch.linalg.norm(preds - y_test).item()
        errors.append(error)
        print(f"rank={rank:<2d}  loss ({kwargs['loss_type']}) = {error:.8f}")

    # Compute baseline error (mean of y_train)
    reg_baseline = CPRegressor(
        vocab_size,
        horizon=x_train.shape[1],
        rank=1,
        device=x_train.device,
        init_method="zeros",
    )
    preds_baseline = reg_baseline.predict(x_test)
    error_baseline = torch.linalg.norm(preds_baseline - y_test).item()

    return errors, ranks, error_baseline


def test(args, seed: int = 0) -> None:
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
    # mean, std = 0.00002437, 0.00014511  # statistics of mremila/tjdnet
    mean, std = 0, 0.1
    cp_cores = [
        # (torch.randn() * std) + mean for _ in range(horizon)
        torch.normal(mean, std, size=(vocab_size, rank_true), out=None)
        for _ in range(horizon)
    ]

    # Sanity check
    # full tensor T(i,j,k) = sum_r A[i,r] * B[j,r] * C[k,r]
    tensor_gt = tl.cp_to_tensor((None, cp_cores))  # shape (I, J, K)

    # 2. Sample from the tensor
    # x = torch.randint(0, vocab_size, (num_pts, horizon))
    x = torch.randint(0, vocab_size, (num_pts, horizon))
    idx_tuple = tuple(x[:, d] for d in range(x.shape[1]))  # H tensors, each (num_pts,)
    y = tensor_gt[idx_tuple]

    # 80% train, 10% val, 10% test
    train_frac, val_frac = 0.8, 0.1
    n_train = int(train_frac * num_pts)
    n_val = int(val_frac * num_pts)

    # slices
    x_train, y_train = x[:n_train], y[:n_train]
    x_val, y_val = x[n_train : n_train + n_val], y[n_train : n_train + n_val]
    x_test, y_test = x[n_train + n_val :], y[n_train + n_val :]

    # 3. run the training routine for several candidate ranks
    errors, ranks, error_baseline = train_tc(
        y_train=y_train,
        x_train=x_train,
        x_test=x_test,
        y_test=y_test,
        x_val=x_test,
        y_val=y_test,
        vocab_size=vocab_size,
        **vars(args),
    )

    # 4. print a tiny report
    print("-" * 80 + "self-test report" + "-" * 80)
    for r, e in zip(ranks, errors):
        print(f"Rank = {r:<2d}  Loss ({args.loss_type}) = {e:.8f}")
    print(f"Baseline loss ({args.loss_type}) = {error_baseline:.8f}")

    # 5. sanity assertion — best error should occur at or below R_true
    best_rank = ranks[int(torch.argmin(torch.tensor(errors)))]
    assert (
        best_rank >= rank_true
    ), f"Expected lowest error at rank >= {rank_true}, got {best_rank}. Check CP fitting pipeline."
    print(
        f"[✓] self-test passed — pipeline recovers the true rank (true_rank={rank_true})"
    )


def print_dist(y):
    """Prints the distribution of the tensor completion model output.

    Args:
        y: torch.Tensor: Model output tensor. Shape: (B, H)

    """
    print("Distribution of py|x:")
    print(f"Mean: {y.mean():.8f}")
    print(f"Std: {y.std():.8f}")


def main(args: Namespace):

    subsets = [
        # {"name": "gpt2_gsm8k", "errors": [], "ranks": []},
        # {"name": "gpt2_poem", "errors": [], "ranks": []},
        # {"name": "gpt2_newline", "errors": [], "ranks": []},
        # {"name": "gpt2_space", "errors": [], "ranks": []},
        # {"name": "meta_llama_llama_2_7b_chat_hf_gsm8k", "errors": [], "ranks": []},
        # {"name": "meta_llama_llama_2_7b_chat_hf_poem", "errors": [], "ranks": []},
        # {"name": "meta_llama_llama_2_7b_chat_hf_newline", "errors": [], "ranks": []},
        {"name": "meta_llama_llama_2_7b_chat_hf_space", "errors": [], "ranks": []},
    ]

    for subset in tqdm.tqdm(subsets, desc="Processing subsets"):
        dataset = load_dataset("mremila/tjdnet", name=subset["name"])
        dataset = dataset.select_columns(["x", "y", "py|x"])

        # Split train to train and validation sets
        dataset["train"], dataset["val"] = (
            dataset["train"].train_test_split(test_size=0.05, seed=42).values()
        )

        # Print distribution of py|x
        print_dist(torch.tensor(dataset["train"]["py|x"]))

        errors, ranks, error_baseline = train_tc(
            x_train=torch.tensor(dataset["train"]["y"]),
            y_train=torch.tensor(dataset["train"]["py|x"]),
            x_test=torch.tensor(dataset["test"]["y"]),
            y_test=torch.tensor(dataset["test"]["py|x"]),
            x_val=torch.tensor(dataset["val"]["y"]),
            y_val=torch.tensor(dataset["val"]["py|x"]),
            vocab_size=torch.max(torch.tensor(dataset["train"]["y"])) + 1,
            # Pass args from command line (make a dict then unpack)
            **vars(args),
        )

        # Print a tiny report
        print("-" * 80 + f"\nSubset: {subset['name']}")
        for r, e in zip(ranks, errors):
            print(f"Rank = {r:<2d}  Loss({args.loss_type}) = {e:.4f}")
        print(f"Baseline loss ({args.loss_type}) = {error_baseline:.4f}")

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
        "--test",
        action="store_true",
        help="Run the test function.",
    )
    # === Optimization args ==========
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10000,
        help="Number of epochs to train.",
    )
    parser.add_argument(
        "--min_epochs",
        type=int,
        default=10,
        help="Minimum number of epochs to train.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Number of epochs to wait for improvement before stopping.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-9,
        help="Absolute tolerance for early stopping.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,  # 0.001% relative tolerance
        help="Relative tolerance for early stopping.",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="mse",
        choices=["msre", "mare", "mse"],
        help="Loss function to use.",
    )
    args = parser.parse_args()
    if args.test:
        test(args)
    else:
        main(args)
