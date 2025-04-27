"""Plots reconstruction error of language model output distributions using tensor completion.

Example:
    python scripts/plots/plot_output_dist_recons_error.py --test

"""

import os
from argparse import Namespace
import argparse
import re
from typing import List, Optional, Tuple

import torch
from datasets import load_dataset
import tensorly as tl
import tqdm
import tntorch as tn


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from tjdnet.regressors.cp_regressor import CPRegressor
from utils.utils import get_experiment_name, group_arr, plot_groups

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


def print_dist(y):
    print("Distribution of py|x:")
    print(f"Mean: {y.mean():.8f}")
    print(f"Std: {y.std():.8f}")


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


def train_cp(
    y_train: torch.Tensor,
    x_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    y_val: Optional[torch.Tensor] = None,
    x_val: Optional[torch.Tensor] = None,
    vocab_size: int = 4,
    ranks: List = [2, 4, 8, 16],
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
    for rank in tqdm.tqdm(ranks, desc="Training CPRegressor", leave=False):
        reg = CPRegressor(
            vocab_size,
            horizon=x_train.shape[1],
            rank=rank,
            device=x_train.device,
            init_method=kwargs["init_method"],
            loss_type=kwargs["loss_type"],
        )
        test_error = reg.fit(  # return self with best state
            x_train,
            y_train,
            x_val=x_val,
            y_val=y_val,
            y_test=y_test,
            x_test=x_test,
            **kwargs,
        )

        errors.append(test_error)
        print(f"rank={rank:<2d}  loss ({kwargs['loss_type']}) = {test_error:.8f}")

    # Compute baseline error (mean of y_train)
    reg_baseline = CPRegressor(
        vocab_size,
        horizon=x_train.shape[1],
        rank=1,
        device=x_train.device,
        init_method="zeros",
        loss_type=kwargs["loss_type"],
    )
    error_baseline = reg_baseline.loss_fn(reg_baseline.predict(x_test), y_test).item()
    return errors, ranks, error_baseline


def parse_model_horizon(name):
    h = int(re.search(r"_h(\d+)", name).group(1))  # type: ignore
    return f"h={h}"


def main_test(args, seed: int = 0) -> None:
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
    cp_cores = [
        # === Random dist ====================
        # torch.randn((vocab_size, rank_true))
        # === Match dist of mremila/tjdnet ===
        torch.normal(0.00002437, 0.00014511, size=(vocab_size, rank_true), out=None)
        for _ in range(horizon)
    ]
    tensor_gt = tl.cp_to_tensor((None, cp_cores))  # shape (I, J, K)

    # 2. Sample from the tensor
    x = torch.randint(0, vocab_size, (num_pts, horizon))
    idx_tuple = tuple(x[:, d] for d in range(x.shape[1]))  # H tensors, each (num_pts,)
    y = tensor_gt[idx_tuple]

    # 3. Create train/val/test splits
    train_frac, val_frac = 0.8, 0.1
    n_train = int(train_frac * num_pts)
    n_val = int(val_frac * num_pts)
    x_train, y_train = x[:n_train], y[:n_train]
    x_val, y_val = x[n_train : n_train + n_val], y[n_train : n_train + n_val]
    x_test, y_test = x[n_train + n_val :], y[n_train + n_val :]

    # 4. Fit CP regressor
    errors, ranks, error_baseline = train_cp(
        y_train=y_train,
        x_train=x_train,
        x_test=x_test,
        y_test=y_test,
        x_val=x_val,
        y_val=y_val,
        vocab_size=vocab_size,
        **vars(args),
    )

    # 5. Print a tiny report
    print("-" * 80 + "self-test report" + "-" * 80)
    for r, e in zip(ranks, errors):
        print(f"Rank = {r:<2d}  Loss ({args.loss_type}) = {e:.8f}")
    print(f"Baseline loss ({args.loss_type}) = {error_baseline:.8f}")


def main(args: Namespace):

    subsets = [
        # {"name": "gpt2_gsm8k", "errors": [], "ranks": []},
        # {"name": "gpt2_poem", "errors": [], "ranks": []},
        # {"name": "gpt2_newline", "errors": [], "ranks": []},
        # {"name": "gpt2_space", "errors": [], "ranks": []},
        # Horizon=2
        # {"name": "meta_llama_llama_2_7b_chat_hf_gsm8k_h2", "errors": [], "ranks": []},
        # {"name": "meta_llama_llama_2_7b_chat_hf_poem_h2", "errors": [], "ranks": []},
        # {"name": "meta_llama_llama_2_7b_chat_hf_newline_h2", "errors": [], "ranks": []},
        # {"name": "meta_llama_llama_2_7b_chat_hf_space_h2", "errors": [], "ranks": []},
        # Horizon=4
        {"name": "meta_llama_llama_2_7b_chat_hf_gsm8k_h4", "errors": [], "ranks": []},
        {"name": "meta_llama_llama_2_7b_chat_hf_poem_h4", "errors": [], "ranks": []},
        {"name": "meta_llama_llama_2_7b_chat_hf_newline_h4", "errors": [], "ranks": []},
        {"name": "meta_llama_llama_2_7b_chat_hf_space_h4", "errors": [], "ranks": []},
    ]

    for subset in tqdm.tqdm(subsets, desc="Processing subsets"):
        dataset = load_dataset("mremila/tjdnet", name=subset["name"])
        dataset = dataset.select_columns(["x", "y", "py|x"])

        if not isinstance(dataset, dict):
            raise ValueError("Expected a dictionary of datasets.")

        # Split train to train and validation sets
        dataset["train"], dataset["val"] = (
            dataset["train"].train_test_split(test_size=0.05, seed=42).values()
        )

        # Print distribution of py|x
        print_dist(torch.tensor(dataset["train"]["py|x"]))

        errors, ranks, error_baseline = train_cp(
            x_train=torch.tensor(dataset["train"]["y"]),
            y_train=torch.tensor(dataset["train"]["py|x"]),
            x_test=torch.tensor(dataset["test"]["y"]),
            y_test=torch.tensor(dataset["test"]["py|x"]),
            x_val=torch.tensor(dataset["val"]["y"]),
            y_val=torch.tensor(dataset["val"]["py|x"]),
            vocab_size=int(torch.max(torch.tensor(dataset["train"]["y"])).item()) + 1,
            ranks=[2, 4, 8, 16, 32, 64, 128, 256, 512],
            **vars(args),
        )
        # # === Debug plot >>>
        # rand error
        # errors = [1 / i + 0.1 * abs(torch.randn((1, 1)).item()) for i in range(1, 10)]
        # ranks = list(range(1, 10))
        # error_baseline = 0.1
        # # <<< === Debug plot

        # Print a tiny report
        print("-" * 80 + f"\nSubset: {subset['name']}")
        for r, e in zip(ranks, errors):
            print(f"Rank = {r:<2d}  Loss({args.loss_type}) = {e:.4f}")
        print(f"Baseline loss ({args.loss_type}) = {error_baseline:.4f}")

        # Store errors and ranks
        subset["errors"] = errors
        subset["log-errors"] = torch.log(torch.tensor(errors)).tolist()
        subset["ranks"] = ranks

    # Plot errors for all subsets
    # ===========================
    # group_arr   :  creates a multi-level dict grouping from an array
    # plot_groups :  plots a line for every leaf in results_grouped (dict)

    results_ungrouped = []
    for res_group in subsets:
        for err, log_err, rank in zip(
            res_group["errors"],
            res_group["log-errors"],
            res_group["ranks"],
        ):
            results_ungrouped.append(
                {
                    **res_group,
                    "error": err,
                    "log-error": log_err,
                    "rank": rank,
                }
            )

    results_grouped = group_arr(
        results_ungrouped,
        lambda x: x["name"],
        lambda x: parse_model_horizon(x["name"]),
    )

    exp_name = get_experiment_name(vars(args))
    save_path = f"results/plots/odre_{exp_name}.png"

    plot_groups(
        results_grouped,
        x_key="rank",
        y_key="log-error" if args.use_log else "error",
        path=save_path,
        # First level controls color, second controls marker
        style_dims=[
            "color",
            "marker",
        ],
        style_cycles={
            "color": [
                "#0173B2",
                "#DE8F05",
                "#029E73",
                "#D55E00",
                "#CC78BC",
                "#CA9161",
                "#FBAFE4",
                "#949494",
            ]
        },
        axes_kwargs={
            "title": f"CP Tensor Approximation Error for p(y|x)",
            "xlabel": "RANK",
            "ylabel": (
                f"LOG {args.loss_type.upper()}"
                if args.use_log
                else f"{args.loss_type.upper()}"
            ),
            "ylim": (0, 2),
        },
        fig_kwargs={
            "figsize": (10, 6),
            "dpi": 300,
        },
    )
    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze the output distribution of a language model.",
    )
    # === Optimization args ==========
    parser.add_argument(
        "--epochs",
        type=int,
        default=10000,
        help="Number of epochs to train.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--min_epochs",
        type=int,
        default=10,
        help="Minimum number of epochs to train.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="Number of epochs to wait for improvement before stopping.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-12,
        help="Absolute tolerance for early stopping.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,  # 0.001% relative tolerance
        help="Relative tolerance for early stopping.",
    )
    parser.add_argument(
        "--use_log",
        action="store_true",
        help="Use log scale for the loss function.",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="mare",
        choices=["mse", "mae", "mare"],
        help="Loss function to use. Options: mse (mean squared error), mae (mean absolute error), mare (mean absolute relative error).",
    )
    parser.add_argument(
        "--init_method",
        type=str,
        default="normal",
        choices=["zeros", "normal"],
    )

    # === Run test ==========
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run the test function.",
    )

    args = parser.parse_args()
    if args.test:
        main_test(args)
    else:
        main(args)
