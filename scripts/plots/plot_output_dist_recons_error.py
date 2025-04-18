"""Plots reconstruction error of language model output distributions using tensor completion.

Example:
    python plot_output_dist_recons_error.py --model meta-llama/Llama-2-7b-chat-hf

"""

from argparse import Namespace
import argparse
from typing import Tuple

import torch
from datasets import load_dataset


def plot_errors(errors: list, ranks: list) -> None:
    """Plots reconstruction error as a function of tensor rank

    Args:
        errors (list): Errors achieved at different tensor ranks.
        ranks (list): Rank values.

    """
    raise NotImplementedError(
        "This function should be implemented to plot the error curves for the spectrum."
    )


def train_tc(
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    tensor_model: str = "mps",
) -> Tuple[list, list]:
    """Train a tensor completion model on the dataset and compute reconstruction error.

    Args:
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        test_loader (torch.utils.data.DataLoader): DataLoader for testing data.
        tensor_model (str): Tensor model to use for completion. Default is "mps".

    Returns:
        - errors (list): Errors achieved at different tensor ranks.
        - ranks (list): Rank values.
    """

    errors = []
    ranks = []
    return errors, ranks


def main(args: Namespace):

    subsets = [
        {"name": "gpt2_gsm8k", "errors": [], "ranks": []},
        {"name": "gpt2_poem", "errors": [], "ranks": []},
        {"name": "gpt2_newline", "errors": [], "ranks": []},
        {"name": "gpt2_space", "errors": [], "ranks": []},
    ]

    for subset in subsets:

        dataset = load_dataset("mremila/tjdnet", name=subset["name"])
        dataset = dataset.select_columns(["x", "y", "py|x"])
        # Make dataloader
        train_dataloader = torch.utils.data.DataLoader(
            dataset["train"], batch_size=args.batch_size, shuffle=False
        )
        test_dataloader = torch.utils.data.DataLoader(
            dataset["test"], batch_size=args.batch_size, shuffle=False
        )
        errors, ranks = train_tc(
            train_dataloader,
            test_dataloader,
            tensor_model="mps",
        )

        # Store errors and ranks
        subsets[subset["name"]]["errors"] = errors
        subset["name"]["ranks"] = ranks

    # Plot errors for all subsets
    plot_errors(errors, ranks)


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
