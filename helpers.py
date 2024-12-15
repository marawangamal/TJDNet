import os
import argparse
import subprocess


import numpy as np
import random

import torch


# TODO: change horizon, horizon_eval to train_horizon, eval_horizon and eval_horizon should default to train_horizon if not specified
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on the ELI5 dataset.")
    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training and evaluation.",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for training."
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Number of warmup steps for learning rate scheduler.",
    )
    parser.add_argument(
        "--grad_clip_val",
        type=float,
        default=None,
        help="Gradient clipping value for training.",
    )
    # Model arguments
    parser.add_argument(
        "--model_head",
        type=str,
        default="mps",
        help="Type of factorization to use for the model.",
        choices=[
            "cp",
            "mps",
            "umps",
            "full",
            "base",
        ],
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        default="word",
        help="Type of tokenizer to use for processing text.",
        choices=["char", "word"],
    )
    parser.add_argument(
        "--scale_loss",
        default=False,
        action="store_true",
        help="Whether to scale the loss during training.",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=256,
        help="Block size for model input sequences.",
    )
    parser.add_argument(
        "--n_embd",
        type=int,
        default=384,
        help="Dimensionality of the model embeddings.",
    )
    parser.add_argument(
        "--n_layer",
        type=int,
        default=6,
        help="Number of hidden layers in the transformer model.",
    )
    parser.add_argument(
        "--n_head",
        type=int,
        default=6,
        help="Number of attention heads in the transformer model.",
    )
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate.")
    parser.add_argument(
        "--positivity_func",
        type=str,
        default="exp",
        choices=["sq", "abs", "exp"],
        help="Positivity function to use for MPSDist.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=2,
        help="Rank of the tensor train decomposition.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=2,
        help="Block size for model input sequences.",
    )
    parser.add_argument(
        "--freeze_base_model",
        default=False,
        action="store_true",
        help="Whether to freeze the base model during training.",
    )
    # Evaluation arguments
    parser.add_argument(
        "--horizon_eval",
        type=int,
        default=2,
        help="Block size for model input sequences.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=500,
        help="Maximum number of tokens to generate during evaluation.",
    )
    # Data Arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="shakespeare",
        help="Type of dataset to use for training.",
        choices=[
            "shakespeare",
            "wikitext",
        ],
    )
    # Misc arguments
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    return parser.parse_args()


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_git_info():
    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
        # Get just the first line of the commit message
        commit_message = (
            subprocess.check_output(["git", "log", "-1", "--pretty=%s"])
            .decode("ascii")
            .strip()
        )
        return {"commit_hash": commit_hash, "commit_message": commit_message}
    except:
        return {
            "commit_hash": "Git commit hash not available",
            "commit_message": "Git commit message not available",
        }
