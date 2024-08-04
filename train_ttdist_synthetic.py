from typing import Tuple
import argparse
import random

import torch
import numpy as np
import wandb

from TJDNet.TTDist import TTDist
from TJDNet.TTDist.init import get_random_mps
from TJDNet.utils import get_entropy_loss, get_preference_loss, check_naninf
from utils.utils import get_experiment_name


# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to investigate using different non-linearities for the TT Dist."
    )
    parser.add_argument(
        "--true_dist",
        type=str,
        default="rand",
        help="True distribution",
        choices=[
            "rand",  # Uniform random initialization [0, 1]
            "randn",  # Normal random initialization
        ],
    )

    parser.add_argument(
        "--init_dist",
        type=str,
        default="randn",
        choices=[
            "rand",
            "randn",
        ],
        help="Initialization method",
    )

    parser.add_argument(
        "--rank", type=int, default=2, help="Rank of the tensor decomposition"
    )
    parser.add_argument(
        "--true_rank", type=int, default=2, help="Rank of the tensor decomposition"
    )

    parser.add_argument(
        "--output_size", type=int, default=3, help="Output size of the tensor"
    )
    parser.add_argument("--vocab_size", type=int, default=8, help="Vocabulary size")
    parser.add_argument(
        "--norm_method", type=str, default="abs", help="Normalization method"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--n_iters", type=int, default=2000, help="Number of iterations"
    )
    parser.add_argument("--log_freq", type=int, default=100, help="Log frequency")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--eps", type=float, default=1e-6, help="Epsilon value for numerical stability"
    )
    parser.add_argument(
        "--eps_norm", type=float, default=0.0, help="Epsilon value for normalization"
    )

    parser.add_argument(
        "--loss_type",
        type=str,
        default="preference",
        choices=[
            "entropy",
            "preference",
        ],
        help="Initialization method",
    )

    args = parser.parse_args()
    return args


def get_sse_and_sse_max(
    learned_ttdist: TTDist, true_ttdist: TTDist
) -> Tuple[float, float, float]:
    learned_dist = learned_ttdist.materialize().detach().numpy().squeeze()
    true_dist = true_ttdist.materialize().detach().numpy().squeeze()
    if learned_ttdist.norm_const is not None:
        learned_norm_const = learned_ttdist.norm_const.detach().numpy().squeeze().max()
    else:
        learned_norm_const = -1
    sse = ((learned_dist - true_dist) ** 2).sum()
    sse_max = ((learned_dist - true_dist) ** 2).max()
    return sse, sse_max, learned_norm_const


def get_true_ttdist(
    output_size: int,
    vocab_size: int,
    batch_size: int,
    rank: int,
    dist: str,
):

    alpha, beta, core = get_random_mps(
        batch_size=batch_size,
        rank=rank,
        vocab_size=vocab_size,
        dist=dist,
    )
    ttdist = TTDist(
        alpha,
        beta,
        core,
        output_size,
        norm_method="abs",
        norm_method_alpha="abs",
        eps=0.0,
    )
    return ttdist


def get_learned_ttdist(
    batch_size: int,
    rank: int,
    vocab_size: int,
    output_size: int,
    init_dist: str,
    norm_method: str,
    norm_method_alpha: str,
):
    alpha, beta, core = get_random_mps(
        batch_size=batch_size, rank=rank, vocab_size=vocab_size, dist=init_dist
    )
    # Forward pass:
    ttdist = TTDist(
        alpha,
        beta,
        core,
        output_size,
        norm_method=norm_method,
        norm_method_alpha=norm_method_alpha,
        eps=0.0,
    )
    return ttdist


def log_results(
    iteration: int, learned_ttdist: TTDist, true_ttdist: TTDist, loss: torch.Tensor
):
    sse, sse_max, norm_const = get_sse_and_sse_max(learned_ttdist, true_ttdist)
    print(
        f"[{iteration}] Loss = {loss.item():.3f} | SSE = {sse:.3f} | SSE_MAX = {sse_max:.3f} | norm_constant = {norm_const:.3f} "
    )
    # Log to W&B
    wandb.log(
        {
            "Loss": loss.item(),
            "SSE": sse,
            "SSE_MAX": sse_max,
            "norm_constant": norm_const,
        }
    )


def main(
    rank,
    true_rank,
    output_size,
    vocab_size,
    norm_method,
    batch_size,
    n_iters,
    log_freq,
    lr,
    eps,
    eps_norm,
    init_dist,
    loss_type,  # entropy or preference
    true_dist,  # normal or dirac
):

    # Assertions
    assert eps != eps_norm, "eps and eps_norm cannot be the same"

    # 1. Get the true distribution
    true_ttdist = get_true_ttdist(
        batch_size=1,
        output_size=output_size,
        vocab_size=vocab_size,
        rank=true_rank,
        dist=true_dist,
    )

    learned_ttdist = get_learned_ttdist(
        batch_size=batch_size,
        rank=rank,
        vocab_size=vocab_size,
        output_size=output_size,
        init_dist=init_dist,
        norm_method=norm_method,
        norm_method_alpha=norm_method,
    )

    optimizer = torch.optim.AdamW([learned_ttdist.core], lr=lr)
    loss_func = {
        "entropy": get_entropy_loss,
        "preference": get_preference_loss,
    }[loss_type]

    for i in range(n_iters):
        samples = true_ttdist.sample(batch_size)
        optimizer.zero_grad()

        loss = loss_func(
            ttdist=learned_ttdist, samples=samples, eps=eps, vocab_size=vocab_size
        )
        loss.backward()

        # Check if loss is NaN/Inf
        check_naninf(loss, msg="Loss")

        # Clip gradients to prevent exploding
        params_dict = learned_ttdist.get_params()
        alpha, beta, core = (
            params_dict["alpha"],
            params_dict["beta"],
            params_dict["core"],
        )
        torch.nn.utils.clip_grad_value_((alpha, beta, core), 1.0)

        if i % log_freq == 0 or i == 0:
            log_results(
                iteration=i,
                learned_ttdist=learned_ttdist,
                true_ttdist=true_ttdist,
                loss=loss,
            )

        optimizer.step()


if __name__ == "__main__":
    args = parse_args()
    experiment_config = vars(args)

    experiment_name = get_experiment_name(experiment_config)
    wandb.init(
        project="TJDNet (Synthetic)",
        config=experiment_config,
        name=experiment_name,
    )
    print(f"Running experiment: {experiment_name}")
    main(**experiment_config)
    wandb.finish()
