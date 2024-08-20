from typing import Tuple
import argparse
import random

import torch
import numpy as np
import wandb

from TJDNet import MPSDist

from TJDNet.loss import get_preference_loss, get_entropy_loss, get_entropy_unnorm_loss
from TJDNet.utils import check_naninf
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
        default="sparse",
        help="True distribution",
        choices=[
            "unit_var",
            "one_hot",
            "sparse",
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
        "--rank", type=int, default=4, help="Rank of the tensor decomposition"
    )
    parser.add_argument(
        "--true_rank", type=int, default=8, help="Rank of the tensor decomposition"
    )

    parser.add_argument(
        "--output_size", type=int, default=5, help="Output size of the tensor"
    )
    parser.add_argument("--vocab_size", type=int, default=3, help="Vocabulary size")
    parser.add_argument(
        "--norm_method", type=str, default="abs", help="Normalization method"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--n_iters", type=int, default=20000, help="Number of iterations"
    )
    parser.add_argument("--log_freq", type=int, default=100, help="Log frequency")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--eps", type=float, default=1e-9, help="Epsilon value for numerical stability"
    )
    parser.add_argument(
        "--eps_norm", type=float, default=0.0, help="Epsilon value for normalization"
    )

    parser.add_argument(
        "--loss_type",
        type=str,
        default="entropy",
        choices=[
            "entropy",
            "entropy:unnorm",
            "preference",
        ],
        help="Initialization method",
    )

    args = parser.parse_args()
    return args


def get_sae_and_mae(
    learned_ttdist: MPSDist, true_ttdist: MPSDist
) -> Tuple[float, float, float]:
    learned_dist = learned_ttdist.materialize().detach().numpy().squeeze()
    true_dist = true_ttdist.materialize().detach().numpy().squeeze()
    if learned_ttdist.norm_const is not None:
        learned_norm_const = learned_ttdist.norm_const.detach().numpy().squeeze().max()
    else:
        learned_norm_const = -1
    sum_abs_error = np.abs(learned_dist - true_dist).sum()
    max_abs_error = np.abs(learned_dist - true_dist).max()
    return sum_abs_error, max_abs_error, learned_norm_const


def log_results(
    iteration: int,
    learned_ttdist: MPSDist,
    true_ttdist: MPSDist,
    loss: torch.Tensor,
):
    sae, mae, norm_const = get_sae_and_mae(learned_ttdist, true_ttdist)
    print(
        f"[{iteration}] Loss = {loss.item():.3f} | SAE = {sae:.3f} | MAE = {mae:.3f} | norm_constant = {norm_const:.3f} "
    )
    # Log to W&B
    wandb.log(
        {
            "Loss": loss.item(),
            "SAE": sae,
            "MAE": mae,
            "norm_constant": norm_const,
        }
    )


def print_transition_matrix(mpsdist: MPSDist, tag: str):
    mat = mpsdist.materialize(n_core_repititions=2).detach().numpy().squeeze()
    # print horizontal line
    print("-" * 80)
    print(f"[{tag}] Transition Matrix")
    print(np.round(mat, 2))
    print("-" * 80)


def main(
    rank,
    true_dist,
    true_rank,
    output_size,
    vocab_size,
    batch_size,
    n_iters,
    log_freq,
    lr,
    eps,
    eps_norm,
    loss_type,  # entropy or preference
    *args,
    **kwargs,
):

    # Assertions
    assert eps != eps_norm, "eps and eps_norm cannot be the same"

    # 1. Get the true distribution
    true_mpsdist = MPSDist(
        n_vocab=vocab_size,
        rank=true_rank,
        init_method=true_dist,
    )
    # Print the true distribution
    print_transition_matrix(true_mpsdist, "MPSDist (True)")

    learned_mpsdist = MPSDist(
        n_vocab=vocab_size,
        rank=rank,
    )

    optimizer = torch.optim.AdamW(learned_mpsdist.parameters(), lr=lr)
    loss_func = {
        "entropy": get_entropy_loss,
        "entropy:unnorm": get_entropy_unnorm_loss,
        "preference": get_preference_loss,
    }[loss_type]

    for i in range(n_iters):

        samples = true_mpsdist.sample(
            n_samples=batch_size,
            max_len=output_size,
        ).detach()
        optimizer.zero_grad()

        loss = loss_func(
            ttdist=learned_mpsdist, samples=samples, eps=eps, vocab_size=vocab_size
        )
        loss.backward()

        # Check if loss is NaN/Inf
        check_naninf(loss, msg="Loss")

        # Clip gradients to prevent exploding
        torch.nn.utils.clip_grad_value_(learned_mpsdist.parameters(), 1.0)

        if i % log_freq == 0 or i == 0:
            log_results(
                iteration=i,
                learned_ttdist=learned_mpsdist,
                true_ttdist=true_mpsdist,
                loss=loss,
            )
            print_transition_matrix(learned_mpsdist, "MPSDist (Learned)")

        optimizer.step()


if __name__ == "__main__":
    args = parse_args()
    experiment_config = vars(args)

    experiment_name = get_experiment_name(experiment_config)
    wandb.init(
        project="tjdnet-synthetic",
        config=experiment_config,
        name=experiment_name,
    )
    print(f"Running experiment: {experiment_name}")
    main(**experiment_config)
    wandb.finish()
