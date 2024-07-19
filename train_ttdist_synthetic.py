"""Script to investigate using different non-linearities for the TT Dist.

Compares: 
1. ReLU
2. Softplus
3. Sigmoid
4. Exponential
5. (Baseline) Unrestricted
"""

from typing import Union, Optional
import os
import argparse
import shutil
import random
import torch
import numpy as np
from TJDNet import sample_from_tensor_dist
from TJDNet.TJDLayer.TTDist import TTDist
from TJDNet.TJDLayer.utils import get_init_params_uniform_std_positive
from utils.utils import get_experiment_name

import matplotlib.pyplot as plt
from torch.utils.tensorboard.writer import SummaryWriter
import wandb


# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def remove_dir(dir_path: str):
    # Avoid accidentally deleting the root directory
    protected_dirs = ["/", ".", ".."]
    min_dir_path_len = 10
    assert dir_path not in protected_dirs, "Refusing to delete protected directory"
    assert len(dir_path) > min_dir_path_len, "Refusing to delete root directory"
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)


def plot_grad_distribution(core):
    grads = core.grad.detach().cpu().numpy().flatten()
    plt.hist(grads, bins=50, alpha=0.75)
    plt.xlabel("Gradient")
    plt.ylabel("Frequency")
    plt.title("Gradient Distribution of Core Parameter")
    plt.grid(True)
    plt.show()


def get_init_params_concentrated(batch_size, rank, output_size, vocab_size):
    alpha = torch.ones(batch_size, rank)
    beta = torch.ones(batch_size, rank)
    # Cores should be zero except for a single vocab index
    coreZero = torch.zeros(rank, vocab_size, rank)
    coreOneHot = torch.zeros(rank, vocab_size, rank)
    coreOneHot[:, 0, :] = torch.eye(rank)
    core = torch.nn.Parameter(
        (coreZero + coreOneHot).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    )
    return alpha, beta, core


def get_init_params_uniform_std(batch_size, rank, output_size, vocab_size):
    alpha = (
        torch.randn(1, rank).repeat(batch_size, 1)
        * torch.sqrt(torch.tensor(1 / vocab_size**output_size))
    ).abs()
    beta = (
        torch.randn(1, rank).repeat(batch_size, 1)
        * torch.sqrt(torch.tensor(1 / vocab_size**output_size))
    ).abs()
    core = torch.nn.Parameter(
        torch.eye(rank)
        .unsqueeze(1)
        .repeat(1, vocab_size, 1)
        .unsqueeze(0)
        .repeat(batch_size, 1, 1, 1)
    )
    return alpha, beta, core


def get_init_params_randn(batch_size, rank, output_size, vocab_size):
    alpha = torch.randn(1, rank).repeat(batch_size, 1)
    beta = torch.randn(1, rank).repeat(batch_size, 1)
    core = torch.nn.Parameter(
        torch.randn(rank, vocab_size, rank).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    )
    return alpha, beta, core


def get_init_params_randn_positive(batch_size, rank, output_size, vocab_size):
    alpha = (torch.randn(1, rank).repeat(batch_size, 1)).abs()
    beta = (torch.randn(1, rank).repeat(batch_size, 1)).abs()
    core = torch.nn.Parameter(
        torch.randn(rank, vocab_size, rank)
        .abs()
        .unsqueeze(0)
        .repeat(batch_size, 1, 1, 1)
    )
    return alpha, beta, core


def get_entropy_loss(
    ttdist: TTDist, samples: torch.Tensor, eps: float = 1e-6, vocab_size: int = 4
):
    probs_tilde, norm_constant = ttdist.get_prob_and_norm(samples)
    # retain_grad on probs_tilde to compute the gradient of the loss w.r.t. probs_tilde
    probs_tilde.retain_grad()
    norm_constant.retain_grad()
    loss = (-torch.log(probs_tilde + eps) + torch.log(norm_constant)).mean()
    return loss, norm_constant


def get_preference_loss(
    ttdist: TTDist,
    samples: torch.Tensor,
    eps: float = 1e-6,
    vocab_size: int = 4,
    neg_samples_multiplier: int = 1000,
    num_neg_batches: int = 10,
):
    samples_pos = samples
    batch_size = samples_pos.shape[0]
    output_size = samples_pos.shape[1]
    # make a batch of negative samples by creating rand tensor of size batch_size x output_size with indices in [0, vocab_size-1]
    probs_tilde_pos, norm_constant_pos = ttdist.get_prob_and_norm(samples_pos)
    # probs_tilde_neg, norm_constant_neg = ttdist.get_prob_and_norm(samples_neg)
    probs_tilde_neg_lst = [
        ttdist.get_prob_and_norm(
            torch.randint(
                0,
                vocab_size,
                (batch_size, output_size),
                dtype=torch.long,
                device=samples.device,
            )
        )[0]
        for _ in range(num_neg_batches)
    ]

    probs_tilde_neg_sum = torch.stack(probs_tilde_neg_lst, dim=0).sum(dim=0)
    preference_loss = -torch.log(probs_tilde_pos + eps) + torch.log(
        probs_tilde_neg_sum + eps
    )
    preference_loss = preference_loss.mean()

    # # the loss is  ...
    # preference_loss = torch.maximum(
    #     (-torch.log(probs_tilde_pos + eps) + torch.log(probs_tilde_neg + eps)).mean(),
    #     torch.zeros_like(probs_tilde_pos),
    # )

    # # Calculate preference diffs
    # preference_diffs = -torch.log(
    #     probs_tilde_pos + eps
    # ) + neg_samples_multiplier * torch.log(probs_tilde_neg + eps)

    # # Filter out negative numbers
    # preference_diffs_pos = preference_diffs[preference_diffs > 0]
    # preference_loss = (
    #     preference_diffs_pos.mean() if len(preference_diffs_pos) > 0 else None
    # )
    # # Check if loss is NaN/Inf
    # if preference_loss is not None and (
    #     torch.isnan(preference_loss) or torch.isinf(preference_loss)
    # ):
    #     print(f"[FAIL]: Loss is NaN. Breaking at")

    return preference_loss, probs_tilde_neg_sum


def get_true_ttdist(output_size: int, vocab_size: int) -> TTDist:
    raise NotImplementedError


def get_expected_sse(learned_dist, true_dist) -> float:
    raise NotImplementedError


def get_expected_sse_max(learned_dist, true_dist) -> float:
    raise NotImplementedError


def main(
    rank: int = 1,
    output_size: int = 3,
    vocab_size: int = 4,
    norm_method: str = "relu",
    batch_size: int = 16,
    n_iters: int = 1000,
    log_freq: int = 100,
    lr: float = 1e-4,
    eps: float = 0.0,
    eps_norm: float = 1e-6,
    init_method: str = "uniform_positive",
    loss_type: str = "entropy",  # entropy or preference
):

    assert eps != eps_norm, "eps and eps_norm cannot be the same"
    assert init_method in [
        "concentrated",
        "uniform",
        "uniform_positive",
        "randn",
        "randn_positive",
    ], f"init_method must be one of ['concentrated', 'uniform', 'uniform_positive', 'randn', 'randn_positive']"
    init_func = {
        "concentrated": get_init_params_concentrated,
        "uniform": get_init_params_uniform_std,
        "uniform_positive": get_init_params_uniform_std_positive,
        "randn": get_init_params_randn,
        "randn_positive": get_init_params_randn_positive,
    }[init_method]
    assert loss_type in [
        "entropy",
        "preference",
    ], "loss_type must be one of ['entropy', 'preference']"

    true_ttdist = get_true_ttdist(output_size, vocab_size)
    samples = true_ttdist.sample(batch_size)

    # Initialize the parameters
    alpha, beta, core = init_func(batch_size, rank, output_size, vocab_size)

    optimizer = torch.optim.AdamW([core], lr=lr)

    for i in range(n_iters):
        optimizer.zero_grad()

        # Forward pass:
        learned_ttdist = TTDist(
            alpha,
            beta,
            core,
            output_size,
            norm_method=norm_method,
            norm_method_alpha=norm_method,
            eps=0.0,
        )
        # loss = {get_entropy_loss}(ttdist, samples, eps=eps)
        loss, norm_constant = {
            "entropy": get_entropy_loss,
            "preference": get_preference_loss,
        }[loss_type](learned_ttdist, samples, eps=eps, vocab_size=vocab_size)

        # Backward pass:
        if loss is None:
            if i % log_freq == 0 or i == 0:
                print("Loss is None. Skipping")
            continue
        loss.backward()

        # Check if loss is NaN/Inf
        if torch.isnan(loss):
            print(f"[FAIL]: Loss is NaN. Breaking at iteration {i}")
            break
        elif torch.isinf(loss):
            print(f"[FAIL]: Loss is Inf. Breaking at iteration {i}")
            break

        # Clip gradients to prevent exploding
        torch.nn.utils.clip_grad_value_((alpha, beta, core), 1.0)

        if i % log_freq == 0 or i == 0:
            # Materialize the learned distribution

            expected_sse = get_expected_sse(learned_ttdist, true_ttdist)
            expected_sse_max = get_expected_sse_max(learned_ttdist, true_ttdist)

            print(
                f"[{i}] {norm_method}: SSE = {expected_sse:.3f} | SSE_MAX: {expected_sse_max:.3f} | Loss = {loss.item():.3f} | norm_constant = {norm_constant:.3f}"
            )

            # Log to W&B
            wandb.log(
                {
                    "SSE": expected_sse,
                    "SSE_MAX": expected_sse_max,
                    "Loss": loss.item(),
                    "norm_constant": norm_constant,
                }
            )

            if core.grad is not None:
                wandb.log({f"core_grad": wandb.Histogram(core.grad.cpu().data.numpy())})

        optimizer.step()


def run_normalizations_exp():
    markers = [
        "o",
        "v",
        "^",
        "<",
        ">",
        "1",
        "2",
        "3",
        "4",
        "s",
        "p",
        "*",
        "h",
        "H",
        "+",
        "x",
        "D",
        "d",
    ]
    plt.figure(figsize=(10, 5))
    for init_method in ["uniform_positive", "randn_positive"]:
        experiment_config = {
            "rank": 2,
            "output_size": 3,
            "vocab_size": 4,
            "norm_method": "abs",
            "lr": 1e-4,
            "n_iters": 20 * 1000,
            "eps": 1e-6,
            "eps_norm": 1e-9,
            "init_method": init_method,
        }
        experiment_name = get_experiment_name(experiment_config)
        remove_dir(os.path.join("logs", experiment_name))
        main(**experiment_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to investigate using different non-linearities for the TT Dist."
    )
    parser.add_argument(
        "--rank", type=int, default=2, help="Rank of the tensor decomposition"
    )
    parser.add_argument(
        "--output_size", type=int, default=3, help="Output size of the tensor"
    )
    parser.add_argument("--vocab_size", type=int, default=4, help="Vocabulary size")
    parser.add_argument(
        "--norm_method", type=str, default="abs", help="Normalization method"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--n_iters", type=int, default=20000, help="Number of iterations"
    )
    parser.add_argument("--log_freq", type=int, default=100, help="Log frequency")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--eps", type=float, default=1e-6, help="Epsilon value for numerical stability"
    )
    parser.add_argument(
        "--eps_norm", type=float, default=0.0, help="Epsilon value for normalization"
    )
    parser.add_argument(
        "--init_method",
        type=str,
        default="randn_positive",
        choices=[
            "concentrated",
            "uniform",
            "uniform_positive",
            "randn",
            "randn_positive",
        ],
        help="Initialization method",
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
    experiment_config = vars(args)

    experiment_name = get_experiment_name(experiment_config)
    wandb.init(
        project="tjdnet",
        config=experiment_config,
        name=experiment_name,
    )
    print(f"Running experiment: {experiment_name}")
    main(**experiment_config)
    wandb.finish()
