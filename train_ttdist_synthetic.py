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
from TJDNet import TTDist, sample_from_tensor_dist
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


def get_init_params_uniform_std_positive(batch_size, rank, output_size, vocab_size):
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


def main(
    rank: int = 1,
    output_size: int = 3,
    vocab_size: int = 4,
    norm_method: str = "relu",
    batch_size: int = 5,
    n_iters: int = 1000,
    log_freq: int = 100,
    lr: float = 1e-4,
    eps: float = 0.0,
    eps_norm: float = 1e-6,
    init_method: str = "uniform_positive",
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

    # Generate a random true distribution
    true_dist = torch.abs(torch.zeros(*[vocab_size for _ in range(output_size)]))
    # Set only one element to 1
    true_dist[tuple([1 for _ in range(output_size)])] = 1
    true_dist = true_dist / true_dist.sum()  # P(d1, d2, ..., dN)

    # Sample `batch_size` random samples from the true distribution
    samples = sample_from_tensor_dist(true_dist, batch_size)

    # Initialize the parameters
    alpha, beta, core = init_func(batch_size, rank, output_size, vocab_size)

    optimizer = torch.optim.AdamW([core], lr=lr)

    sse_loss_values = []
    sse_loss_baseline = []
    target_unnorm_prob, alt_unnorm_prob = 100, 100
    for i in range(n_iters):
        optimizer.zero_grad()

        # Forward pass:
        ttdist = TTDist(
            alpha,
            beta,
            core,
            output_size,
            norm_method=norm_method,
            norm_method_alpha=norm_method,
            eps=0.0,
        )
        probs_tilde, norm_constant = ttdist.get_prob_and_norm(samples)
        # retain_grad on probs_tilde to compute the gradient of the loss w.r.t. probs_tilde
        probs_tilde.retain_grad()
        norm_constant.retain_grad()
        loss = (-torch.log(probs_tilde + eps) + torch.log(norm_constant)).mean()

        # Backward pass:
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
            target_unnorm_prob = (
                torch.abs(alpha[0].reshape(1, -1))
                @ torch.abs(core[0, :, 1, :])
                @ torch.abs(beta[0].reshape(-1, 1))
            ).item()
            alt_unnorm_prob = sum(
                [
                    (
                        torch.abs(alpha[0].reshape(1, -1))
                        @ torch.abs(core[0, :, i, :])
                        @ torch.abs(beta[0].reshape(-1, 1))
                    ).item()
                    for i in range(vocab_size)
                    if i != 1
                ]
            )
            norm_constant = norm_constant[0].item()
            learned_dist = ttdist.materialize().detach().numpy().squeeze()
            learned_dist = learned_dist / learned_dist.reshape(batch_size, -1).sum(
                1
            ).reshape(tuple([batch_size, *[1 for _ in range(output_size)]]))
            sse = ((learned_dist - true_dist.detach().numpy()) ** 2).sum()
            sse_max = ((learned_dist - true_dist.detach().numpy()) ** 2).max()
            print(
                f"[{i}] {norm_method}: SSE = {sse:.3f} | SSE_MAX: {sse_max:.3f} | Loss = {loss.item():.3f} | target_prob = {target_unnorm_prob:.3f} | alt_prob = {alt_unnorm_prob:.3f} | norm_constant = {norm_constant:.3f}"
            )
            # Log SSE and loss to TensorBoard
            # writer.add_scalar(f"SSE", sse, i)
            # writer.add_scalar(f"Loss", loss.item(), i)
            # writer.add_histogram(f"Core_Gradients", core.grad, i)
            # writer.add_scalar(f"target_prob", target_unnorm_prob, i)

            # Log to W&B
            wandb.log(
                {
                    "SSE": sse,
                    "SSE_MAX": sse_max,
                    "Loss": loss.item(),
                    "target_prob": target_unnorm_prob,
                    "alt_prob": alt_unnorm_prob,
                    "norm_constant": norm_constant,
                }
            )

            if core.grad is not None:
                wandb.log(
                    {f"gradients/core": wandb.Histogram(core.grad.cpu().data.numpy())}
                )

        optimizer.step()

    if target_unnorm_prob > alt_unnorm_prob:
        print(
            f"[PASS]: {norm_method} target_prob ({target_unnorm_prob}) > alt_prob ({alt_unnorm_prob})"
        )
    else:
        print(
            f"[FAIL]: {norm_method} target_prob ({target_unnorm_prob}) < alt_prob ({alt_unnorm_prob})"
        )

    return sse_loss_values, sse_loss_baseline


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
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size")
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

    args = parser.parse_args()
    experiment_config = vars(args)

    experiment_name = get_experiment_name(experiment_config)
    wandb.init(
        project="tjdnet",
        config=experiment_config,
        name=experiment_name,
    )
    main(**experiment_config)
    wandb.finish()
