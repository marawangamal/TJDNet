"""Script to investigate using different non-linearities for the TT Dist.

Compares: 
1. ReLU
2. Softplus
3. Sigmoid
4. Exponential
5. (Baseline) Unrestricted
"""

import os
import shutil
import random
import torch
import numpy as np
from TJDNet import TTDist, sample_from_tensor_dist
from utils.utils import get_experiment_name

import matplotlib.pyplot as plt
from torch.utils.tensorboard.writer import SummaryWriter


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


def get_init_params_randn(rank, output_size, vocab_size):
    # alpha.shape
    # torch.Size([2])
    # core.shape
    # torch.Size([2, 4, 2])
    alpha = torch.nn.Parameter(torch.randn(rank))
    beta = torch.nn.Parameter(torch.randn(rank))
    core = torch.nn.Parameter(torch.randn(rank, vocab_size, rank))
    return alpha, beta, core


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


def get_init_params_uniform(batch_size, rank, output_size, vocab_size):
    alpha = torch.ones(batch_size, rank) * torch.sqrt(torch.tensor(1 / output_size))
    beta = torch.ones(batch_size, rank) * torch.sqrt(torch.tensor(1 / output_size))
    core = torch.nn.Parameter(
        torch.eye(rank)
        .unsqueeze(1)
        .repeat(1, vocab_size, 1)
        .unsqueeze(0)
        .repeat(batch_size, 1, 1, 1)
    )
    return alpha, beta, core


def get_init_params_uniform_std(batch_size, rank, output_size, vocab_size):
    # alpha = torch.ones(batch_size, rank) * torch.sqrt(torch.tensor(1 / output_size))
    # beta = torch.ones(batch_size, rank) * torch.sqrt(torch.tensor(1 / output_size))
    # Make alpha and beta random with std 1/sqrt(rank)
    alpha = torch.randn(1, rank).repeat(batch_size, 1) * torch.sqrt(
        torch.tensor(1 / rank)
    )
    beta = torch.randn(1, rank).repeat(batch_size, 1) * torch.sqrt(
        torch.tensor(1 / rank)
    )
    core = torch.nn.Parameter(
        torch.eye(rank)
        .unsqueeze(1)
        .repeat(1, vocab_size, 1)
        .unsqueeze(0)
        .repeat(batch_size, 1, 1, 1)
    )
    return alpha, beta, core


def main(
    writer: SummaryWriter,
    rank: int = 1,
    output_size: int = 3,
    vocab_size: int = 4,
    norm_method: str = "relu",
    batch_size: int = 5,
    n_iters: int = 1000,
    log_freq: int = 100,
    lr: float = 1e-4,
    eps: float = 0.0,
):
    # Generate a random true distribution
    true_dist = torch.abs(torch.zeros(*[vocab_size for _ in range(output_size)]))
    # Set only one element to 1
    true_dist[tuple([1 for _ in range(output_size)])] = 1
    true_dist = true_dist / true_dist.sum()  # P(d1, d2, ..., dN)

    # Sample `batch_size` random samples from the true distribution
    samples = sample_from_tensor_dist(true_dist, batch_size)

    # Initialize the parameters
    alpha, beta, core = get_init_params_uniform_std(
        batch_size, rank, output_size, vocab_size
    )

    optimizer = torch.optim.Adam([alpha, beta, core], lr=lr)

    sse_loss_values = []
    sse_loss_baseline = []
    for i in range(n_iters):
        optimizer.zero_grad()

        # Forward pass:
        ttdist = TTDist(
            alpha, beta, core, output_size, norm_method=norm_method, eps=0.0
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
            print(f"WARNING: Loss is NaN. Breaking at iteration {i}")
            break
        elif torch.isinf(loss):
            print(f"WARNING: Loss is Inf. Breaking at iteration {i}")
            break

        # Clip gradients to prevent exploding
        torch.nn.utils.clip_grad_value_((alpha, beta, core), 1.0)

        if i % log_freq == 0 or i == 0:
            # Materialize the learned distribution
            learned_dist = ttdist.materialize().detach().numpy().squeeze()
            sse = ((learned_dist - true_dist.detach().numpy()) ** 2).sum()
            # sse_baseline = ((0 - true_dist.detach().numpy()) ** 2).sum()
            print(f"[{i}] {norm_method}: SSE = {sse:.3f} | Loss = {loss.item():.3f}")
            # Log SSE and loss to TensorBoard
            writer.add_scalar(f"SSE", sse, i)
            writer.add_scalar(f"Loss", loss.item(), i)
            writer.add_histogram(f"Core_Gradients", core.grad, i)

        optimizer.step()

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
    for rank in [2]:
        experiment_config = {
            "rank": rank,
            "output_size": 3,
            "vocab_size": 4,
            "norm_method": "relu",
            "lr": 1e-4,
            "n_iters": 10000,
            "eps": 1e-6,
        }
        experiment_name = get_experiment_name(experiment_config)
        remove_dir(os.path.join("logs", experiment_name))
        writer = SummaryWriter(log_dir=os.path.join("logs", experiment_name))
        main(writer=writer, **experiment_config)


if __name__ == "__main__":
    run_normalizations_exp()
