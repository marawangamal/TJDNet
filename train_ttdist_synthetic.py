"""Script to investigate using different non-linearities for the TT Dist.

Compares: 
1. ReLU
2. Softplus
3. Sigmoid
4. Exponential
5. (Baseline) Unrestricted
"""

import os
import random
import torch
import numpy as np
from TJDNet import TTDist, sample_from_tensor_dist

import matplotlib.pyplot as plt

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def make_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def main(
    rank: int = 1,
    output_size: int = 4,
    vocab_size: int = 4,
    norm_method: str = "relu",
    batch_size: int = 5,
    n_iters: int = 1000,
    log_freq: int = 100,
):
    # Generate a random true distribution
    true_dist = torch.abs(torch.randn(*[vocab_size for _ in range(output_size)]))
    true_dist = true_dist / true_dist.sum()  # P(d1, d2, ..., dN)

    # Sample `batch_size` random samples from the true distribution
    samples = sample_from_tensor_dist(true_dist, batch_size)

    alpha = torch.nn.Parameter(torch.randn(batch_size, rank))
    beta = torch.nn.Parameter(torch.randn(batch_size, rank))
    core = torch.nn.Parameter(torch.randn(batch_size, rank, vocab_size, rank))

    optimizer = torch.optim.Adam([alpha, beta, core], lr=1e-3)

    sse_loss_values = []
    sse_loss_baseline = []
    for i in range(n_iters):
        optimizer.zero_grad()

        # Forward pass:
        ttdist = TTDist(alpha, beta, core, output_size, norm_method=norm_method)
        probs_tilde, norm_constant = ttdist.get_prob_and_norm(samples)
        loss = (-torch.log(probs_tilde) + torch.log(norm_constant)).mean()

        if i % log_freq == 0 or i == 0:
            # Materialize the learned distribution
            learned_dist = ttdist.materialize().detach().numpy().squeeze()
            sse = ((learned_dist - true_dist.detach().numpy()) ** 2).sum()
            sse_baseline = ((0 - true_dist.detach().numpy()) ** 2).sum()
            print(f"[{i}] {norm_method}: SSE = {sse:.3f} | Loss = {loss.item():.3f}")
            sse_loss_values.append(sse)
            sse_loss_baseline.append(sse_baseline)

        # Backward pass:
        loss.backward()
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
    normalizations_methods = [
        # "relu",
        "abs",
        # "softmax",
        # "sigmoid",
    ]

    plt.figure(figsize=(10, 5))
    for i, norm_method in enumerate(normalizations_methods):
        log_dir = "logs/ttdist-synthetic"
        sse_loss_values, sse_loss_values_baseline = main(
            norm_method=norm_method, log_freq=100, n_iters=1000
        )
        plt.plot(
            sse_loss_values, label=f"{norm_method}", marker=markers[i % len(markers)]
        )
        plt.plot(
            sse_loss_values_baseline,
            label=f"{norm_method}_baseline",
            marker=markers[i % len(markers)],
        )
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("MSE Curve")
    plt.legend()
    make_dir_if_not_exists(log_dir)
    plt.savefig(f"{log_dir}/loss_curve.png")


if __name__ == "__main__":
    run_normalizations_exp()
