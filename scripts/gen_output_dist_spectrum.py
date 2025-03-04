"""Generates a spectrum plot of the output distribution for a language model.

Example:
    python gen_output_dist_spectrum.py --model meta-llama/Llama-2-7b-chat-hf
"""

from argparse import Namespace
import argparse
import os
from typing import Union
from tqdm import tqdm

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

import numpy as np
import matplotlib.pyplot as plt


def get_spectrum(output_mat):
    """Get spectrum of 2D matrix by computing singular values"""
    _, s, _ = torch.linalg.svd(output_mat)
    return s


def plot_spectrum(spectrum, save_path=None):
    """Plot the spectrum and save the figure if a path is provided"""
    plt.figure(figsize=(10, 6))

    # Convert to numpy for plotting
    if torch.is_tensor(spectrum):
        spectrum_np = spectrum.cpu().numpy()
    else:
        spectrum_np = spectrum

    # Plot singular values
    plt.semilogy(np.arange(1, len(spectrum_np) + 1), spectrum_np, "o-")
    plt.grid(True, which="both", ls="--")
    plt.xlabel("Index")
    plt.ylabel("Singular Value (log scale)")
    plt.title("Spectrum of Output Distribution Matrix")

    # Add a horizontal line at y=1 for reference
    plt.axhline(y=1, color="r", linestyle="-", alpha=0.3)

    # Add info about the effective rank
    effective_rank = (spectrum_np > 1e-10).sum()
    total_energy = spectrum_np.sum()
    energy_90 = np.searchsorted(np.cumsum(spectrum_np) / total_energy, 0.9) + 1

    plt.annotate(
        f"Effective rank: {effective_rank}",
        xy=(0.02, 0.95),
        xycoords="axes fraction",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    plt.annotate(
        f"90% energy at k={energy_90}",
        xy=(0.02, 0.89),
        xycoords="axes fraction",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    return plt.gcf()


def generate_output_distribution_spectrum(
    model: torch.nn.Module,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    start_str: str = "\n",
    checkpoint_steps: int = 100,  # Save progress every n tokens
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    # Ensure 'results' directory exists for checkpoint
    os.makedirs("results", exist_ok=True)
    checkpoint_path = "results/output_mat_checkpoint.pt"

    # Initialize or resume
    vocab_size = len(tokenizer.get_vocab())
    output_mat = torch.zeros((vocab_size, vocab_size))
    start_idx = 0

    if os.path.exists(checkpoint_path):
        # If there's a checkpoint, load it
        saved_mat, saved_idx = torch.load(checkpoint_path)
        # Copy the saved matrix into our newly allocated matrix in case sizes match
        output_mat[: saved_mat.shape[0], : saved_mat.shape[1]] = saved_mat
        start_idx = saved_idx
        print(f"Resuming from {checkpoint_path} at token index {start_idx}.")

    input_ids = torch.tensor(tokenizer.encode(start_str, return_tensors="pt")).to(
        device
    )  # Shape: (1, seq_len)
    model.to(device)

    for i in tqdm(
        range(start_idx, vocab_size),
        desc="Processing tokens",
        unit="token",
        leave=False,
        dynamic_ncols=True,
        smoothing=0.1,
        colour="green",
    ):
        with torch.no_grad():
            # p(y2) = model(x, y1)
            outputs = model(
                torch.cat(
                    [input_ids, torch.tensor([i]).to(device).reshape(1, 1)], dim=-1
                )
            )
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits[0, -1])
        output_mat[i] = probs[0]

        # Periodically save checkpoint
        if (i + 1) % checkpoint_steps == 0 or i + 1 == vocab_size:
            torch.save((output_mat, i + 1), checkpoint_path)
            if (i + 1) < vocab_size:
                print(f"Checkpoint saved at index {i + 1}.")

    return output_mat


def main(args: Namespace):
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Generate output distribution spectrum (resuming or starting fresh)
    print("Generating output distribution matrix...")
    output_mat = generate_output_distribution_spectrum(model, tokenizer)

    print("Computing spectrum...")
    spectrum = get_spectrum(output_mat)

    print("Plotting spectrum...")
    os.makedirs("results", exist_ok=True)
    plot_spectrum(spectrum, save_path=f"results/spectrum_plot_{args.model}.png")

    # Print some statistics
    print(f"Top 10 singular values: {spectrum[:10]}")
    print(f"Sum of all singular values: {spectrum.sum()}")
    print(f"Effective rank (singular values > 1e-10): {(spectrum > 1e-10).sum()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze the output spectrum of language models"
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="gpt2",  # meta-llama/Llama-2-7b-chat-hf
        help="Hugging Face model identifier (default: gpt2)",
    )
    args = parser.parse_args()
    main(args)
