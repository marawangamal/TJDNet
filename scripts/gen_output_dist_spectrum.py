if __name__ == "__main__":
    pass
# https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html


from argparse import Namespace
import argparse
import os
from typing import Any, Union
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
):
    vocab_size = len(tokenizer.get_vocab())
    output_mat = torch.zeros((vocab_size, vocab_size))
    input_ids = torch.tensor(
        tokenizer.encode(start_str, return_tensors="pt")
    )  # (1, input_seq_len)

    # Get dist over first output y_1
    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits
    # P(y_1| x)  v-dimensional
    probs = torch.nn.functional.softmax(logits[0, -1])  # (1, vocab_size)

    for i in tqdm(
        range(vocab_size),
        desc="Processing tokens",
        unit="token",
        leave=False,
        dynamic_ncols=True,
        smoothing=0.1,
        colour="green",
    ):
        with torch.no_grad():
            outputs = model(
                torch.cat([input_ids, torch.tensor([i]).reshape(1, 1)], dim=-1)
            )
        logits = outputs.logits
        # P(y_1=j, y_2| x) v-dimensional
        probs = torch.nn.functional.softmax(logits[0, -1])
        output_mat[i] = probs[0]

    return output_mat


def main(args: Namespace):
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Generate output distribution spectrum
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
