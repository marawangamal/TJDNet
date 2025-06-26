"""
Minimal script to analyze eigen spectrum of p(y1, y2|x) for datasets with varying rank complexity.

Usage:
    python scripts/plots/plot_dataset_rank_simple.py --model gpt2
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


def get_samples(debug=False, num_samples=2):
    """Get samples from three datasets with varying complexity"""

    # WikiText-2 (low rank) - structured text
    wikitext = load_dataset(
        "wikitext", "wikitext-2-raw-v1", split=f"train[:{num_samples*5}]"
    )
    wikitext_samples = [
        sample["text"][:200] for sample in wikitext if sample["text"].strip()
    ][:num_samples]

    # SST-2 (medium rank) - sentiment analysis
    sst2 = load_dataset("glue", "sst2", split="train[:100]")
    sst2_samples = [
        sample["sentence"] for sample in sst2 if sample["sentence"].strip()
    ][:num_samples]

    if debug:
        print("=== Low Rank Samples (WikiText-2) ===")
        for i, sample in enumerate(wikitext_samples):
            print(f"{i+1}. {sample[:100]}...")
        print("\n=== Medium Rank Samples (SST-2) ===")
        for i, sample in enumerate(sst2_samples):
            print(f"{i+1}. {sample[:100]}...")
        print("=== Done ===")

    return {
        "low_rank": wikitext_samples,
        "medium_rank": sst2_samples,
    }


def get_spectrum(model, tokenizer, text, device):
    """Get spectrum of p(y1, y2|x) for given text"""
    # Tokenize and get logits
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
    with torch.no_grad():
        outputs = model(inputs["input_ids"].to(device))
        logits = outputs.logits[0, -1, :]  # Last position

    # Create joint probability matrix (simplified)
    p = torch.softmax(logits, dim=-1)
    joint_prob = torch.outer(p, p)  # p(y1, y2|x) ≈ p(y1|x) * p(y2|x)

    # Get spectrum
    U, s, Vh = randomized_svd(
        joint_prob.cpu().numpy(), n_components=50, random_state=42
    )
    return torch.tensor(s)


def plot_spectra(spectra, save_path=None):
    """Plot spectra comparison"""
    plt.figure(figsize=(10, 6))

    colors = {"low_rank": "blue", "medium_rank": "orange"}
    names = {"low_rank": "WikiText-2", "medium_rank": "SST-2"}

    for category, spectrum_list in spectra.items():
        avg_spectrum = torch.stack(spectrum_list).mean(dim=0)
        normalized = avg_spectrum / avg_spectrum[0]
        plt.semilogy(
            normalized,
            color=colors[category],
            linewidth=2,
            label=f"{names[category]} ({len(spectrum_list)} samples)",
        )

    plt.xlabel("Singular Value Index")
    plt.ylabel("Normalized Singular Value")
    plt.title("Eigen Spectrum: p(y1, y2|x) Across Datasets")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="distilbert/distilgpt2")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing progress checkpoint.",
    )
    args = parser.parse_args()

    # Progress file path
    progress_path = "results/spectrum_progress.pt"
    os.makedirs(os.path.dirname(progress_path), exist_ok=True)

    # Load model
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)

    # Get samples
    print("Loading datasets...")
    datasets = get_samples(debug=True)

    # Try to load progress, unless overwrite is set
    if os.path.exists(progress_path) and not args.overwrite:
        print(f"Loading progress from {progress_path}...")
        spectra = torch.load(progress_path)
    else:
        if os.path.exists(progress_path) and args.overwrite:
            print(f"Overwrite flag set. Removing existing progress at {progress_path}.")
            os.remove(progress_path)
        spectra = {"low_rank": [], "medium_rank": []}

    # Compute spectra
    print("Computing spectra...")
    for category, samples in datasets.items():
        for i, text in enumerate(samples):
            # Skip if already computed
            if len(spectra[category]) > i:
                continue
            try:
                # Compute spectrum
                spectrum = get_spectrum(model, tokenizer, text, args.device)
                spectra[category].append(spectrum)
                print(f"{category} [{i+1}/{len(samples)}]: {spectrum[:3]}...")
                # Save progress
                torch.save(spectra, progress_path)
            except Exception as e:
                print(f"Error with {category}: {e}")

    # Plot
    print("Plotting...")
    plot_spectra(spectra, "results/spectrum_comparison.png")

    # Print summary
    print("\nSUMMARY:")
    for category, spectrum_list in spectra.items():
        if spectrum_list:
            avg_spectrum = torch.stack(spectrum_list).mean(dim=0)
            rank_99 = (
                torch.cumsum(avg_spectrum**2, 0) / (avg_spectrum**2).sum() < 0.99
            ).sum().item() + 1
            print(f"{category}: rank for 99% variance = {rank_99}")


if __name__ == "__main__":
    main()
