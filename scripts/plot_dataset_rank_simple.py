"""
Minimal script to analyze eigen spectrum of p(y1, y2|x) for datasets with varying rank complexity.

Usage:
    python scripts/plots/plot_dataset_rank_simple.py --model gpt2
"""

import os
import argparse
from tqdm import tqdm
from tabulate import tabulate

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


# Low rank datasets (structured, algorithmic, or formulaic answers):
#   - gsm8k (grade school math, train: 7473)
#   - aqua_rat (algebra word problems, train: 10160)
#   - asdiv (arithmetic, train: 2305)
#   - wikitext-2 (structured Wikipedia text, train: 36718)
#   - math_qa (math QA, train: 29937)
# Medium rank datasets (natural language, but with some structure):
#   - sst2 (sentiment analysis, train: 67349)
#   - imdb (movie reviews, train: 25000)
#   - ag_news (news categorization, train: 120000)
# High rank datasets (open-ended, diverse, conversational, or noisy):
#   - reddit (open-domain discussion, millions of samples)
#   - openwebtext (web crawl, millions of samples)
#   - c4 (Colossal Clean Crawled Corpus, millions of samples)
#   - stackexchange_qa (Q&A, diverse topics)
def get_samples(debug=False, num_samples=5):
    """Get samples from several datasets, using dataset names as keys."""
    dataset_configs = {
        # Low rank
        "aqua_rat": {
            "hf_name": ("aqua_rat",),
            "load_kwargs": {"split": f"train[:{num_samples*5}]"},
            "field": "question",
            "post": lambda s: s + " Answer:",
        },
        # "gsm8k": {
        #     "hf_name": ("gsm8k",),
        #     "load_kwargs": {"split": f"train[:{num_samples*5}]"},
        #     "field": "question",
        #     "post": lambda s: s + " Answer:",
        # },
        "wikitext2": {
            "hf_name": ("wikitext", "wikitext-2-raw-v1"),
            "load_kwargs": {"split": f"train[:{num_samples*5}]"},
            "field": "text",
            "post": lambda s: s[:200],
        },
        # Medium rank
        "sst2": {
            "hf_name": ("glue", "sst2"),
            "load_kwargs": {"split": "train[:100]"},
            "field": "sentence",
            "post": lambda s: s,
        },
        # High rank
        "reddit": {
            "hf_name": ("reddit",),
            "load_kwargs": {
                "split": f"train[:{num_samples*5}]",
                "trust_remote_code": True,
            },
            "field": "body",
            "post": lambda s: s,
        },
    }

    samples = {}
    for name, cfg in dataset_configs.items():
        ds = load_dataset(*cfg["hf_name"], **cfg["load_kwargs"])
        s = [
            cfg["post"](sample[cfg["field"]])
            for sample in ds
            if sample[cfg["field"]].strip()
        ][:num_samples]
        samples[name] = s
        if debug:
            print(f"=== {name} ===")
            for i, sample in enumerate(s):
                print(f"{i+1}. {sample[:100]}...")
    if debug:
        print("=== Done ===")
    return samples


def get_joint_prob(model, tokenizer, text, device, top_k=None):
    """Compute joint probability p(y1, y2 | x) for given text. If top_k is None, use all vocab."""
    x = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)[
        "input_ids"
    ].to(device)

    with torch.no_grad():
        out = model(x)
        logits_y1 = out.logits[0, -1, :]  # (vocab_size,)
        p_y1 = torch.softmax(logits_y1, dim=-1)

        if top_k is None:
            topk_p_y1 = p_y1
            topk_y1 = torch.arange(p_y1.size(0), device=device)
        else:
            topk_p_y1, topk_y1 = torch.topk(p_y1, top_k)
        joint = torch.zeros(
            (len(topk_y1), len(topk_y1)), device=device, dtype=torch.float64
        )

        for i, y1 in enumerate(
            tqdm(topk_y1, desc=f"Computing joint p(y1,y2|x) for k={top_k}", leave=False)
        ):
            x_y1 = torch.cat([x[0], y1.unsqueeze(0)]).unsqueeze(0)  # (1, L+1)
            out2 = model(x_y1)
            logits_y2 = out2.logits[0, -1, :]
            p_y2_given_y1 = torch.softmax(logits_y2, dim=-1)
            if top_k is None:
                topk_p_y2 = p_y2_given_y1
            else:
                topk_p_y2, _ = torch.topk(p_y2_given_y1, top_k)
            joint[i, :] = topk_p_y1[i] * topk_p_y2  # p(y1) * p(y2|y1)

    return joint.cpu()


def plot_spectra(spectra, save_path=None):
    """Plot spectra comparison"""
    plt.figure(figsize=(10, 6))

    colors = {
        "wikitext2": "blue",
        "sst2": "orange",
        "aqua_rat": "green",
        "reddit": "red",
    }
    names = {
        "wikitext2": "WikiText-2",
        "sst2": "SST-2",
        "aqua_rat": "AQuA (Math QA)",
        "reddit": "Reddit",
    }

    line_styles = {
        "wikitext2": "-",
        "sst2": "--",
        "aqua_rat": "-.",
        "reddit": ":",
    }

    for category in spectra.keys():
        spectrum_list = spectra[category]
        color = colors.get(category, None)
        name = names.get(category, category)
        style = line_styles.get(category, "-")

        # Plot each individual spectrum
        for i, spectrum in enumerate(spectrum_list):
            tensor = torch.as_tensor(spectrum)
            normalized = tensor / tensor[0]
            alpha = (
                0.3 if len(spectrum_list) > 1 else 1.0
            )  # More transparent if multiple samples
            plt.semilogy(
                normalized,
                color=color,
                linestyle=style,
                linewidth=1.5,
                alpha=alpha,
                label=(
                    f"{name} (sample {i+1})" if i == 0 else None
                ),  # Only label first line per category
            )

    plt.xlabel("Singular Value Index")
    plt.ylabel("Normalized Singular Value")
    plt.title("Eigen Spectrum: p(y1, y2|x) Across Datasets (All Samples)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def main():
    # Parse arguments
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
    parser.add_argument(
        "--top_k",
        type=int,
        default=1000,
        help="Number of top tokens to consider for joint probability computation.",
    )
    args = parser.parse_args()

    model_name = args.model.replace("/", "_")

    # Define filenames
    progress_path = f"results/spectrum_progress_{model_name}_topk{args.top_k}.pt"
    plot_path = f"results/spectrum_comparison_{model_name}_topk{args.top_k}.png"

    # Progress file path
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
        spectra = torch.load(progress_path, weights_only=False)
    else:
        if os.path.exists(progress_path) and args.overwrite:
            print(f"Overwrite flag set. Removing existing progress at {progress_path}.")
            os.remove(progress_path)
        spectra = {k: [] for k in ["wikitext2", "sst2", "aqua_rat", "reddit"]}

    # Compute spectra
    for category, samples in datasets.items():
        for i, text in enumerate(samples):
            # Skip if already computed
            if len(spectra[category]) > i:
                continue
            try:
                print(f"[{i+1}/{len(samples)}] Computing spectra for {category}...")
                # Compute spectrum
                p_y1y2 = get_joint_prob(model, tokenizer, text, args.device, args.top_k)
                # Save p(y1, y2 | x)
                # torch.save(p_y1y2, f"results/p_y1y2_{model_name}_{category}_{i}.pt")
                print(f"p_y1y2.shape: {p_y1y2.shape}")
                # Use minimum dimension to get all singular values
                n_components = min(p_y1y2.shape)
                _, spectrum, _ = randomized_svd(
                    p_y1y2.cpu().numpy(), n_components=n_components, random_state=42
                )
                spectra[category].append(torch.tensor(spectrum))
                print(f"{category} [{i+1}/{len(samples)}]: {spectrum[:3]}...")
                # Save progress
                torch.save(spectra, progress_path)
            except Exception as e:
                print(f"Error with {category}: {e}")

    # Plot
    print("Plotting...")
    plot_spectra(spectra, plot_path)

    summary_rows = []
    var_target = 0.99
    for category, spectrum_list in spectra.items():
        if spectrum_list:
            ranks = []
            for spectrum in spectrum_list:
                spectrum = spectrum.cpu()
                cumsum = torch.cumsum(spectrum**2, 0)
                total = (spectrum**2).sum()
                rank = ((cumsum / total) < var_target).sum().item() + 1
                ranks.append(rank)
            mean_rank = np.mean(ranks)
            std_rank = np.std(ranks)
            summary_rows.append([category, f"{mean_rank:.1f} ± {std_rank:.1f}"])
    print(
        tabulate(
            summary_rows,
            headers=[
                f"Category",
                f"Emprical Rank for {var_target*100}% Variance (mean ± std)",
            ],
            tablefmt="github",
        )
    )


if __name__ == "__main__":
    main()

# Model: meta-llama/Llama-2-7b-chat-hf
# | Category   | Rank for 99.0% Variance (mean ± std)   |
# |------------|----------------------------------------|
# | wikitext2  | 2.2 ± 0.7                              |
# | sst2       | 4.2 ± 1.9                              |
# | aqua_rat   | 3.6 ± 2.1                              |
# | reddit     | 5.0 ± 4.7                              |
