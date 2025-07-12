"""Analyze eigen spectrum of p(y1, y2|x) for datasets with varying rank complexity.

Usage:
    python scripts/plot_dataset_rank_simple.py --model gpt2
"""

import os
import argparse
from typing import Dict, List, Tuple
from tqdm import tqdm
from tabulate import tabulate

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset


# TODO: Add STEMP and other low-rank datasets.
# Low rank datasets (structured, algorithmic, or formulaic answers):
#   - gsm8k (grade school math, train: 7473)
#   - aqua_rat (algebra word problems, train: 10160)
#   - csqa (commonsense QA, train: 9741)
#   - mbpp (Python programming, train: 374)
#   - humaneval (Python programming, test: 164)
#   - ai2_reasoning (science reasoning, train: 1119)
#   - asdiv (arithmetic, train: 2305)
#   - wikitext-2 (structured Wikipedia text, train: 36718)
#   - math_qa (math QA, train: 29937)
# Medium rank datasets (natural language, but with some structure):
#   - sst2 (sentiment analysis, train: 67349)
#   - hellaswag (commonsense reasoning, train: 39905)
#   - imdb (movie reviews, train: 25000)
#   - ag_news (news categorization, train: 120000)
# High rank datasets (open-ended, diverse, conversational, or noisy):
#   - reddit (open-domain discussion, millions of samples)
#   - openwebtext (web crawl, millions of samples)
#   - c4 (Colossal Clean Crawled Corpus, millions of samples)
#   - stackexchange_qa (Q&A, diverse topics)
def get_samples(debug=False, num_samples=5):
    """Get samples from several datasets, using dataset names as keys."""
    # Canonical few-shot context for each dataset
    aqua_rat_fewshot = (
        "Q: Two friends plan to walk along a 43-km trail, starting at opposite ends of the trail at the same time. If Friend P's rate is 15% faster than Friend Q's, how many kilometers will Friend P have walked when they pass each other?\nOptions: A)21 B)21.5 C)22 D)22.5 E)23\nAnswer: E\n\n"
        "Q: In the coordinate plane, points (x, 1) and (5, y) are on line k. If line k passes through the origin and has slope 1/5, then what are the values of x and y respectively?\nOptions: A)4 and 1 B)1 and 5 C)5 and 1 D)3 and 5 E)5 and 3\nAnswer: C\n\n"
    )
    gsm8k_fewshot = (
        "Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\n"
        "Answer: Natalia sold 48/2 = 24 clips in May. Natalia sold 48+24 = 72 clips altogether in April and May. #### 72\n\n"
        "Q: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\n"
        "Answer: Weng earns 12/60 = $0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $10. #### 10\n\n"
    )
    csqa_fewshot = (
        "Q: What is the primary function of the heart?\nA) To pump blood throughout the body\nB) To digest food\nC) To produce hormones\nD) To filter waste\nAnswer: A\n\n"
        "Q: Which planet is closest to the Sun?\nA) Earth\nB) Venus\nC) Mercury\nD) Mars\nAnswer: C\n\n"
    )
    mbpp_fewshot = (
        "def add_numbers(a, b):\n    return a + b\n\n"
        "def multiply_numbers(a, b):\n    return a * b\n\n"
    )
    humaneval_fewshot = (
        "def add(a, b):\n    return a + b\n\n"
        "def multiply(a, b):\n    return a * b\n\n"
    )
    hellaswag_fewshot = (
        "A person is walking down the street. They see a dog and\nA) pet the dog\nB) run away from the dog\nC) ignore the dog\nD) feed the dog\nAnswer: A\n\n"
        "A car is driving on the highway. The driver sees a red light and\nA) speeds up\nB) slows down and stops\nC) changes lanes\nD) honks the horn\nAnswer: B\n\n"
    )
    ai2_reasoning_fewshot = (
        "Q: If a train travels 60 miles in 2 hours, what is its speed?\nA) 30 mph\nB) 60 mph\nC) 120 mph\nD) 15 mph\nAnswer: A\n\n"
        "Q: A store sells shirts for $20 each. If you buy 3 shirts, how much do you pay?\nA) $40\nB) $60\nC) $80\nD) $100\nAnswer: B\n\n"
    )

    dataset_configs = {
        # Low rank
        "aqua_rat": {
            "hf_name": ("aqua_rat",),
            "load_kwargs": {"split": f"train[:{num_samples*5}]"},
            "format_fn": lambda sample: (
                aqua_rat_fewshot
                + f"Q: {sample['question']}\nOptions: {' '.join(sample['options'])}\nAnswer: "
                "(Please answer with the letter of the correct option, e.g., 'A', 'B', etc.)\n"
            ),
        },
        "gsm8k": {
            "hf_name": ("gsm8k", "main"),
            "load_kwargs": {"split": f"train[:{num_samples*5}]"},
            "format_fn": lambda sample: (
                gsm8k_fewshot + f"Q: {sample['question']}\nAnswer: "
                "(Please show your step-by-step reasoning and end with the answer after '####', e.g., '#### 42')\n"
            ),
        },
        "wikitext2": {
            "hf_name": ("wikitext", "wikitext-2-raw-v1"),
            "load_kwargs": {"split": f"train[:{num_samples*5}]"},
            "format_fn": lambda sample: sample["text"][:200],
        },
        "humaneval": {
            "hf_name": ("openai_humaneval",),
            "load_kwargs": {"split": f"test[:{num_samples*5}]"},
            "format_fn": lambda sample: (
                humaneval_fewshot + f"{sample['prompt']}\n"
                "(Please complete the function implementation)\n"
            ),
        },
        # Medium rank
        "sst2": {
            "hf_name": ("glue", "sst2"),
            "load_kwargs": {"split": "train[:100]"},
            "format_fn": lambda sample: sample["sentence"],
        },
        # High rank
        "reddit": {
            "hf_name": ("reddit",),
            "load_kwargs": {
                "split": f"train[:{num_samples*5}]",
                "trust_remote_code": True,
            },
            "format_fn": lambda sample: sample["body"],
        },
        # "hellaswag": {
        #     "hf_name": ("hellaswag",),
        #     "load_kwargs": {"split": f"train[:{num_samples*5}]"},
        #     "format_fn": lambda sample: (
        #         hellaswag_fewshot
        #         + f"{sample['ctx']} {sample['endings'][0]}\nA) {sample['endings'][0]}\nB) {sample['endings'][1]}\nC) {sample['endings'][2]}\nD) {sample['endings'][3]}\nAnswer: "
        #         "(Please answer with the letter of the correct option, e.g., 'A', 'B', etc.)\n"
        #     ),
        # },
        # "ai2_reasoning": {
        #     "hf_name": ("ai2_arc", "ARC-Challenge"),
        #     "load_kwargs": {"split": f"train[:{num_samples*5}]"},
        #     "format_fn": lambda sample: (
        #         ai2_reasoning_fewshot
        #         + f"Q: {sample['question']}\nA) {sample['choices']['text'][0]}\nB) {sample['choices']['text'][1]}\nC) {sample['choices']['text'][2]}\nD) {sample['choices']['text'][3]}\nAnswer: "
        #         "(Please answer with the letter of the correct option, e.g., 'A', 'B', etc.)\n"
        #     ),
        # },
        # "csqa": {
        #     "hf_name": ("commonsense_qa",),
        #     "load_kwargs": {"split": f"train[:{num_samples*5}]"},
        #     "format_fn": lambda sample: (
        #         csqa_fewshot
        #         + f"Q: {sample['question']}\nA) {sample['choices']['text'][0]}\nB) {sample['choices']['text'][1]}\nC) {sample['choices']['text'][2]}\nD) {sample['choices']['text'][3]}\nE) {sample['choices']['text'][4]}\nAnswer: "
        #         "(Please answer with the letter of the correct option, e.g., 'A', 'B', etc.)\n"
        #     ),
        # },
        # "mbpp": {
        #     "hf_name": ("mbpp",),
        #     "load_kwargs": {"split": f"train[:{num_samples*5}]"},
        #     "format_fn": lambda sample: (
        #         mbpp_fewshot + f"def {sample['entry_point']}({sample['prompt']}):\n"
        #         "(Please complete the function implementation)\n"
        #     ),
        # },
    }

    samples = {}
    for name, cfg in dataset_configs.items():
        ds = load_dataset(*cfg["hf_name"], **cfg["load_kwargs"])
        samples[name] = [cfg["format_fn"](sample) for sample in ds][:num_samples]
        if debug:
            print(f"=== {name} ===")
            for i, sample in enumerate(samples[name]):
                print(f"{i+1}. {sample[:50]}...{sample[-50:]} (len={len(sample)})")
    if debug:
        print("=== Done ===")
    return samples


def matrix_rank_from_spectrum(
    spectrum, matrix_shape: Tuple[int, int], eps=torch.finfo(torch.float64).eps
):
    """Compute matrix rank from spectrum.

    Args:
        spectrum (torch.Tensor): Spectrum of the matrix.
        matrix_shape (tuple): Shape of the matrix.
        eps (float): Tolerance.
    """
    thresh = spectrum.max() * max(matrix_shape) * eps
    rank = (spectrum > thresh).sum().item()
    return rank


def num_zero_rows_cols(matrix: torch.Tensor, eps=torch.finfo(torch.float64).eps):
    """Compute number of zero rows and columns in a matrix."""
    num_zero_rows = (matrix < 10 * eps).all(dim=0).sum().item()
    num_zero_cols = (matrix < 10 * eps).all(dim=1).sum().item()
    return num_zero_rows, num_zero_cols


def matrix_row_col_hist(matrix: torch.Tensor, eps=torch.finfo(torch.float32).eps):
    """Compute histogram of row and column sums in a matrix."""
    ys = matrix.max(dim=0).values.cpu()
    xs = matrix.max(dim=1).values.cpu()
    return ys, xs


def get_joint_prob(model, tokenizer, text, device, top_k=None):
    """Compute joint probability p(y1, y2 | x) for given text. If top_k is None, use all vocab."""
    x = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)[
        "input_ids"
    ].to(device)

    with torch.no_grad():
        out = model(x)
        logits_y1 = out.logits[0, -1, :]  # (vocab_size,)
        p_y1 = torch.softmax(logits_y1, dim=-1)  # p(Y1)

        if top_k is None:
            topk_p_y1 = p_y1
            topk_y1 = torch.arange(p_y1.size(0), device=device)
        else:
            topk_p_y1, topk_y1 = torch.topk(p_y1, top_k)  # (top_k,)
        joint = torch.zeros(
            (len(topk_y1), len(topk_y1)), device=device, dtype=torch.float64
        )

        for i, y1 in enumerate(
            tqdm(topk_y1, desc=f"Computing joint p(y1,y2|x) for k={top_k}", leave=False)
        ):
            x_y1 = torch.cat([x[0], y1.unsqueeze(0)]).unsqueeze(0)  # (1, L+1)
            out2 = model(x_y1)
            logits_y2 = out2.logits[0, -1, :]
            p_y2_given_y1 = torch.softmax(logits_y2, dim=-1)  # (vocab_size,)
            if top_k is None:
                topk_p_y2 = p_y2_given_y1
            else:
                topk_p_y2, _ = torch.topk(p_y2_given_y1, top_k)
            joint[i, :] = topk_p_y1[i] * topk_p_y2  # p(Y1=y1) * p(Y2|Y1=y1)

    return joint.cpu()


def plot_spectra(spectra, save_path=None):
    """Plot spectra comparison"""
    plt.figure(figsize=(10, 6))

    colors = {
        "wikitext2": "blue",
        "sst2": "orange",
        "aqua_rat": "green",
        "reddit": "red",
        "gsm8k": "purple",
        "csqa": "brown",
        "mbpp": "pink",
        "humaneval": "gray",
        "hellaswag": "cyan",
        "ai2_reasoning": "magenta",
    }
    names = {
        "wikitext2": "WikiText-2",
        "sst2": "SST-2",
        "aqua_rat": "AQuA (Math QA)",
        "reddit": "Reddit",
        "gsm8k": "GSM8K",
        "csqa": "CSQA",
        "mbpp": "MBPP",
        "humaneval": "HumanEval",
        "hellaswag": "HellaSwag",
        "ai2_reasoning": "AI2 Reasoning",
    }

    line_styles = {
        "wikitext2": "-",
        "sst2": "--",
        "aqua_rat": "-.",
        "reddit": ":",
        "gsm8k": "-",
        "csqa": "--",
        "mbpp": "-.",
        "humaneval": ":",
        "hellaswag": "-",
        "ai2_reasoning": "--",
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
    # meta-llama/Llama-2-7b-chat-hf
    # deepseek-ai/DeepSeek-R1
    # deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing progress checkpoint.",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Use random initializations for the model.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5000,
        help="Number of top tokens to consider for joint probability computation.",
    )
    args = parser.parse_args()

    model_name = args.model.replace("/", "_")

    # Define filenames
    rstr = "_rand" if args.random else ""
    progress_path = f"results/spectrum_progress_{model_name}_topk{args.top_k}{rstr}.pt"
    plot_path = f"results/spectrum_comparison_{model_name}_topk{args.top_k}{rstr}.png"

    # Progress file path
    os.makedirs(os.path.dirname(progress_path), exist_ok=True)

    # Load model
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    if args.random:
        config = AutoConfig.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_config(config).to(args.device)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)

    # Describe model (params and vocab size)
    print("=== Model Description ===")
    print(f"Model: {args.model}")
    print(f"Params: {sum(p.numel() for p in model.parameters())}")
    print(f"Size (GB): {sum(p.numel() for p in model.parameters()) * 4 / 1e9:.2f}")
    print(f"Vocab size: {len(tokenizer.get_vocab())}")
    print("=== Done ===")

    # Get samples
    print("Loading datasets...")
    datasets = get_samples(debug=True)

    # Try to load progress, unless overwrite is set
    if os.path.exists(progress_path) and not args.overwrite:
        print(f"Loading progress from {progress_path}...")
        ckpt = torch.load(progress_path, weights_only=False)
        joint_stats = {
            k: ckpt.get(
                k,
                {
                    "spectra": [],
                    "num_zero_rows": [],
                    "num_zero_cols": [],
                },
            )
            for k in datasets.keys()
        }
        print(f"Loaded spectra for categories: {list(joint_stats.keys())}")
    else:
        if os.path.exists(progress_path) and args.overwrite:
            print(f"Overwrite flag set. Removing existing progress at {progress_path}.")
            os.remove(progress_path)
        joint_stats = {
            k: {
                "spectra": [],
                "num_zero_rows": [],
                "num_zero_cols": [],
                "max_ys": [],
                "max_xs": [],
            }
            for k in datasets.keys()
        }

    # Compute and save joint stats for all datasets
    total_samples = sum(len(samples) for samples in datasets.values())
    pbar = tqdm(total=total_samples, desc="Computing spectra")
    for category, samples in datasets.items():
        pbar.set_postfix({"category": category})
        for i, text in enumerate(samples):
            if len(joint_stats[category]["spectra"]) <= i:  # skip if already computed
                try:
                    # ompute joint
                    p_y1y2 = get_joint_prob(
                        model, tokenizer, text, args.device, args.top_k
                    )

                    # compute spectrum
                    _, spectrum, _ = randomized_svd(
                        p_y1y2.cpu().numpy(),
                        n_components=min(p_y1y2.shape),
                        random_state=42,
                    )

                    # compute sparsity
                    num_zero_rows, num_zero_cols = num_zero_rows_cols(p_y1y2)
                    max_ys, max_xs = matrix_row_col_hist(p_y1y2)

                    # save progress
                    joint_stats[category]["spectra"].append(torch.tensor(spectrum))
                    joint_stats[category]["num_zero_rows"].append(num_zero_rows)
                    joint_stats[category]["num_zero_cols"].append(num_zero_cols)
                    joint_stats[category]["max_ys"].append(max_ys)
                    joint_stats[category]["max_xs"].append(max_xs)
                    torch.save(joint_stats, progress_path)
                except Exception as e:
                    print(f"Error with {category}: {e}")
            pbar.update(1)
    pbar.close()

    # Plot spectra
    print(f"Plotting {model_name} {rstr}...")
    print(f"Saving to {plot_path}...")
    spectra = {k: v["spectra"] for k, v in joint_stats.items()}
    plot_spectra(spectra, plot_path)

    # Compute summary statistics
    summary_rows = []
    var_target = 0.99
    for category, j_stat in joint_stats.items():
        if j_stat:
            ranks = []
            energies = []
            for spectrum in j_stat["spectra"]:
                spectrum = spectrum.cpu()
                # rank
                rank = matrix_rank_from_spectrum(spectrum, (args.top_k, args.top_k))
                ranks.append(rank)

                # energy
                cumsum = torch.cumsum(spectrum**2, 0)
                total = (spectrum**2).sum()
                energy = ((cumsum / total) < var_target).sum().item() + 1
                energies.append(energy)

            summary_rows.append(
                [
                    category,
                    f"{np.mean(ranks):.1f} ± {np.std(ranks):.1f}",
                    f"{np.mean(energies):.1f} ± {np.std(energies):.1f}",
                    f"{np.mean(j_stat['num_zero_rows']):.1f} ± {np.std(j_stat['num_zero_rows']):.1f}",
                    f"{np.mean(j_stat['num_zero_cols']):.1f} ± {np.std(j_stat['num_zero_cols']):.1f}",
                ]
            )
    # Sort by matrix rank (extract mean rank from the formatted string)
    summary_rows.sort(key=lambda row: float(row[1].split(" ± ")[0]))

    print(
        tabulate(
            summary_rows,
            headers=[
                f"Category",
                f"Matrix Rank (matlab threshold)",
                f"Spectral Energy (99%)",
                f"Zero Rows",
                f"Zero Cols",
            ],
            tablefmt="github",
        )
    )


if __name__ == "__main__":
    main()


# Plotting deepseek-ai_DeepSeek-R1-0528-Qwen3-8B ...
# Saving to results/spectrum_comparison_deepseek-ai_DeepSeek-R1-0528-Qwen3-8B_topk5000.png...
# | Category   | Matrix Rank (matlab threshold)   | Spectral Energy (99%)   | Zero Rows   | Zero Cols   |
# |------------|----------------------------------|-------------------------|-------------|-------------|
# | gsm8k      | 137.0 ± 0.0                      | 1.0 ± 0.0               | 0.0 ± 0.0   | 0.0 ± 0.0   |
# | aqua_rat   | 200.0 ± 0.0                      | 1.0 ± 0.0               | 0.0 ± 0.0   | 0.0 ± 0.0   |
# | humaneval  | 537.6 ± 771.3                    | 1.2 ± 0.4               | 0.0 ± 0.0   | 0.0 ± 0.0   |
# | reddit     | 1519.2 ± 747.3                   | 1.6 ± 0.5               | 0.0 ± 0.0   | 0.0 ± 0.0   |
# | wikitext2  | 1960.3 ± 664.0                   | 1.3 ± 0.5               | 0.0 ± 0.0   | 0.0 ± 0.0   |
# | sst2       | 3290.2 ± 663.9                   | 2.2 ± 0.4               | 0.0 ± 0.0   | 0.0 ± 0.0   |

# Saving to results/spectrum_comparison_meta-llama_Llama-2-7b-chat-hf_topk5000.png...
# | Category   | Matrix Rank (matlab threshold)   | Spectral Energy (99%)   | Zero Rows   | Zero Cols   |
# |------------|----------------------------------|-------------------------|-------------|-------------|
# | gsm8k      | 241.0 ± 0.0                      | 1.0 ± 0.0               | 0.0 ± 0.0   | 0.0 ± 0.0   |
# | humaneval  | 404.6 ± 473.7                    | 1.0 ± 0.0               | 0.0 ± 0.0   | 0.0 ± 0.0   |
# | aqua_rat   | 951.0 ± 0.0                      | 1.0 ± 0.0               | 0.0 ± 0.0   | 0.0 ± 0.0   |
# | reddit     | 2079.6 ± 778.0                   | 1.6 ± 0.5               | 0.0 ± 0.0   | 0.0 ± 0.0   |
# | sst2       | 3400.6 ± 612.1                   | 1.4 ± 0.5               | 0.0 ± 0.0   | 0.0 ± 0.0   |
# | wikitext2  | 3828.6 ± 952.6                   | 2.2 ± 0.7               | 0.0 ± 0.0   | 0.0 ± 0.0   |


# Plotting meta-llama_Llama-2-7b-chat-hf _rand...
# Saving to results/spectrum_comparison_meta-llama_Llama-2-7b-chat-hf_topk5000_rand.png...
# | Category   | Matrix Rank (matlab threshold)   | Spectral Energy (99%)   |
# |------------|----------------------------------|-------------------------|
# | aqua_rat   | 1295.0 ± 0.0                     | 26.0 ± 0.0              |
# | gsm8k      | 1311.0 ± 0.0                     | 27.0 ± 0.0              |
# | sst2       | 1322.6 ± 22.1                    | 27.2 ± 1.5              |
# | wikitext2  | 1338.0 ± 17.9                    | 26.8 ± 1.5              |
# | reddit     | 1339.6 ± 9.6                     | 25.4 ± 1.0              |
