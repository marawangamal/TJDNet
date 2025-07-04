"""
Minimal script to analyze eigen spectrum of p(y1, y2|x) for datasets with varying rank complexity.

Usage:
    python scripts/plots/plot_dataset_rank_simple.py --model gpt2

Prompting Instructions and Few-Shot Examples for Datasets
---------------------------------------------------------

AQuA-RAT (Algebra Question Answering):
--------------------------------------
- Each sample is a multiple-choice math word problem.
- Prompt format:
    [QUESTION]\nOptions: [A) ... B) ... C) ... D) ... E) ...]\nAnswer:
- For few-shot learning, provide several Q&A pairs in the same format.
- Example:

Q: Two friends plan to walk along a 43-km trail, starting at opposite ends of the trail at the same time. If Friend P's rate is 15% faster than Friend Q's, how many kilometers will Friend P have walked when they pass each other?
Options: A)21 B)21.5 C)22 D)22.5 E)23
Answer: E

Q: In the coordinate plane, points (x, 1) and (5, y) are on line k. If line k passes through the origin and has slope 1/5, then what are the values of x and y respectively?
Options: A)4 and 1 B)1 and 5 C)5 and 1 D)3 and 5 E)5 and 3
Answer: C

- For LLMs, you can concatenate several such Q&A pairs for few-shot prompting.

GSM8K (Grade School Math 8K):
-----------------------------
- Each sample is a free-form grade school math word problem.
- Prompt format:
    [QUESTION]\nAnswer:
- For few-shot learning, provide several Q&A pairs in the same format.
- Example:

Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Answer: Natalia sold 48/2 = 24 clips in May. Natalia sold 48+24 = 72 clips altogether in April and May. #### 72

Q: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer: Weng earns 12/60 = $0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $10. #### 10

- For LLMs, you can concatenate several such Q&A pairs for few-shot prompting.

MATH (Mathematics Dataset):
--------------------------
- Each sample is a high school mathematics problem with step-by-step solution.
- Prompt format:
    [PROBLEM]\nSolution:
- For few-shot learning, provide several problem-solution pairs in the same format.
- Example:

Problem: Find the value of x in the equation 2x + 5 = 13.
Solution: To solve for x, we need to isolate it on one side of the equation.
2x + 5 = 13
Subtract 5 from both sides: 2x = 8
Divide both sides by 2: x = 4
The value of x is 4.

Problem: What is the area of a circle with radius 3?
Solution: The area of a circle is given by the formula A = πr².
Given r = 3, we have A = π(3)² = π(9) = 9π.
The area of the circle is 9π square units.

- For LLMs, you can concatenate several such problem-solution pairs for few-shot prompting.

General Tips:
-------------
- For all datasets, few-shot learning is typically done by concatenating several Q&A pairs in the prompt, followed by a new question for the model to answer.
- For AQuA-RAT, always include the options and ask for the answer letter.
- For GSM8K, show the step-by-step reasoning and end with the answer after '####'.
- For MATH, provide detailed step-by-step solutions with clear explanations.
"""

import os
import argparse
from tqdm import tqdm
from tabulate import tabulate

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset


# TODO: Add CSQA, STEMP, and other low-rank datasets.
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
def get_samples(debug=False, num_samples=25):
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
    math_fewshot = (
        "Problem: Find the value of x in the equation 2x + 5 = 13.\n"
        "Solution: To solve for x, we need to isolate it on one side of the equation.\n"
        "2x + 5 = 13\n"
        "Subtract 5 from both sides: 2x = 8\n"
        "Divide both sides by 2: x = 4\n"
        "The value of x is 4.\n\n"
        "Problem: What is the area of a circle with radius 3?\n"
        "Solution: The area of a circle is given by the formula A = πr².\n"
        "Given r = 3, we have A = π(3)² = π(9) = 9π.\n"
        "The area of the circle is 9π square units.\n\n"
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
        "math": {
            "hf_name": ("hendrycks_math",),
            "load_kwargs": {"split": f"train[:{num_samples*5}]"},
            "format_fn": lambda sample: (
                math_fewshot + f"Problem: {sample['problem']}\nSolution: "
                "(Please provide a detailed step-by-step solution)\n"
            ),
        },
        "wikitext2": {
            "hf_name": ("wikitext", "wikitext-2-raw-v1"),
            "load_kwargs": {"split": f"train[:{num_samples*5}]"},
            "format_fn": lambda sample: sample["text"][:200],
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
        "gsm8k": "purple",
    }
    names = {
        "wikitext2": "WikiText-2",
        "sst2": "SST-2",
        "aqua_rat": "AQuA (Math QA)",
        "reddit": "Reddit",
        "gsm8k": "GSM8K",
    }

    line_styles = {
        "wikitext2": "-",
        "sst2": "--",
        "aqua_rat": "-.",
        "reddit": ":",
        "gsm8k": "-",
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
        spectra = {k: [] for k in datasets.keys()}

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
                # === DEBUG >>>>
                # p_y1y2 = torch.randn(5000, 5000)
                # p_y1y2 = torch.nn.functional.softmax(p_y1y2.reshape(-1)).reshape(
                #     5000, 5000
                # )
                # === DEBUG <<<<
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
    print(f"Plotting {model_name} {rstr}...")
    print(f"Saving to {plot_path}...")
    plot_spectra(spectra, plot_path)
    max_fp = torch.finfo(torch.float32).max

    summary_rows = []
    var_target = 0.99
    for category, spectrum_list in spectra.items():
        if spectrum_list:
            ranks = []
            non_zero_counts = []
            for spectrum in spectrum_list:
                spectrum = spectrum.cpu()
                non_zero_counts.append((spectrum > 1e-10).sum().item())
                cumsum = torch.cumsum(spectrum**2, 0)
                total = (spectrum**2).sum()
                rank = ((cumsum / total) < var_target).sum().item() + 1
                ranks.append(rank)
            summary_rows.append(
                [
                    category,
                    f"{np.mean(ranks):.1f} ± {np.std(ranks):.1f}",
                    f"{np.mean(non_zero_counts):.1f} ± {np.std(non_zero_counts):.1f}",
                ]
            )
    print(
        tabulate(
            summary_rows,
            headers=[
                f"Category",
                f"Emprical Rank for {var_target*100}% Variance (mean ± std)",
                f"Non-zero Eigenvalues (mean ± std)",
            ],
            tablefmt="github",
        )
    )


if __name__ == "__main__":
    main()


# Random matrix 5000x5000
# | Category   | Emprical Rank for 99.0% Variance (mean ± std)   |
# |------------|-------------------------------------------------|
# | aqua_rat   | 3657.6 ± 0.5                                    |
# | gsm8k      | 3657.6 ± 0.5                                    |
# | wikitext2  | 3657.2 ± 0.4                                    |
# | sst2       | 3657.8 ± 1.0                                    |
# | reddit     | 3657.4 ± 0.5                                    |

# meta-llama_Llama-2-7b-chat-hf
# | Category   | 99.0% Variance (mean ± std)                     | Non-zero Eigenvalues (mean ± std)   |
# |------------|-------------------------------------------------|-------------------------------------|
# | aqua_rat   | 1.0 ± 0.0                                       | 157.0 ± 0.0                         |
# | gsm8k      | 1.0 ± 0.0                                       | 64.0 ± 0.0                          |
# | wikitext2  | 2.2 ± 0.7                                       | 1071.2 ± 588.8                      |
# | sst2       | 1.4 ± 0.5                                       | 687.4 ± 105.4                       |
# | reddit     | 1.6 ± 0.5                                       | 386.6 ± 120.1                       |


# meta-llama_Llama-2-7b-chat-hf (random init)
# | Category   | 99.0% Variance (mean ± std)                     | Non-zero Eigenvalues (mean ± std)   |
# |------------|-------------------------------------------------|-------------------------------------|
# | aqua_rat   | 24.0 ± 0.0                                      | 1303.0 ± 0.0                        |
# | gsm8k      | 28.0 ± 0.0                                      | 1335.0 ± 0.0                        |
# | wikitext2  | 28.0 ± 0.6                                      | 1340.2 ± 21.0                       |
# | sst2       | 25.4 ± 1.4                                      | 1326.2 ± 9.7                        |
# | reddit     | 26.6 ± 0.5                                      | 1330.6 ± 14.1                       |


# | Dataset   | Rank (pretrained) | Rank (random) |
# |-----------|-------------------|---------------|
# | gsm8k     | 64.0 ± 0.0       | 1335.0 ± 0.0  |
# | aqua_rat  | 157.0 ± 0.0      | 1303.0 ± 0.0  |
# | reddit    | 386.6 ± 120.1    | 1330.6 ± 14.1 |
# | sst2      | 687.4 ± 105.4    | 1326.2 ± 9.7  |
# | wikitext2 | 1071.2 ± 588.8   | 1340.2 ± 21.0 |
