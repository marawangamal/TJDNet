"""Generates a spectrum plot of the output distribution for a language model.

Example:
    python gen_output_dist_spectrum.py --model meta-llama/Llama-2-7b-chat-hf
"""

from argparse import Namespace
import argparse
import os
import os.path as osp
from typing import Union
from tqdm import tqdm

import torch
from sklearn.utils.extmath import randomized_svd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

import numpy as np
import matplotlib.pyplot as plt

from dataloaders.gsm8k import ChatTemplateGSM8k
from dataloaders.sharegpt import ChatTemplateShareGPT
from utils.utils import group_arr, plot_groups, replace_spec_chars


# Answers retrieved using `scripts/chat.py`
PROMPTS = [
    {
        "name": "newline",
        "question": "\n",
        "answer": "",  # answer recieved from the model
    },
    {"name": "space", "question": " ", "answer": ""},  # answer recieved from the model
    {
        "name": "poem",
        "question": "Write a poem.",
        "answer": """Moonlit Serenade
                    The night sky is painted with hues of blue,
                    A gentle breeze whispers secrets anew.
                    The stars shine bright, like diamonds in the air,
                    As I sit on the porch, with my guitar there.

                    The moon, a glowing orb of white,
                    Casts a silver light, on this peaceful night.
                    The world is hushed, in a quiet sleep,
                    As I strum the chords, my heart doth keep.

                    The music flows, like a river's stream,
                    As I sing of love, of dreams, of hope, of scheme.
                    The notes dance, like fireflies in flight,
                    As I weave a tale, of love's sweet delight.

                    The world may be busy, with its noise and fray,
                    But in this moment, all is calm, all is gray.
                    The music washes over me, like a soothing balm,
                    And I am free, my heart, my soul, my all.

                    The night air whispers secrets in my ear,
                    Of the beauty of the world, of love, of fear.
                    The stars shine bright, like diamonds in the sky,
                    As I play on, my heart, my soul, on high.

                    So let the music flow, let the notes be free,
                    For in this moment, that's all that matters to me.
                """,  # answer recieved from the model
    },
    {
        "name": "gsm8k",
        "question": ChatTemplateGSM8k.TEMPLATE_FEW_SHOT.format(
            question="Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
            answer="",
        ),
        "answer": "",  # answer recieved from the model
    },
    {
        "name": "sharegpt",
        "question": ChatTemplateShareGPT.get_sample_prompt(is_few_shot=True),
        "answer": "",  # answer recieved from the model
    },
]

# We create the `PROMPTS_WINDOWED` list from the `PROMPTS` list by sliding over the answer

PROMPTS_WINDOWED = [
    {
        "name": "newline",
        "value": "\n",
    },
    {
        "name": "space",
        "value": " ",
    },
    {
        "name": "poem",
        "value": "Write a poem.",
    },
    {
        "name": "gsm8k",
        "value": ChatTemplateGSM8k.TEMPLATE_FEW_SHOT.format(
            question="Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
            answer="",
        ),
    },
    {
        "name": "sharegpt",
        "value": ChatTemplateShareGPT.get_sample_prompt(is_few_shot=True),
    },
]


def get_spectrum(output_mat: torch.Tensor) -> torch.Tensor:
    """Get spectrum of 2D matrix by computing singular values

    Args:
        output_mat (torch.Tensor): 2D matrix. Shape: (vocab_size, vocab_size)

    Returns:
        torch.Tensor: Singular values of the matrix. Shape: (vocab_size,)
    """
    spectrum_tests = [
        {
            "test": lambda x: torch.isnan(x).any(),
            "error": "Matrix contains NaN values",
        },
        {
            "test": lambda x: torch.isinf(x).any(),
            "error": "Matrix contains infinite values",
        },
        {
            "test": lambda x: x.ndim != 2,
            "error": "Matrix must be 2D",
        },
    ]

    for test in spectrum_tests:
        if test["test"](output_mat):
            raise ValueError(test["error"])

    # Print min/max values for debugging
    print(f"Min value: {output_mat.min()}")
    print(f"Max value: {output_mat.max()}")
    # _, s, _ = torch.linalg.svd(output_mat, full_matrices=False)
    U, s, Vh = randomized_svd(
        output_mat.detach().cpu().numpy(), n_components=1000, random_state=0
    )
    return torch.tensor(s)


def plot_spectrum(spectrums, save_path=None):
    """Plot the spectrum for multiple prompts and save the figure if a path is provided

    Args:
        spectrums: Dictionary mapping prompt names to their spectrum values
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(10, 6))

    # Plot each spectrum with a different color/marker
    for name, spectrum in spectrums.items():
        # Convert to numpy for plotting
        if torch.is_tensor(spectrum):
            spectrum_np = spectrum.cpu().numpy()
        else:
            spectrum_np = spectrum

        # Plot singular values
        plt.semilogy(np.arange(1, len(spectrum_np) + 1), spectrum_np, "o-", label=name)

        # Calculate energy metrics for annotation
        total_energy = spectrum_np.sum()
        energy_90 = np.searchsorted(np.cumsum(spectrum_np) / total_energy, 0.9) + 1

        # Annotate the 90% energy point
        plt.annotate(
            f"{name}: 90% energy at k={energy_90}",
            xy=(energy_90, spectrum_np[energy_90 - 1]),
            xytext=(10, 10),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
        )

    # Add grid and labels
    plt.grid(True, which="both", ls="--")
    plt.xlabel("Index")
    plt.ylabel("Singular Value (log scale)")
    plt.title("Spectrum of Output Distribution Matrix")

    # Add a horizontal line at y=1 for reference
    plt.axhline(y=1, color="r", linestyle="-", alpha=0.3)

    # Add legend
    plt.legend(loc="best")

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

    # Use 'total' and 'initial' in tqdm, then manually call pbar.update(1)
    with tqdm(
        total=vocab_size,
        initial=start_idx,
        desc="Processing tokens",
        unit="token",
        leave=False,
        dynamic_ncols=True,
        smoothing=0.1,
        colour="green",
    ) as pbar:
        for i in range(start_idx, vocab_size):
            with torch.no_grad():
                outputs = model(
                    torch.cat(
                        [input_ids, torch.tensor([i]).to(device).reshape(1, 1)], dim=-1
                    )
                )
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits[0, -1])
            output_mat[i] = probs[0]

            # Periodically save checkpoint
            if (i + 1) % checkpoint_steps == 0 or (i + 1) == vocab_size:
                torch.save((output_mat, i + 1), checkpoint_path)
                if (i + 1) < vocab_size:
                    pbar.set_postfix_str(f"Checkpoint saved at index {i+1}")

            # Manually update the tqdm bar by 1 step
            pbar.update(1)

    return output_mat


def generate_output_distribution_spectrum_batched(
    model: torch.nn.Module,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    start_str: str = "\n",
    checkpoint_steps: int = 5,  # Save progress every n tokens
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 16,
    resume: bool = True,
    checkpoint_path: str = "results/output_mat_batched_checkpoint.pt",
):
    """Batched version of generate_output_distribution_spectrum"""
    # Ensure 'results' directory exists for checkpoint
    os.makedirs("results", exist_ok=True)

    # Initialize or resume
    vocab_size = len(tokenizer.get_vocab())
    p_y1_y2 = torch.zeros((vocab_size, vocab_size))
    start_idx = 0

    if os.path.exists(checkpoint_path) and resume:
        # If there's a checkpoint, load it
        saved_mat, saved_idx = torch.load(checkpoint_path)
        # Copy the saved matrix into our newly allocated matrix in case sizes match
        p_y1_y2[: saved_mat.shape[0], : saved_mat.shape[1]] = saved_mat
        start_idx = saved_idx
        print(f"Resuming from {checkpoint_path} at token index {start_idx}.")
    else:
        print(f"Starting fresh at token index {start_idx}.")

    x = torch.tensor(tokenizer.encode(start_str, return_tensors="pt")).to(
        device
    )  # Shape: (1, seq_len)
    model.to(device)

    # Use 'total' and 'initial' in tqdm, then manually call pbar.update(1)
    with tqdm(
        total=vocab_size,
        initial=start_idx,
        desc="Processing tokens",
        unit="token",
        leave=False,
        dynamic_ncols=True,
        smoothing=0.1,
        colour="green",
    ) as pbar:
        for i in range(start_idx, vocab_size, batch_size):
            with torch.no_grad():
                y1 = (
                    torch.arange(i, min(i + batch_size, vocab_size))
                    .reshape(-1, 1)
                    .to(device)
                )  # Shape: (batch_size, 1)
                outputs = model(torch.cat([x.repeat(y1.size(0), 1), y1], dim=-1))
            logits = outputs.logits  # Shape: (batch_size, 2, vocab_size)
            p_y1 = torch.nn.functional.softmax(
                logits[:, -2], dim=-1
            )  # Shape: (batch_size, vocab_size)
            p_y2_g_y1 = torch.nn.functional.softmax(
                logits[:, -1], dim=-1
            )  # Shape: (batch_size, vocab_size)
            p_y1_y2[i : min(i + batch_size, p_y1_y2.size(1))] = p_y2_g_y1 * p_y1

            # Periodically save checkpoint
            if i % (checkpoint_steps * batch_size) == 0:
                torch.save((p_y1_y2, i), checkpoint_path)
                pbar.set_postfix_str(f"Checkpoint saved at {i}")

            # Manually update the tqdm bar by 1 step
            pbar.update(batch_size)

            # if i > 100:
            #     break

    return p_y1_y2


def main(args: Namespace):
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    save_dir = f"results/plot_output_dist_spectrum/{replace_spec_chars(args.model)}"
    os.makedirs(save_dir, exist_ok=True)

    if args.sample:
        # === DEBUG =============================================================
        # Sample from the output distribution
        print("Sampling from the output distribution...")
        output = model.generate(
            torch.tensor(
                tokenizer.encode(ChatTemplateGSM8k.get_sample_prompt(is_few_shot=True))
            ).unsqueeze(0),
            max_new_tokens=32,
        )
        print(tokenizer.decode(output[0], skip_special_tokens=True))

    spectrums = []

    for prompt in PROMPTS_WINDOWED:
        # Generate output distribution spectrum (resuming or starting fresh)
        print(f"[{prompt['name']}] Generating output distribution spectrum...")
        output_mat = generate_output_distribution_spectrum_batched(
            model,
            tokenizer,
            start_str=prompt["value"],
            checkpoint_path=osp.join(
                save_dir, f"output_dist_2d_{replace_spec_chars(prompt['name'])}.pt"
            ),
        )

        print(f"[{prompt['name']}]Computing spectrum...")
        spectrum = get_spectrum(output_mat)
        # spectrums[prompt["name"]] = spectrum
        spectrums.extend(
            [
                {
                    "prompt": prompt["name"],
                    "index": i,
                    "spectrum": s,
                }
                for i, s in enumerate(spectrum)
            ]
        )

        print(f"[{prompt['name']}] Plotting spectrum...")
        os.makedirs("results", exist_ok=True)

        # Print some statistics
        print(f"[{prompt['name']}] Top 10 singular values: {spectrum[:10]}")
        print(f"[{prompt['name']}] Sum of all singular values: {spectrum.sum()}")
        print(
            f"[{prompt['name']}] Effective rank (singular values > 1e-10): {(spectrum > 1e-10).sum()}"
        )

    grouped = group_arr(
        spectrums,
        # prompt dataset
        lambda x: x["prompt"].split("::")[0],
        # prompt index
        lambda x: x["prompt"].split("::")[1] if "::" in x["prompt"] else "0",
    )

    plot_groups(
        grouped,
        x_key="index",
        y_key="spectrum",
        path=osp.join(save_dir, f"output_dist_2d_spectrum.png"),
        # First level controls color, second controls marker
        axes_kwargs={
            "title": f"Spectrum of Matrix P(y1,y2|x)",
            "xlabel": "Index",
            "ylabel": "Singular Value (log scale)",
        },
        fig_kwargs={
            "figsize": (10, 6),
            "dpi": 300,
        },
        style_dims=[
            "color",
            "marker",
            "linestyle",
        ],
        style_cycles={
            "color": [
                "#0173B2",
                "#DE8F05",
                "#029E73",
                "#D55E00",
                "#CC78BC",
                "#CA9161",
                "#FBAFE4",
                "#949494",
            ],
            "marker": ["o", "s", "D", "X", "^", "v"],
            "linestyle": ["-", "--", "-.", ":"],
        },
    )

    # plot_spectrum(
    #     spectrums,
    #     save_path=osp.join(save_dir, f"output_dist_2d_spectrum.png"),
    # )

    #     # Add grid and labels
    # plt.grid(True, which="both", ls="--")
    # plt.xlabel("Index")
    # plt.ylabel("Singular Value (log scale)")
    # plt.title("Spectrum of Output Distribution Matrix")

    # # Add a horizontal line at y=1 for reference
    # plt.axhline(y=1, color="r", linestyle="-", alpha=0.3)

    # # Add legend
    # plt.legend(loc="best")

    # plt.tight_layout()


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
    parser.add_argument(
        "-s",
        "--sample",
        action="store_true",
        help="Sample from the output distribution",
    )
    args = parser.parse_args()
    main(args)
