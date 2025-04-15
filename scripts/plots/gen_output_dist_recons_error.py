"""Plots reconstruction error of language model output distributions using tensor completion.

Example:
    python gen_output_dist_recons_error.py --model meta-llama/Llama-2-7b-chat-hf
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

from dataloaders.gsm8k import ChatTemplateGSM8k
from dataloaders.sharegpt import ChatTemplateShareGPT

PROMPTS = [
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
        "value": ChatTemplateGSM8k.get_sample_prompt_few_shot(),
    },
    {
        "name": "sharegpt",
        "value": ChatTemplateShareGPT.get_sample_prompt_few_shot(),
    },
]


def generate_dataset(
    model: torch.nn.Module,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    horizon: int = 4,
    start_str: str = "\n",
    checkpoint_steps: int = 5,  # Save progress every n tokens
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 16,
    resume: bool = True,
    checkpoint_path: str = "results/output_mat_batched_checkpoint.pt",
) -> torch.utils.data.TensorDataset:
    """Generates dataset for tensor completion. (i.e., samples from T where T is of shape V^H

    Args:
        model (torch.nn.Module): Model used to generate the dataset.
        tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): Tokenizer for the model.
        horizon (int, optional): Number of tokens to generate. Defaults to 4.
        start_str (str, optional): Defaults to "\n".
        checkpoint_steps (int, optional): Defaults to 5.
        batch_size (int, optional): Defaults to 16.
        resume (bool, optional): Defaults to True.
        checkpoint_path (str, optional): Defaults to "results/output_mat_batched_checkpoint.pt".

    Returns:
        torch.utils.data.TensorDataset: Dataset tuple (X, y). Tensor X is of shape (batch_size, horizon) each value is an integer in [0, V).
            Tensor y is of shape (batch_size,) and each value is a real number, reflecting the measured probability. (i.e., y[0] = P(X[0]))
    """
    raise NotImplementedError(
        "This function should be implemented to generate the dataset for the model."
    )


def plot_errors(spectrums, save_path=None) -> None:
    raise NotImplementedError(
        ""
        "This function should be implemented to plot the error curves for the spectrum."
    )


def get_tensor_completion_error(
    dataset: torch.utils.data.TensorDataset,
) -> torch.Tensor:
    """_summary_

    Args:
        dataset (torch.utils.data.TensorDataset): Tensor dataset containing the model outputs.

    Returns:
        torch.Tensor: Tensor containing the tensor completion error.
    """
    raise NotImplementedError(
        "This function should be implemented to compute the tensor completion error."
    )


def main(args: Namespace):
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    datasets = {}

    for prompt in PROMPTS:
        print(f"[{prompt['name']}] Generating output distribution dataset...")
        dataset = generate_dataset(
            model,
            tokenizer,
            start_str=prompt["value"],
            # Caches the generated dataset to avoid re-generating it
            checkpoint_path=f"results/output_tcds_{prompt['name']}.pt",
        )

        print(f"[{prompt['name']}]Computing reconstruction error...")
        error = get_tensor_completion_error(dataset)
        datasets[prompt["name"]] = error

        print(f"[{prompt['name']}] Plotting error curves...")
        os.makedirs("results", exist_ok=True)

    plot_errors(
        datasets,
        save_path=f"results/plots/tensor_completion_plot_{args.model.split('/')[-1]}.png",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze the output distribution of a language model.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="gpt2",  # meta-llama/Llama-2-7b-chat-hf
        help="Hugging Face model identifier (default: gpt2)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=4,
    )
    args = parser.parse_args()
    main(args)
