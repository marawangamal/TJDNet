"""Plots reconstruction error of language model output distributions using tensor completion.

Example:
    python gen_output_dist_recons_error.py --model meta-llama/Llama-2-7b-chat-hf
"""

from argparse import Namespace
import argparse
import datetime
import json
import os
from typing import Union

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


# def generate_dataset(
#     model: torch.nn.Module,
#     tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
#     horizon: int = 4,
#     start_str: str = "\n",
#     checkpoint_steps: int = 5,  # Save progress every n tokens
#     device: str = "cuda" if torch.cuda.is_available() else "cpu",
#     batch_size: int = 16,
#     resume: bool = True,
#     checkpoint_path: str = "results/output_mat_batched_checkpoint.pt",
#     num_samples: int = 1000,
# ):
#     """Generates dataset for tensor completion. (i.e., samples from T where T is of shape V^H

#     Args:
#         model (torch.nn.Module): Model used to generate the dataset.
#         tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): Tokenizer for the model.
#         horizon (int, optional): Number of tokens to generate. Defaults to 4.
#         start_str (str, optional): Defaults to "\n".
#         checkpoint_steps (int, optional): Defaults to 5.
#         batch_size (int, optional): Defaults to 16.
#         resume (bool, optional): Defaults to True.
#         checkpoint_path (str, optional): Defaults to "results/output_mat_batched_checkpoint.pt".

#     """

#     for i in range(0, num_samples, batch_size):
#         y = torch.randint(0, len(tokenizer), (batch_size, horizon), device=device)
#         outputs = model(y)
#         logits = outputs.logits  # Shape: (B, H, V)
#         p_y = torch.nn.functional.softmax(logits, dim=-1)

#         # Save to dataset
#         # ...


def generate_dataset(
    model: torch.nn.Module,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    horizon: int = 4,
    batch_size: int = 32,
    num_samples: int = 1000,
    resume: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    checkpoint_path: str = "results/output_tcds/checkpoint.pt",
):
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
        num_samples (int, optional): Total number of samples to generate. Defaults to 1000.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # Initialize or resume progress
    progress = 0
    if resume and os.path.exists(checkpoint_path):
        try:
            saved_data = torch.load(checkpoint_path)
            progress = saved_data.get("progress", 0)
            print(f"Resuming from sample {progress}")
        except Exception as e:
            print(f"Could not load checkpoint: {e}. Starting from beginning.")

    model.to(device)
    model.eval()

    # Open a file for streaming outputs instead of keeping everything in memory
    output_file = checkpoint_path.replace(".pt", ".jsonl")
    mode = "a" if resume and progress > 0 else "w"

    with torch.no_grad(), open(output_file, mode) as f:
        for i in range(progress, num_samples, batch_size):
            # Adjust batch size for the last iteration
            batch_size_curr = min(batch_size, num_samples - i)
            # Generate completely random sequences
            y = torch.randint(
                0, len(tokenizer), (batch_size_curr, horizon), device=device
            )

            # Get model predictions
            outputs = model(y)
            logits = outputs.logits  # Shape: (B, H, V)
            p_y = torch.nn.functional.softmax(logits, dim=-1)

            # Decode tokens for debugging/verification
            decoded = [
                tokenizer.decode(y[j].cpu().tolist()) for j in range(batch_size_curr)
            ]

            # Save samples to file (streaming instead of keeping in memory)
            for j in range(batch_size_curr):
                sample_data = {
                    "id": i + j,
                    "tokens": y[j].cpu().tolist(),
                    "text": decoded[j],
                    "logits": logits[j].cpu().tolist(),
                    "probs": p_y[j].cpu().tolist(),
                }
                f.write(json.dumps(sample_data) + "\n")
                f.flush()  # Ensure data is written immediately

            progress = i + batch_size_curr
            checkpoint_data = {
                "progress": progress,
                "timestamp": datetime.datetime.now().isoformat(),
                "total_samples": num_samples,
            }

            torch.save(checkpoint_data, checkpoint_path)
            print(
                f"Progress: {progress}/{num_samples} samples ({progress/num_samples*100:.2f}%)"
            )

            # Optional: clear CUDA cache periodically
            if device == "cuda" and (i + batch_size_curr) % (10 * batch_size) == 0:
                torch.cuda.empty_cache()

    print(f"Dataset generation complete. {progress} samples saved to {output_file}")
    return output_file


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
            # Caches the generated dataset to avoid re-generating it
            checkpoint_path=f"results/output_tcds_{prompt['name']}/checkpoint.pt",
        )

        print(f"[{prompt['name']}] Computing reconstruction error...")
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
