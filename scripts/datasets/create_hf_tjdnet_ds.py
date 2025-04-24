"""Plots reconstruction error of language model output distributions using tensor completion.

Example:
    python plot_output_dist_recons_error.py --model meta-llama/Llama-2-7b-chat-hf

"""

from argparse import Namespace
import re
from tqdm import tqdm
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
        "value": "Write a short 10 word poem.",
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
    batch_size: int = 32,
    num_samples: int = 1000,
    resume: bool = True,
    start_str: str = "\n",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    checkpoint_steps: int = 10,
    checkpoint_path: str = "results/output_tcds/checkpoint.pt",
):
    """Generates dataset for tensor completion. (i.e., samples from T where T is of shape V^H
    Args:
        model (torch.nn.Module): Model used to generate the dataset.
        tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): Tokenizer for the model.
        horizon (int, optional): Number of tokens to generate. Defaults to 4.
        batch_size (int, optional): Defaults to 32.
        num_samples (int, optional): Total number of samples to generate. Defaults to 1000.
        resume (bool, optional): Whether to resume from checkpoint. Defaults to True.
        device (str, optional): Device to run on. Defaults to "cuda" if available else "cpu".
        checkpoint_path (str, optional): Path to save checkpoint. Defaults to "results/output_tcds/checkpoint.pt".

    Sample output:
        {"id": 991, "x": [198], "x_decoded": "\n", "y": [39831, 41763, 3767, 40692], "y_decoded": " Farmingamen continuedZX", "py|x": 3.333-27}

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
    else:
        print("Starting from scratch.")
    model.to(device)
    model.eval()

    # Open a file for streaming outputs instead of keeping everything in memory
    output_file = checkpoint_path.replace(".pt", ".jsonl")
    mode = "a" if resume and progress > 0 else "w"

    # Initialize tqdm progress bar
    pbar = tqdm(total=num_samples, initial=progress, desc="Generating samples")

    data = []

    with torch.no_grad(), open(output_file, mode) as f:
        for i_prog in range(progress, num_samples, batch_size):
            # Adjust batch size for the last iteration
            batch_size_curr = min(batch_size, num_samples - i_prog)

            x = (
                torch.tensor(tokenizer.encode(start_str), device=device)
                .reshape(1, -1)
                .repeat(batch_size_curr, 1)
            )

            # === Generate random seqs ==========
            # # Generate completely random sequences
            # y = torch.randint(
            #     0, len(tokenizer), (batch_size_curr, horizon), device=device
            # )

            # outputs = model(torch.cat([x, y], dim=1))
            # logits = outputs.logits  # Shape: (B, H, V)

            # === Generate realisitic seqs ==========
            outputs = model.generate(
                x,
                max_new_tokens=horizon,
                return_dict_in_generate=True,
                output_logits=True,
                # Make samples more diverse
                do_sample=True,
                top_k=50,
                top_p=0.95,
            )
            y = outputs.sequences[:, x.size(1) :]
            logits = torch.stack(outputs.logits, dim=1)

            probs = torch.nn.functional.softmax(logits, dim=-1)
            # (B, H, V) -> (B, H)
            prob_seq = torch.gather(
                probs,  # Shape: (B, H, V)
                index=y.unsqueeze(-1),  # Shape: (B, H, 1)
                dim=-1,
            ).squeeze(-1)
            py = torch.prod(prob_seq, dim=-1)  # Shape: (B, H) -> (B)

            # Decode tokens for debugging/verification
            decoded = [
                tokenizer.decode(y[j].cpu().tolist()) for j in range(batch_size_curr)
            ]

            for j in range(batch_size_curr):
                sample_data = {
                    "id": i_prog + j,
                    "x": x[j].cpu().tolist(),
                    "x_decoded": start_str,
                    "y": y[j].cpu().tolist(),
                    "y_decoded": decoded[j],
                    "py|x": py[j].cpu().tolist(),
                }
                data.append(sample_data)

            if (i_prog * batch_size) % checkpoint_steps == 0:
                for d in data:
                    f.write(json.dumps(d) + "\n")
                f.flush()  # Ensure data is written immediately

                # Save checkpoint
                checkpoint_data = {
                    "progress": i_prog + batch_size_curr,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "total_samples": num_samples,
                }
                torch.save(checkpoint_data, checkpoint_path)

                # Clear data list after writing to file
                data = []

            # Update the progress bar
            pbar.update(batch_size_curr)

            # Optional: clear CUDA cache periodically
            if device == "cuda" and (i_prog + batch_size_curr) % (10 * batch_size) == 0:
                torch.cuda.empty_cache()

    pbar.close()
    print(f"Dataset generation complete. {progress} samples saved to {output_file}")
    return output_file


def fmt(model_name: str) -> str:
    """Format the model name for use in file paths. Remove special characters and lowercase."""
    name = model_name.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)  # replace non-alphanumerics with underscore
    name = re.sub(r"_+", "_", name).strip(
        "_"
    )  # collapse multiple underscores & strip edges
    return name


def main(args: Namespace):
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    for prompt in PROMPTS:
        print(f"[{prompt['name']}] Generating output distribution dataset...")
        for num_samples, split in zip(
            [args.num_samples, int(args.num_samples / 10)],
            ["train", "test"],
        ):
            generate_dataset(
                model,
                tokenizer,
                horizon=args.horizon,
                num_samples=num_samples,
                start_str=prompt["value"],
                checkpoint_path=f"datasets/tjdnet/{fmt(args.model)}/{prompt['name']}/{split}.pt",
                resume=not args.overwrite,
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
        default=32,
    )
    parser.add_argument(
        "-n",
        "--num_samples",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
    )
    args = parser.parse_args()
    main(args)
