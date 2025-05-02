"""Generates a spectrum plot of the output distribution for a language model.

Example:
    python gen_output_dist_spectrum.py --model meta-llama/Llama-2-7b-chat-hf
"""

from argparse import Namespace
import argparse
import os
import os.path as osp
from typing import Callable, Union
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

from utils.utils import group_arr, plot_conf_bands, plot_groups, replace_spec_chars

"""You are a helpful assistant that answers questions step by step. \n  \n Now solve the following problem using the exact format shown above: \n [QUESTION] {question} \n [ANSWER]"""

# Answers retrieved using `scripts/chat.py`
PROMPTS = [
    {
        "name": "newline",
        "question": "\n",
        "answer": "It looks like you've entered some formatting code. If you're trying to send a message, I'd be happy to help with that. Is there something specific you'd like to talk about or ask? I can assist you with a question, provide information on a topic, or even engage in a conversation. Let me know how I can help!",  # answer recieved from the model
    },
    {"name": "space", "question": " ", "answer": ""},  # answer recieved from the model
    {
        "name": "poem",
        "question": "Write a poem.",
        "answer": """
Moonlit Serenade

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
For in this moment, that's all that matters to me.""",
    },
    {
        "name": "gsm8k",
        "question": """You are a mathematical reasoning assistant that solves problems step by step. \n  \n FORMAT INSTRUCTIONS: \n 1. Show your work with explanations \n 2. End every answer with: #### [numerical_answer_only]<eot_id> \n  \n EXAMPLE: \n [QUESTION] Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? \n [ANSWER]  Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. #### 72 <eot_id> \n  \n Now solve the following problem using the exact format shown above. Don't repeat the question: \n [QUESTION] \n Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn? \n [ANSWER]""",
        "answer": """
To find out how much Weng earned, we need to convert the time from minutes to hours and then multiply it by her hourly wage.

First, we know that there are 60 minutes in an hour. To convert 50 minutes to hours, we can divide 50 by 60.

50 minutes / 60 = 5/6 hours

Since Weng earns $12 an hour, we multiply the number of hours she worked (5/6) by her hourly wage (12).

(5/6) * 12 = 10

So, Weng earned 10 dollars yesterday.

#### 10 """,
    },
    {
        "name": "sharegpt",
        "question": 'You are a helpful assistant that answers questions step by step. \n  \n Now solve the following problem using the exact format shown above: \n [QUESTION] complete the following code from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n     \n [ANSWER]',
        "answer": '''
from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """Check if in given list of numbers, are any two numbers closer to each other than given threshold.
    Args:
        numbers (List[float]): A list of floating point numbers.
        threshold (float): The maximum distance between two numbers for them to be considered close.
    Returns:
        bool: True if any two numbers in the list are closer than the threshold, False otherwise.
    """
    
    # Sort the list in ascending order
    numbers.sort()
    
    # Iterate over the list
    for i in range(len(numbers) - 1):
        # Calculate the distance between the current number and the next one
        distance = numbers[i + 1] - numbers[i]
        
        # If the distance is less than or equal to the threshold, return True
        if distance <= threshold:
            return True
    
    # If no two numbers are closer than the threshold, return False
    return False''',
    },
]

# PROMPTS_V1 = [
#     {
#         "name": "newline",
#         "value": "\n",
#     },
#     {
#         "name": "space",
#         "value": " ",
#     },
#     {
#         "name": "poem",
#         "value": "Write a poem.",
#     },
#     {
#         "name": "gsm8k",
#         "value": "You are a mathematical reasoning assistant that solves problems step by step. \n  \n FORMAT INSTRUCTIONS: \n 1. Show all your work with clear explanations \n 2. For each calculation, use the format: <<calculation=result>>result \n 3. End every answer with: #### [numerical_answer_only]{eos_token} \n  \n EXAMPLE: \n [QUESTION] Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? \n [ANSWER]  Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. #### 72 {eos_token} \n  \n Now solve the following problem using the exact format shown above: \n [QUESTION] \n Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
#     },
#     {
#         "name": "sharegpt",
#         "value": 'You are a helpful assistant that answers questions step by step. \n  \n Now solve the following problem using the exact format shown above: \n [QUESTION] complete the following code from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n     \n [ANSWER]',
#     },
# ]


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


def prepare_data_from_spectrums(spectrums_list):
    """
    Prepare data for plot_conf_bands directly from the spectrums list.

    Parameters:
    -----------
    spectrums_list : list
        List of dictionaries with 'prompt', 'index', and 'spectrum' keys

    Returns:
    --------
    tuple: (x_values, y_values_by_group)
    """
    import numpy as np
    from collections import defaultdict

    # Group by prompt
    data_by_prompt = defaultdict(list)
    all_indices = set()

    for item in spectrums_list:
        # Extract the base prompt name (e.g., 'poem' from 'poem::2')
        base_prompt = item["prompt"].split("::")[0]

        # Add to the appropriate group
        data_by_prompt[base_prompt].append((item["index"], item["spectrum"]))
        all_indices.add(item["index"])

    # Sort indices
    x_values = sorted(all_indices)

    # Prepare y_values_by_group
    y_values_by_group = {}

    for prompt_name, values in data_by_prompt.items():
        # Group by sequence (e.g., all values for 'poem::0', then all for 'poem::2', etc.)
        values_by_seq = defaultdict(list)

        for idx, val in values:
            # Count occurrences of each index to determine which sequence it belongs to
            seq_idx = len([i for i, v in values if i == idx]) - 1
            values_by_seq[seq_idx].append((idx, val))

        # Create list of arrays, one for each sequence
        y_arrays = []

        for seq_idx in sorted(values_by_seq.keys()):
            seq_values = sorted(values_by_seq[seq_idx])
            y_arrays.append([v for _, v in seq_values])

        y_values_by_group[prompt_name] = y_arrays

    return np.array(x_values), y_values_by_group


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
    model_fn: Callable[[], torch.nn.Module],
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    start_str: str = "\n",
    checkpoint_steps: int = 5,  # Save progress every n tokens
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 16,
    resume: bool = True,
    checkpoint_path: str = "results/output_mat_batched_checkpoint.pt",
    loading_message: str = "Processing tokens...",
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
        # Check if completed
        if start_idx >= vocab_size:
            print(f"Checkpoint {checkpoint_path} already completed.")
            return p_y1_y2
        print(f"Resuming from {checkpoint_path} at token index {start_idx}.")
    else:
        print(f"Starting fresh at token index {start_idx}.")

    x = torch.tensor(tokenizer.encode(start_str, return_tensors="pt")).to(
        device
    )  # Shape: (1, seq_len)
    model_fn().to(device)

    # Use 'total' and 'initial' in tqdm, then manually call pbar.update(1)
    with tqdm(
        total=vocab_size,
        initial=start_idx,
        desc=loading_message,
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
                outputs = model_fn(torch.cat([x.repeat(y1.size(0), 1), y1], dim=-1))
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
    model_fn = lambda: AutoModelForCausalLM.from_pretrained(
        args.model,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    save_dir = f"results/plot_output_dist_spectrum/{replace_spec_chars(args.model)}"
    os.makedirs(save_dir, exist_ok=True)

    PROMPTS_WINDOWED = []
    for prompt in PROMPTS:

        if not prompt["name"] in ["poem", "gsm8k"]:
            continue

        # Split the answer into lines
        x_tokens = tokenizer.encode(prompt["question"])
        y_tokens = tokenizer.encode(prompt["answer"])
        # Create a window of 5 lines
        for j, i in enumerate(range(0, len(y_tokens) - 1, 2)):
            if j >= args.num_dists_per_prompt:
                break
            # Create a new prompt with the first line as the question and the next 4 lines as the answer
            PROMPTS_WINDOWED.append(
                {
                    "name": f"{prompt['name']}::{i}",
                    "value": tokenizer.decode(
                        x_tokens + y_tokens[: i + 2], skip_special_tokens=True
                    ),
                }
            )

    spectrums = []
    spectrums_dict = {}
    y_values_by_group = {}
    for i, prompt in enumerate(PROMPTS_WINDOWED):

        # ========== Generate output dist and spectrum
        print(f"[{prompt['name']}] Generating output distribution spectrum...")
        output_mat = generate_output_distribution_spectrum_batched(
            model_fn,
            tokenizer,
            start_str=prompt["value"],
            checkpoint_path=osp.join(
                save_dir, f"output_dist_2d_{replace_spec_chars(prompt['name'])}.pt"
            ),
            loading_message=f"Processing {prompt['name']}...",
        )

        print(f"[{prompt['name']}]Computing spectrum...")
        spectrum = get_spectrum(output_mat)

        # ====== Debug with dummy data
        # spectrum = torch.linspace(1, 1000, 1000)  # Start at 1
        # spectrum = spectrum ** (-1 - i * 0.5)  # Direct power law with i-based exponent

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
        spectrums_dict[prompt["name"]] = spectrum  # used for legacy plot
        if not prompt["name"].split("::")[0] in y_values_by_group:
            # Initialize the list for the first occurrence of the prompt
            y_values_by_group[prompt["name"].split("::")[0]] = []
        y_values_by_group[prompt["name"].split("::")[0]].append(spectrum)

        print(f"[{prompt['name']}] Plotting spectrum...")
        os.makedirs("results", exist_ok=True)

        # Print some statistics
        print(f"[{prompt['name']}] Top 10 singular values: {spectrum[:10]}")
        print(f"[{prompt['name']}] Sum of all singular values: {spectrum.sum()}")
        print(
            f"[{prompt['name']}] Effective rank (singular values > 1e-10): {(spectrum > 1e-10).sum()}"
        )

    # ==== Grouped plot
    grouped = group_arr(
        spectrums,
        lambda x: x["prompt"],  # e.g. poem::2, gsm8k::2 -- group by [x] = [y]
        lambda x: x["prompt"].split("::")[0],  # e.g. poem, gsm8k -- group by [x]
    )

    # Create a plot with confidence bands for the first group level
    plot_groups(
        grouped,
        x_key="index",
        y_key="spectrum",
        path=osp.join(save_dir, f"output_dist_2d_spectrum.png"),
        style_dims=[
            "color",
            "color",
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
        },
        axes_kwargs={
            "title": f"Spectrum of Matrix P(y1,y2|x)",
            "xlabel": "Index",
            "ylabel": "Singular Value (log scale)",
            "yscale": "log",
        },
        fig_kwargs={
            "figsize": (10, 6),
            "dpi": 300,
        },
    )

    # ==== Spectrum plot
    plot_conf_bands(
        x_values=np.arange(0, len(next(iter(y_values_by_group.items()))[1][0])),
        y_values_by_group=y_values_by_group,
        save_path=osp.join(save_dir, f"output_dist_2d_spectrum_confidence.png"),
        title=f"Spectrums of Joint Distributions P(y1,y2|x)",
    )

    # ==== Spectrum plot
    # plot_spectrum(
    #     spectrums_dict,
    #     save_path=osp.join(save_dir, f"output_dist_2d_spectrum.png"),
    # )


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
        "-n",
        "--num_dists_per_prompt",
        type=int,
        default=3,
        help="Number of distributions to generate per prompt (i.e. sequence length // 2)",
    )
    args = parser.parse_args()
    main(args)
