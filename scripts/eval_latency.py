""" "Benchmarking script for evaluating the latency and memory usage of different models.

Examples:

python scripts/eval_latency.py --device cuda --model_family llama --out_seq_len 32 --inp_seq_len 8
python scripts/eval_latency.py --device cuda --model_family gpt2 --out_seq_len 128 --inp_seq_len 256

"""

import gc
import time
import psutil
import argparse
from tqdm import tqdm

import torch
import pandas as pd
from statistics import mean, stdev

from tjdnet.distributions._base import BaseDistConfig
from tjdnet.distributions.tpnet import TensorParamNetConfig
from tjdnet.models._tjd import TJDConfig
from tjdnet.models.tjdgpt2 import TJDGPT2
from tjdnet.models.tjdllama import TJDLLAMA
from utils.latency import benchmark_model_v2


def log_results(results, output_format="markdown", cols=None):
    """Generate a formatted table of benchmark results for multiple models.

    Args:
        results (dict): Dictionary of benchmark results for each model (eg. {model_name: {metric_name: metric_value}}).
        output_format (str): Output format for the table (markdown, latex, html, or default).
        cols (list): List of columns to include in the table.
    """
    # Create a list to hold each model's data
    table_data = []

    for model_name, model_stats in results.items():
        row_data = {"Model": model_name}

        # Process each metric that has the expected structure
        for metric_name, metric_values in model_stats.items():
            if isinstance(metric_values, dict) and all(
                k in metric_values for k in ["mean", "std"]
            ):
                # For each metric, add mean ± std
                row_data[f"{metric_name}"] = (
                    f"{metric_values['mean']:.3f} ± {metric_values['std']:.3f}"
                )

        # Filter columns if specified
        if cols is not None:
            filtered_row = {col: row_data[col] for col in cols if col in row_data}
            table_data.append(filtered_row)
        else:
            table_data.append(row_data)

    # Create DataFrame
    df = pd.DataFrame(table_data)

    # Output in the desired format
    try:
        if output_format.lower() == "markdown":
            result = df.to_markdown(index=False)
        elif output_format.lower() == "latex":
            result = df.to_latex(index=False)
        elif output_format.lower() == "html":
            result = df.to_html(index=False)
        else:
            # Default to print in a readable format
            result = df.to_string(index=False)
    except ImportError:
        # Fallback if dependencies are missing
        result = df.to_string(index=False)

    print(result)
    return result


def main(args):
    # Define experiments
    gen_kwargs = {
        "max_new_tokens": args.out_seq_len,
        "num_beams": args.num_beams,
        "top_k": args.top_k,
        "do_sample": False,
    }
    gpt_experiments = [
        {
            "name": "gpt2",
            "model_fn": lambda: TJDGPT2(
                TJDConfig(
                    base_dist=BaseDistConfig(
                        vocab_size=768,
                        horizon=1,
                        rank=1,
                        param_net=TensorParamNetConfig(),
                    ),
                    model_head="base",
                )
            ),
            "benchmark_fn": lambda model, input_ids: model.generate(
                input_ids, **gen_kwargs
            ),
        },
        {
            "name": "gpt2::cp::rank1::horizon1",
            "model_fn": lambda: TJDGPT2(
                TJDConfig(
                    base_dist=BaseDistConfig(
                        vocab_size=768,
                        horizon=1,
                        rank=1,
                        param_net=TensorParamNetConfig(),
                    ),
                    model_head="cp",
                ),
            ),
            "benchmark_fn": lambda model, input_ids: model.generate(
                input_ids, **gen_kwargs
            ),
        },
        {
            "name": "gpt2::cp::rank2::horizon2",
            "model_fn": lambda: TJDGPT2(
                TJDConfig(
                    base_dist=BaseDistConfig(
                        vocab_size=768,
                        horizon=2,
                        rank=2,
                        param_net=TensorParamNetConfig(),
                    ),
                    model_head="cp",
                ),
            ),
            "benchmark_fn": lambda model, input_ids: model.generate(
                input_ids, **gen_kwargs
            ),
        },
        {
            "name": "gpt2::cp::rank2::horizon2",
            "model_fn": lambda: TJDGPT2(
                TJDConfig(
                    base_dist=BaseDistConfig(
                        vocab_size=768,
                        horizon=2,
                        rank=4,
                        param_net=TensorParamNetConfig(),
                    ),
                    model_head="cp",
                ),
            ),
            "benchmark_fn": lambda model, input_ids: model.generate(
                input_ids, **gen_kwargs
            ),
        },
        {
            "name": "gpt2::cp::rank4::horizon4",
            "model_fn": lambda: TJDGPT2(
                TJDConfig(
                    base_dist=BaseDistConfig(
                        vocab_size=768,
                        horizon=4,
                        rank=4,
                        param_net=TensorParamNetConfig(),
                    ),
                    model_head="cp",
                ),
            ),
            "benchmark_fn": lambda model, input_ids: model.generate(
                input_ids, **gen_kwargs
            ),
        },
    ]

    llama_model_kwargs = {
        "hf_model_name": "meta-llama/Llama-2-7b-chat-hf",
    }
    llama_experiments = [
        {
            "name": "llama",
            "model_fn": lambda: TJDLLAMA(
                TJDConfig(
                    base_dist=BaseDistConfig(
                        vocab_size=32000,
                        horizon=1,
                        rank=1,
                        param_net=TensorParamNetConfig(),
                    ),
                    model_head="base",
                    model_kwargs=llama_model_kwargs,
                )
            ),
            "benchmark_fn": lambda model, input_ids: model.generate(
                input_ids, **gen_kwargs
            ),
        },
        {
            "name": "llama::cp::nlayers2::rank16::horizon2",
            "model_fn": lambda: TJDLLAMA(
                TJDConfig(
                    base_dist=BaseDistConfig(
                        vocab_size=32000,
                        horizon=2,
                        rank=16,
                        param_net=TensorParamNetConfig(
                            num_layers=2,
                        ),
                    ),
                    model_head="cp",
                    model_kwargs=llama_model_kwargs,
                ),
            ),
            "benchmark_fn": lambda model, input_ids: model.generate(
                input_ids, **gen_kwargs
            ),
        },
        {
            "name": "llama::cp::nlayers2::rank32::horizon2",
            "model_fn": lambda: TJDLLAMA(
                TJDConfig(
                    base_dist=BaseDistConfig(
                        vocab_size=32000,
                        horizon=2,
                        rank=32,
                        param_net=TensorParamNetConfig(
                            num_layers=2,
                        ),
                    ),
                    model_head="cp",
                    model_kwargs=llama_model_kwargs,
                ),
            ),
            "benchmark_fn": lambda model, input_ids: model.generate(
                input_ids, **gen_kwargs
            ),
        },
    ]

    # Run benchmarks
    exps = {
        "llama": llama_experiments,
        "gpt2": gpt_experiments,
    }[args.model_family]

    print(f"Starting benchmarks ({args.device})...")
    results = {}
    input_ids = torch.randint(0, 100, (args.batch_size, args.inp_seq_len)).to(
        args.device
    )
    for exp in exps:
        try:
            print(f"\nBenchmarking {exp['name']}...")
            model = exp["model_fn"]().to(args.device)
            benchmark_fn = exp["benchmark_fn"]
            results[exp["name"]] = benchmark_model_v2(
                model, benchmark_fn, benchmark_fn_kwargs={"input_ids": input_ids}
            )
            # Add empty Accuracy column
            results[exp["name"]]["Accuracy"] = {"mean": 0, "std": 0}
            # Clean up to avoid memory accumulation between experiments
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"Error benchmarking {exp['name']}: {str(e)}")

    # Print results
    log_results(results, cols=["Model", "Latency [s]", "Accuracy"])
    # Print results (detailed)
    log_results(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model benchmarks")
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
    )
    parser.add_argument(
        "--model_family",
        type=str,
        choices=["gpt2", "llama"],
        default="gpt2",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,  # Note: must be 1 for generation
    )
    parser.add_argument(
        "--inp_seq_len",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--out_seq_len",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=32,
    )
    args = parser.parse_args()
    main(args)
