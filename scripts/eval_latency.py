"""Benchmarking script for evaluating the latency and memory usage of different models.

This script benchmarks the latency and memory usage of different models on a specified device.

Usage:
    python scripts/eval_latency.py --device [device] --model_family [model_family] --out_seq_len [out_seq_len] --inp_seq_len [inp_seq_len]

Example:
    python scripts/eval_latency.py --device cuda --model_family llama --out_seq_len 32 --inp_seq_len 8
    python scripts/eval_latency.py --device cuda --model_family gpt2 --out_seq_len 128 --inp_seq_len 256

"""

import gc
import argparse
import itertools

import torch
import pandas as pd

from utils.latency import benchmark_model_v2
from utils.models import create_model_gpt_fn, create_model_llama_fn


class DataParallelWithGenerate(torch.nn.DataParallel):
    """Wrapper around DataParallel to expose generate method"""

    def generate(self, *args, **kwargs):
        # Call the module's generate method directly
        return self.module.generate(*args, **kwargs)


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
    common_kwargs = {
        "benchmark_fn": lambda model, input_ids: model.generate(
            input_ids, **gen_kwargs
        ),
    }
    gpt_experiments = (
        [
            {
                "name": "gpt2::base",
                "model_fn": create_model_gpt_fn(1, 1, model_head="base"),
                **common_kwargs,
            }
        ]
        + [
            {
                "name": f"gpt2::cp::horizon{h}::rank{r}",
                "model_fn": create_model_gpt_fn(
                    rank=r,
                    horizon=h,
                    model_head="cp",
                ),
                **common_kwargs,
            }
            for (h, r) in itertools.product([2, 4], [4, 8, 16])
        ]
        + [
            {
                "name": f"gpt2::ucp:horizon{h}::rank{r}",
                "model_fn": create_model_gpt_fn(
                    rank=r,
                    horizon=h,
                    model_head="ucp",
                ),
                **common_kwargs,
            }
            for (h, r) in itertools.product([2, 4], [4, 8, 16])
            # for (r, h) in itertools.product([4, 8, 16], [2, 4])
        ]
    )

    llama_experiments = (
        [
            {
                "name": "llama::base",
                "model_fn": create_model_llama_fn(
                    1,
                    1,
                    model_head="base",
                ),
                **common_kwargs,
            }
        ]
        + [
            {
                "name": f"llama::cp::nl2::rank{r}::horizon{h}",
                "model_fn": create_model_llama_fn(
                    rank=r,
                    horizon=h,
                    model_head="cp",
                ),
                **common_kwargs,
            }
            for (r, h) in zip([4, 8, 16, 32], [2, 2, 2, 2])
        ]
        + [
            {
                "name": f"llama::ucp:horizon{h}::rank{r}",
                "model_fn": create_model_llama_fn(
                    rank=r,
                    horizon=h,
                    model_head="ucp",
                ),
                **common_kwargs,
            }
            for (r, h) in zip([4, 8, 16, 32], [2, 2, 2, 2])
        ]
    )

    # Run benchmarks
    exps = {
        "llama": llama_experiments,
        "gpt2": gpt_experiments,
    }[args.model_family]

    print(f"Starting benchmarks ({args.device})...")
    results = {}
    input_ids_dict = {
        "bs::1": torch.randint(0, 100, (1, args.inp_seq_len)).to(args.device),
        # "bs::8": torch.randint(0, 100, (8, args.inp_seq_len)).to(args.device),
        # "bs::32": torch.randint(0, 100, (32, args.inp_seq_len)).to(args.device),
    }
    for exp in exps:
        try:
            for input_name, input_ids in input_ids_dict.items():
                exp_name = f"{exp['name']}::{input_name}"
                print(f"\nBenchmarking {exp_name}...")
                model = exp["model_fn"]().to(args.device)
                if args.data_parallel and torch.cuda.device_count() > 1:
                    print("Using DataParallel")
                    model = DataParallelWithGenerate(model)
                benchmark_fn = exp["benchmark_fn"]
                results[exp_name] = benchmark_model_v2(
                    model, benchmark_fn, benchmark_fn_kwargs={"input_ids": input_ids}
                )
                # Add empty Accuracy column
                results[exp_name]["Accuracy"] = {"mean": 0, "std": 0}
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
    parser.add_argument(
        "-p",  #
        "--data_parallel",
        action="store_true",
        help="Use data parallelism",
    )
    args = parser.parse_args()
    main(args)
