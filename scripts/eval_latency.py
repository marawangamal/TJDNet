"""Benchmarking script for evaluating the latency and memory usage of different models.

This script benchmarks the latency and memory usage of different models on a specified device.

Usage:
    python scripts/eval_latency.py --device [device] --model_family [model_family] --out_seq_len [out_seq_len] --inp_seq_len [inp_seq_len]

Example:
    python scripts/eval_latency.py --device cuda --model_family llama --inp_seq_len 8 --out_seq_len 32
    python scripts/eval_latency.py --device cuda --model_family gpt2  --inp_seq_len 8 --out_seq_len 64

"""

import gc
import argparse
import itertools
import traceback

import torch
import pandas as pd

from utils.latency import benchmark_model_v2
from utils.models import create_model, train_forward
from utils.utils import replace_spec_chars


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
            elif isinstance(metric_values, int):
                # For integer values, just add the value
                row_data[f"{metric_name}"] = f"{metric_values}"
            elif isinstance(metric_values, float):
                # For float values, format to 3 decimal places
                row_data[f"{metric_name}"] = f"{metric_values:.3f}"

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


def get_params(model):
    # Get the number of parameters in the model head
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return param_count / 1e6  # Convert to millions


def main(args):
    # Define experiments
    gen_kwargs = {
        "max_new_tokens": args.out_seq_len,
        "num_beams": args.num_beams,
        "top_k": args.top_k,
        "do_sample": False,
    }
    common_kwargs = {
        "eval": {
            "benchmark_fn": lambda model, input_ids: model.generate(
                input_ids, **gen_kwargs
            ),
        },
        "train": {"benchmark_fn": train_forward},
    }[args.mode]

    exp_set_a = (
        []
        # Baseline
        + [
            {
                "name": f"{replace_spec_chars(args.model)}::baseline",
                "model_fn": create_model(
                    rank=1,
                    horizon=1,
                    model_head="cp",
                    hidden_dim=5120,
                ),
                **common_kwargs,
            }
        ]
        # CP
        + [
            {
                "name": f"{replace_spec_chars(args.model)}::cp::rank{r}::horizon{h}::hd{hd}",
                "model_fn": create_model(
                    rank=r,
                    horizon=h,
                    model_head="cp",
                    hidden_dim=hd,
                ),
                **common_kwargs,
            }
            for (r, h, hd) in zip([8, 8], [2, 3], [5120, 5120])
        ]
        # TMTP
        + [
            {
                "name": f"{replace_spec_chars(args.model)}::cpo::rank{r}::horizon{h}::hd{hd}",
                "model_fn": create_model(
                    rank=r,
                    horizon=h,
                    model_head="cpo",
                    hidden_dim=hd,
                ),
                **common_kwargs,
            }
            for (r, h, hd) in zip([8, 8], [2, 3], [2048, 2048])
        ]
        # MTP
        + [
            {
                "name": f"{replace_spec_chars(args.model)}::cpo::rank{r}::horizon{h}::hd{hd}",
                "model_fn": create_model(
                    rank=r,
                    horizon=h,
                    model_head="cpo",
                    hidden_dim=hd,
                ),
                **common_kwargs,
            }
            for (r, h, hd) in zip([1, 1], [2, 3], [5120, 5120])
        ]
        # MPS
        + [
            {
                "name": f"{replace_spec_chars(args.model)}::mps::rank{r}::horizon{h}::hd{hd}",
                "model_fn": create_model(
                    rank=r,
                    horizon=h,
                    model_head="mps",
                    hidden_dim=hd,
                ),
                **common_kwargs,
            }
            for (r, h, hd) in zip([2, 2], [2, 3], [2048, 2048])
        ]
    )

    exp_set_b = [
        {
            "name": f"{replace_spec_chars(args.model)}::{args.exp_grid_model_head}::rank{r}::horizon{h}::hd{hd}",
            "model_fn": create_model(
                rank=r,
                horizon=h,
                model_head=args.exp_grid_model_head,
                hidden_dim=hd,
                use_memory_efficient_loss=args.use_memory_efficient_loss,
            ),
            **common_kwargs,
        }
        for (r, h, hd) in itertools.product(
            [1, 2, 4, 8, 16], [2, 4, 8, 16, 32], [5120, 768]
        )
    ]

    exps = {"compare": exp_set_a, "grid": exp_set_b}[args.exp]

    print(f"Starting benchmarks ({args.device})...")
    results = {}
    input_ids_dict = {
        f"bs::{args.batch_size}": torch.randint(
            0, 100, (args.batch_size, args.inp_seq_len)
        ).to(args.device),
        # "bs::8": torch.randint(0, 100, (8, args.inp_seq_len)).to(args.device),
        # "bs::32": torch.randint(0, 100, (32, args.inp_seq_len)).to(args.device),
    }
    for exp in exps:
        try:
            for input_name, input_ids in input_ids_dict.items():
                exp_name = f"{exp['name']}::{input_name}"
                print(f"\nBenchmarking {exp_name}...")

                # Build model
                model = exp["model_fn"]().to(args.device)
                if args.data_parallel and torch.cuda.device_count() > 1:
                    print("Using DataParallel")
                    model = DataParallelWithGenerate(model)

                # Run benchmark
                benchmark_fn = exp["benchmark_fn"]
                benchmark_results = benchmark_model_v2(
                    model, benchmark_fn, benchmark_fn_kwargs={"input_ids": input_ids}
                )

                # Save results
                results[exp_name] = benchmark_results
                results[exp_name]["Accuracy"] = {"mean": 0, "std": 0}
                results[exp_name]["Params [M]"] = get_params(model)

                # Clean up
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        except Exception as e:
            print(f"Error benchmarking {exp['name']}: {str(e)}")
            traceback.print_exc()  # This will print the full stack trace

    log_results(
        results,
        cols=["Model", "Latency [s]", "Params [M]", "Accuracy"],
        output_format="latex",
    )
    print("\n\n")
    # Print results
    log_results(results, cols=["Model", "Latency [s]", "Params [M]", "Accuracy"])
    # Print results (detailed)
    print("\n\n")
    log_results(results)
    log_results(results, output_format="string")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model benchmarks")
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="gpt2",
        help="Huggingface model name or path",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval"],
        default="eval",
    )
    parser.add_argument(
        "--exp",
        type=str,
        choices=["compare", "grid"],
        default="compare",
        help="Experiment to run",
    )
    parser.add_argument(
        "--exp_grid_model_head",
        type=str,
        choices=["cp", "cpo", "mps", "umps"],
        default="cp",
        help="Model head to use for the grid search",
    )
    parser.add_argument(
        "--use_memory_efficient_loss",
        action="store_true",
        default=False,
        help="Use memory efficient loss",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=1,  # Note: must be 1 for generation
    )
    parser.add_argument(
        "-i",
        "--inp_seq_len",
        type=int,
        default=256,
    )
    parser.add_argument(
        "-o",
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
