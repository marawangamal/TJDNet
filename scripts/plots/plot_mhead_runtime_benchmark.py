"""Plot latency and memory while varying rank & horizon.

Usage:
    python scripts/plots/plot_mhead_runtime_benchmark.py --device [device] --mode [train/eval]

Example:
    python scripts/plots/plot_mhead_runtime_benchmark.py --device cuda --mode train

"""

import gc
import argparse
import itertools


import torch


import re

from utils.latency import benchmark_model_v2
from utils.models import create_model_gpt_fn, train_forward
from utils.utils import group_arr, plot_groups


def parse_model_head(name):
    return re.search(r"mh(\w+)::", name).group(1)  # type: ignore


def parse_model_horizon(name):
    h = int(re.search(r"h(\d+)", name).group(1))  # type: ignore
    return f"h={h}"


def main(args):

    gen_kwargs = {
        "max_new_tokens": args.out_seq_len,
        "do_sample": False,
    }
    common_kwargs = {
        "eval": {
            "benchmark_fn": lambda model, input_ids: model.generate(
                input_ids, **gen_kwargs
            ),
        },
        "train": {"benchmark_fn": train_forward},
        "x_axis_label": "Params",
        "y_axis_label": "Latency [s]",
    }[args.mode]

    exps = (
        [
            {
                "name": f"gpt2::mhcp::hd768::r{r}::h{h}",
                "params": h * r,
                "group": "cp",
                "model_fn": create_model_gpt_fn(
                    r,
                    h,
                    model_head="cp",
                    use_memory_efficient_loss=args.use_memory_efficient_loss,
                ),
                **common_kwargs,
            }
            # for (r, h) in zip([1, 2], [1, 2])
            for (r, h) in itertools.product([1, 2, 4, 8, 16, 32], [1, 2, 4])
        ]
        + [
            {
                "name": f"gpt2::mhucp::hd768::r{r}::h{h}",
                "params": r,
                "model_fn": create_model_gpt_fn(
                    r,
                    h,
                    model_head="ucp",
                    use_memory_efficient_loss=args.use_memory_efficient_loss,
                ),
                **common_kwargs,
            }
            for (r, h) in itertools.product([1, 2, 4, 8, 16, 32], [1, 2, 4])
        ]
        + [
            {
                "name": f"gpt2::mhmps::hd768::r{r}::h{h}",
                "params": h * r**2,
                "model_fn": create_model_gpt_fn(
                    r,
                    h,
                    model_head="mps",
                    use_memory_efficient_loss=args.use_memory_efficient_loss,
                ),
                **common_kwargs,
            }
            for (r, h) in itertools.product([1, 2, 4], [1, 2, 4])
        ]
        + [
            {
                "name": f"gpt2::mhumps::hd768::r{r}::h{h}",
                "params": r**2,
                "model_fn": create_model_gpt_fn(
                    r,
                    h,
                    model_head="umps",
                    use_memory_efficient_loss=args.use_memory_efficient_loss,
                ),
                **common_kwargs,
            }
            for (r, h) in itertools.product([1, 2, 4], [1, 2, 4])
        ]
    )

    print(f"Starting benchmarks ({args.device})...")
    results = []
    input_ids = torch.randint(0, 100, (args.batch_size, args.inp_seq_len)).to(
        args.device
    )
    for exp in exps:
        try:
            print(f"\nBenchmarking {exp['name']}...")
            model = exp["model_fn"]().to(args.device)
            benchmark_fn = exp["benchmark_fn"]
            res = benchmark_model_v2(
                model, benchmark_fn, benchmark_fn_kwargs={"input_ids": input_ids}
            )

            # Append to results
            results.append(
                {
                    **exp,
                    "latency": res["Latency [s]"]["mean"],
                    "memory": res["GPU Memory (allocated)[MB]"]["mean"],
                }
            )

            # Clean up
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"Error benchmarking {exp['name']}: {str(e)}")

    results_grouped = group_arr(
        results,
        lambda x: parse_model_head(x["name"]),
        lambda d: parse_model_horizon(d["name"]),
    )

    # Plot results
    for exp_kwargs in [
        {
            "y_key": "latency",
            "y_label": "Latency [s]",
            "path": f"results/plots/model_head_latency_benchmark_{args.mode}_ume{args.use_memory_efficient_loss}.png",
        },
        {
            "y_key": "memory",
            "y_label": "GPU Memory [MB]",
            "path": f"results/plots/model_head_memory_benchmark_{args.mode}_ume{args.use_memory_efficient_loss}.png",
        },
    ]:
        plot_groups(
            results_grouped,
            x_key="params",
            x_label="Params",
            # First level controls color, second controls marker
            style_dims=[
                "color",
                "marker",
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
                ]
            },
            **exp_kwargs,
        )

    # save_fig(
    #     results,
    #     path=f"results/plots/cp_gpu_mem_benchmark_{args.mode}_ume{args.use_memory_efficient_loss}.png",
    #     y_axis="GPU Memory (allocated)[MB]",
    # )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval"],
        default="eval",
    )
    parser.add_argument(
        "-u",
        "--use_memory_efficient_loss",
        action="store_true",
        help="Use memory efficient loss",
        default=False,
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--inp_seq_len", type=int, default=256)
    parser.add_argument("--out_seq_len", type=int, default=128)
    args = parser.parse_args()

    main(args)
