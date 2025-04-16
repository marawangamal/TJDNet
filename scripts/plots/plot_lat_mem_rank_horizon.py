"""Plot latency and memory while varying rankm & horizon.

Usage:
    python scripts/plots/plot_lat_mem_rank_horizon.py --device [device] --mode [train/eval]

Example:
    python scripts/plots/plot_lat_mem_rank_horizon.py --device cuda --mode train

"""

import gc
import argparse
import itertools


import torch


import matplotlib.pyplot as plt
import re

from utils.latency import benchmark_model_v2
from utils.models import create_model_gpt_fn, train_forward


def save_fig(results, path="latency_benchmark.png", y_axis="Latency [s]"):
    # Group data by model head and horizon
    data = {}

    for name, metrics in results.items():
        # Get model head, rank and horizon from name
        model_head = re.search(r"mh(\w+)::", name).group(1)  # type: ignore
        r = int(re.search(r"r(\d+)", name).group(1))  # type: ignore
        h = int(re.search(r"h(\d+)", name).group(1))  # type: ignore

        # Store data point
        key = (model_head, h)
        if key not in data:
            data[key] = {"ranks": [], "latencies": []}
        data[key]["ranks"].append(r)
        data[key]["latencies"].append(metrics[y_axis]["mean"])

    # Get unique horizons and model heads
    unique_horizons = sorted(set(h for _, h in data.keys()))
    unique_model_heads = sorted(set(mh for mh, _ in data.keys()))

    # Create color map for horizons and markers for model heads
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*"]
    colors = plt.cm.tab10.colors  # type: ignore

    # Create plot
    plt.figure()
    for (model_head, h), points in data.items():
        horizon_idx = unique_horizons.index(h)
        head_idx = unique_model_heads.index(model_head)

        plt.plot(
            points["ranks"],
            points["latencies"],
            marker=markers[head_idx % len(markers)],
            color=colors[horizon_idx % len(colors)],
            linestyle="-",
            label=f"{model_head}, h={h}",
        )

    plt.xlabel("Rank")
    plt.ylabel(y_axis)
    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    plt.close()


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
    }[args.mode]

    exps = [
        {
            "name": f"gpt2::mhcp::hd768::r{r}::h{h}",
            "model_fn": create_model_gpt_fn(
                r,
                h,
                model_head="cp",
                param_net_config={"hidden_dim": 768, "use_decoder": True},
                use_memory_efficient_loss=args.use_memory_efficient_loss,
            ),
            **common_kwargs,
        }
        for (r, h) in itertools.product([1, 2, 4, 8, 16, 32], [1, 2, 4])
    ] + [
        {
            "name": f"gpt2::mhucp::hd768::r{r}::h{h}",
            "model_fn": create_model_gpt_fn(
                r,
                h,
                model_head="ucp",
                param_net_config={"hidden_dim": 768, "use_decoder": True},
                use_memory_efficient_loss=args.use_memory_efficient_loss,
            ),
            **common_kwargs,
        }
        for (r, h) in itertools.product([1, 2, 4, 8, 16, 32], [1, 2, 4])
    ]

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

    save_fig(
        results,
        path=f"results/plots/cp_gpu_lat_benchmark_{args.mode}_ume{args.use_memory_efficient_loss}.png",
        y_axis="Latency [s]",
    )
    save_fig(
        results,
        path=f"results/plots/cp_gpu_mem_benchmark_{args.mode}_ume{args.use_memory_efficient_loss}.png",
        y_axis="GPU Memory (allocated)[MB]",
    )


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
