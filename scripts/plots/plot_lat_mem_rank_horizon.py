import gc
import argparse
import itertools


import torch


import matplotlib.pyplot as plt
import re

from utils.latency import benchmark_model_v2
from utils.models import create_model_gpt_fn


def save_fig(results, path="latency_benchmark.png", y_axis="Latency [s]"):
    # Group data by horizon
    data = {}

    for name, metrics in results.items():
        # Get rank and horizon from name
        r = int(re.search(r"r(\d+)", name).group(1))
        h = int(re.search(r"h(\d+)", name).group(1))

        # Store data point
        if h not in data:
            data[h] = {"ranks": [], "latencies": []}
        data[h]["ranks"].append(r)
        data[h]["latencies"].append(metrics[y_axis]["mean"])

    # Create plot
    plt.figure()
    for h, points in data.items():
        plt.plot(points["ranks"], points["latencies"], "o-", label=f"h={h}")

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

    exps = [
        {
            "name": f"gpt2::r{r}::h{h}",
            "model_fn": create_model_gpt_fn(r, h),  # Pass current r, h values
            "benchmark_fn": lambda model, input_ids: model.generate(
                input_ids, **gen_kwargs
            ),
        }
        for (r, h) in itertools.product([1, 2, 4, 8, 16, 32, 64], [1, 2, 4])
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
        path="results/plots/cp_gpu_lat_benchmark.png",
    )
    save_fig(
        results,
        path="results/plots/cp_gpu_mem_benchmark.png",
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
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--inp_seq_len", type=int, default=256)
    parser.add_argument("--out_seq_len", type=int, default=128)
    args = parser.parse_args()

    main(args)
