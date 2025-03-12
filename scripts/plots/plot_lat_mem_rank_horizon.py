import gc
import argparse
import itertools


import torch
from tjdnet.distributions._base import BaseDistConfig
from tjdnet.distributions.tpnet import TensorParamNetConfig
from tjdnet.models._tjd import TJDConfig
from tjdnet.models.tjdgpt2 import TJDGPT2


import matplotlib.pyplot as plt
import re

from utils.latency import benchmark_model_v2


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

    bast_dist_kwargs = {"vocab_size": 768}
    gen_kwargs = {
        "max_new_tokens": args.out_seq_len,
        "do_sample": False,
    }

    def create_model_fn(rank, horizon):
        return lambda: TJDGPT2(
            TJDConfig(
                base_dist=BaseDistConfig(
                    **bast_dist_kwargs,
                    rank=rank,
                    horizon=horizon,
                    param_net=TensorParamNetConfig(),
                ),
                model_head="cp",
            )
        )

    exps = [
        {
            "name": f"gpt2::r{r}::h{h}",
            "model_fn": create_model_fn(r, h),  # Pass current r, h values
            "benchmark_fn": lambda model, input_ids: model.generate(
                input_ids, **gen_kwargs
            ),
        }
        #  2, 4, 8, 16, 32,
        # 2, 3,
        for (r, h) in itertools.product([1, 2, 4, 8, 16, 32, 64], [1, 2, 4])
        # for (r, h) in zip([1, 2, 4], [1, 2, 4])
        # for (r, h) in zip([64], [5])
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

    save_fig(results)
    save_fig(results, path="gpu_mem_benchmark.png", y_axis="GPU Memory (allocated)[MB]")


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
