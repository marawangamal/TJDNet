""""Benchmarking script for evaluating the latency and memory usage of different models.

Examples:

python eval_latency.py --device cuda --model_family llama --out_seq_len 32 --inp_seq_len 8
python eval_latency.py --device cuda --model_family gpt2 --out_seq_len 128 --inp_seq_len 256

"""

import argparse
import torch
from tqdm import tqdm
import time
import psutil
import gc
from statistics import mean, stdev

from distributions._base import BaseDistConfig
from distributions.tpnet import TensorParamNetConfig
from models._tjd import TJDConfig
from models.tjdgpt2 import TJDGPT2
from models.tjdllama import TJDLLAMA


def get_gpu_memory_stats():
    """Get current GPU memory usage statistics."""
    if not torch.cuda.is_available():
        return None

    stats = {}
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024**2)  # MB
        reserved = torch.cuda.memory_reserved(i) / (1024**2)  # MB
        stats[f"cuda:{i}"] = {
            "allocated_mb": allocated,
            "reserved_mb": reserved,
        }
    return stats


def get_cpu_memory_stats():
    """Get current CPU memory usage statistics."""
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        "rss_mb": memory_info.rss / (1024**2),  # Resident Set Size in MB
        "vms_mb": memory_info.vms / (1024**2),  # Virtual Memory Size in MB
    }


def benchmark_model_v2(
    model, benchmark_fn, benchmark_fn_kwargs={}, num_runs=10, num_warmup=3
):
    print("Warming up...")
    for _ in tqdm(range(num_warmup), desc="Warmup", leave=False):
        _ = benchmark_fn(model, **benchmark_fn_kwargs)

    # Clear cache before starting benchmark
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Get baseline memory usage before benchmarking
    baseline_gpu_mem = get_gpu_memory_stats()
    baseline_cpu_mem = get_cpu_memory_stats()

    latencies = []
    peak_gpu_mem = []
    peak_cpu_mem = []

    for i in tqdm(range(num_runs), desc="Benchmark", leave=False):
        # Clear cache before each run
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Start tracking peak memory
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Run benchmark
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        _ = benchmark_fn(model, **benchmark_fn_kwargs)
        torch.cuda.synchronize()
        end_time = time.perf_counter()

        # Record latency
        latencies.append(end_time - start_time)

        # Get peak memory usage
        if torch.cuda.is_available():
            gpu_stats = {}
            for i in range(torch.cuda.device_count()):
                peak_allocated = torch.cuda.max_memory_allocated(i) / (1024**2)  # MB
                peak_reserved = torch.cuda.max_memory_reserved(i) / (1024**2)  # MB
                gpu_stats[f"cuda:{i}"] = {
                    "peak_allocated_mb": peak_allocated,
                    "peak_reserved_mb": peak_reserved,
                }
            peak_gpu_mem.append(gpu_stats)

        # Get current CPU memory usage
        peak_cpu_mem.append(get_cpu_memory_stats())

    # Calculate memory deltas from baseline
    mem_stats = {
        "gpu_baseline": baseline_gpu_mem,
        "cpu_baseline": baseline_cpu_mem,
        "peak_gpu": peak_gpu_mem,
        "peak_cpu": peak_cpu_mem,
    }

    return {
        "mean": mean(latencies),
        "std": stdev(latencies) if len(latencies) > 1 else 0,
        "min": min(latencies),
        "max": max(latencies),
        "all_latencies": latencies,
        "memory": mem_stats,
    }


def log_summary(result):
    print(f"Mean latency: {result['mean']:.3f}s ± {result['std']:.3f}s")
    print(f"Min latency: {result['min']:.3f}s")
    print(f"Max latency: {result['max']:.3f}s")

    # Log memory usage summary
    if torch.cuda.is_available() and result["memory"]["peak_gpu"]:
        # Get the first device's stats
        device = list(result["memory"]["peak_gpu"][0].keys())[0]
        peak_allocs = [
            run[device]["peak_allocated_mb"] for run in result["memory"]["peak_gpu"]
        ]
        print(
            f"Mean GPU memory (allocated): {mean(peak_allocs):.2f} MB ± {stdev(peak_allocs) if len(peak_allocs) > 1 else 0:.2f} MB"
        )
        print(f"Max GPU memory (allocated): {max(peak_allocs):.2f} MB")

    # Log CPU memory
    if result["memory"]["peak_cpu"]:
        peak_rss = [run["rss_mb"] for run in result["memory"]["peak_cpu"]]
        print(
            f"Mean CPU memory (RSS): {mean(peak_rss):.2f} MB ± {stdev(peak_rss) if len(peak_rss) > 1 else 0:.2f} MB"
        )
        print(f"Max CPU memory (RSS): {max(peak_rss):.2f} MB")


def log_results(results):
    # Print results
    print("\nBenchmark Results:")
    print("-" * 50)
    for model_name, stats in results.items():
        print(f"\n{model_name}:")
        print(f"Mean latency: {stats['mean']:.3f}s ± {stats['std']:.3f}s")
        print(f"Min: {stats['min']:.3f}s | Max: {stats['max']:.3f}s")

        # Memory usage
        if torch.cuda.is_available() and stats["memory"]["peak_gpu"]:
            # Get the first device's stats
            device = list(stats["memory"]["peak_gpu"][0].keys())[0]
            peak_allocs = [
                run[device]["peak_allocated_mb"] for run in stats["memory"]["peak_gpu"]
            ]
            print(
                f"GPU memory (allocated): {mean(peak_allocs):.2f} MB ± {stdev(peak_allocs) if len(peak_allocs) > 1 else 0:.2f} MB"
            )

        # CPU memory
        if stats["memory"]["peak_cpu"]:
            peak_rss = [run["rss_mb"] for run in stats["memory"]["peak_cpu"]]
            print(
                f"CPU memory (RSS): {mean(peak_rss):.2f} MB ± {stdev(peak_rss) if len(peak_rss) > 1 else 0:.2f} MB"
            )


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
            "name": "llama::cp::rank2::horizon2",
            "model_fn": lambda: TJDLLAMA(
                TJDConfig(
                    base_dist=BaseDistConfig(
                        vocab_size=32000,
                        horizon=2,
                        rank=2,
                        param_net=TensorParamNetConfig(),
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
            "name": "llama::cp::rank4::horizon4",
            "model_fn": lambda: TJDLLAMA(
                TJDConfig(
                    base_dist=BaseDistConfig(
                        vocab_size=32000,
                        horizon=4,
                        rank=4,
                        param_net=TensorParamNetConfig(),
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
            # Run experiment
            results[exp["name"]] = benchmark_model_v2(
                model, benchmark_fn, benchmark_fn_kwargs={"input_ids": input_ids}
            )
            log_summary(results[exp["name"]])

            # Clean up to avoid memory accumulation between experiments
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"Error benchmarking {exp['name']}: {str(e)}")

    # Print results
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
