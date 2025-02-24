""""Benchmarking script for evaluating the latency of different models.

Examples:

python eval_latency.py --device cuda --model_family llama --out_seq_len 32 --inp_seq_len 8
python eval_latency.py --device cuda --model_family gpt2 --out_seq_len 128 --inp_seq_len 256

"""

import argparse
import torch
from tqdm import tqdm
import time
from statistics import mean, stdev

from distributions._base import BaseDistConfig
from distributions.tpnet import TensorParamNetConfig
from models._tjd import TJDConfig
from models.tjdgpt2 import TJDGPT2
from models.tjdllama import TJDLLAMA


def benchmark_model_v2(
    model, benchmark_fn, benchmark_fn_kwargs={}, num_runs=10, num_warmup=3
):

    print("Warming up...")
    for _ in tqdm(range(num_warmup), desc="Warmup", leave=False):
        _ = benchmark_fn(model, **benchmark_fn_kwargs)

    latencies = []
    for i in tqdm(range(num_runs), desc="Benchmark", leave=False):
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        _ = benchmark_fn(model, **benchmark_fn_kwargs)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - start_time)

    return {
        "mean": mean(latencies),
        "std": stdev(latencies) if len(latencies) > 1 else 0,
        "min": min(latencies),
        "max": max(latencies),
        "all_latencies": latencies,
    }


def log_summary(result):
    print(f"Mean latency: {result['mean']:.3f}s ± {result['std']:.3f}s")
    print(f"Min latency: {result['min']:.3f}s")
    print(f"Max latency: {result['max']:.3f}s")


def log_results(results):
    # Print results
    print("\nBenchmark Results:")
    print("-" * 50)
    # for model_name, stats in results.items():
    for model_name, stats in results.items():
        print(f"\n{model_name}:")
        print(f"Mean latency: {stats['mean']:.3f}s ± {stats['std']:.3f}s")
        print(f"Min: {stats['min']:.3f}s | Max: {stats['max']:.3f}s")


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
    input_ids = torch.randint(0, 100, (args.batch_size, args.inp_seq_len)).to("cuda")
    for exp in exps:
        try:
            print(f"\nBenchmarking {exp['name']}...")
            model = exp["model_fn"]().to("cuda")
            benchmark_fn = exp["benchmark_fn"]
            # Run experiment
            results[exp["name"]] = benchmark_model_v2(
                model, benchmark_fn, benchmark_fn_kwargs={"input_ids": input_ids}
            )
            log_summary(results[exp["name"]])
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
