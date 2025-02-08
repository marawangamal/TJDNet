import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from statistics import mean, stdev

from distributions._base import BaseDistConfig
from distributions.tpnet import TensorParamNetConfig
from models._tjd import TJDConfig
from models.tjdllama import TJDLLAMA


def benchmark_model(model, tokenizer, generate_fn, num_runs=10):
    # Prepare input
    input_text = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    input_ids = tokenizer.encode(input_text)
    input_ids = torch.tensor([input_ids]).cuda()

    # Warmup
    print("Warming up...")
    for _ in range(3):
        _ = generate_fn(model, input_ids)

    # Benchmark
    latencies = []
    print("Running benchmark...")
    for i in range(num_runs):
        torch.cuda.synchronize()  # Ensure previous run is complete
        start_time = time.perf_counter()

        _ = generate_fn(model, input_ids)

        torch.cuda.synchronize()  # Ensure generation is complete
        end_time = time.perf_counter()
        latency = end_time - start_time
        latencies.append(latency)
        print(f"Run {i+1}/{num_runs}: {latency:.3f}s")

    # Calculate statistics
    avg_latency = mean(latencies)
    std_latency = stdev(latencies) if len(latencies) > 1 else 0

    return {
        "mean": avg_latency,
        "std": std_latency,
        "min": min(latencies),
        "max": max(latencies),
        "all_latencies": latencies,
    }


if __name__ == "__main__":
    base_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

    MODELS = {
        "llama": {
            "model": AutoModelForCausalLM.from_pretrained(
                "meta-llama/Llama-2-7b-chat-hf",
                low_cpu_mem_usage=True,
                device_map="auto",
            ),
            "tokenizer": base_tokenizer,
            "generate_fn": lambda model, input_ids: model.generate(
                input_ids,
                max_new_tokens=100,
                num_beams=1,
                top_k=200,
                do_sample=True,
            ),
        },
        "tjd-llama": {
            "model": TJDLLAMA(
                TJDConfig(
                    base_dist=BaseDistConfig(
                        vocab_size=len(base_tokenizer),
                        horizon=1,
                        rank=1,
                        param_net=TensorParamNetConfig(),
                    ),
                    model_head="base",
                )
            ).cuda(),
            "tokenizer": base_tokenizer,
            "generate_fn": lambda model, input_ids: model.generate(
                input_ids,
                max_new_tokens=100,
                num_beams=1,
                top_k=200,
                do_sample=True,
                horizon=1,
            ),
        },
    }

    # Run benchmarks
    print("Starting benchmarks...")
    results = {}
    for model_name, config in MODELS.items():
        print(f"\nBenchmarking {model_name}...")
        try:
            results[model_name] = benchmark_model(
                config["model"], config["tokenizer"], config["generate_fn"]
            )
        except Exception as e:
            print(f"Error benchmarking {model_name}: {str(e)}")
            continue

    # Print results
    print("\nBenchmark Results:")
    print("-" * 50)
    for model_name, stats in results.items():
        print(f"\n{model_name}:")
        print(f"Mean latency: {stats['mean']:.3f}s ± {stats['std']:.3f}s")
        print(f"Min latency: {stats['min']:.3f}s")
        print(f"Max latency: {stats['max']:.3f}s")
