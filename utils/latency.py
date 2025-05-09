""" "Benchmarking script for evaluating the latency and memory usage of different models.

Examples:

python scripts/eval_latency.py --device cuda --model_family llama --out_seq_len 32 --inp_seq_len 8
python scripts/eval_latency.py --device cuda --model_family gpt2 --out_seq_len 128 --inp_seq_len 256

"""

import gc
import time
import psutil
from tqdm import tqdm

import torch
from statistics import mean, stdev


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
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        _ = benchmark_fn(model, **benchmark_fn_kwargs)
        if torch.cuda.is_available():
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

    # Process GPU memory stats
    gpu_memory = {}
    if torch.cuda.is_available() and peak_gpu_mem:
        # Use device 0 for simplicity
        device = f"cuda:0"
        allocated_vals = [run[device]["peak_allocated_mb"] for run in peak_gpu_mem]
        reserved_vals = [run[device]["peak_reserved_mb"] for run in peak_gpu_mem]

        gpu_memory["allocated"] = {
            "mean": mean(allocated_vals),
            "std": stdev(allocated_vals) if len(allocated_vals) > 1 else 0,
            "min": min(allocated_vals),
            "max": max(allocated_vals),
        }

        gpu_memory["reserved"] = {
            "mean": mean(reserved_vals),
            "std": stdev(reserved_vals) if len(reserved_vals) > 1 else 0,
            "min": min(reserved_vals),
            "max": max(reserved_vals),
        }

    # Process CPU memory stats
    cpu_memory = {}
    if peak_cpu_mem:
        rss_vals = [run["rss_mb"] for run in peak_cpu_mem]
        cpu_memory["rss"] = {
            "mean": mean(rss_vals),
            "std": stdev(rss_vals) if len(rss_vals) > 1 else 0,
            "min": min(rss_vals),
            "max": max(rss_vals),
        }

    return {
        "Latency [s]": {
            "mean": mean(latencies),
            "std": stdev(latencies) if len(latencies) > 1 else 0,
            "min": min(latencies),
            "max": max(latencies),
        },
        "GPU Memory (allocated)[MB]": gpu_memory.get("allocated", {}),
        "GPU Memory (reserved) [MB]": gpu_memory.get("reserved", {}),
        "CPU Memory (rss) [MB]": cpu_memory.get("rss", {}),
    }


def get_params(model):
    # Get the number of parameters in the model head
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return param_count / 1e9  # Convert to billions
