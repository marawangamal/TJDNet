from typing import Callable
import torch
import psutil


def get_peak_memory_usage(fn: Callable, device: str = "cuda", **kwargs) -> float:
    """Get the peak memory usage of a function.

    Args:
        fn (Callable): The function to measure the memory usage of.
        device (str): The device to measure the memory usage on.
        kwargs (dict): The keyword arguments to pass to the function.

    Returns:
        float: The peak memory usage of the function in MB.
    """
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    fn(**kwargs)

    if device == "cuda":
        memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
        torch.cuda.empty_cache()
    else:
        memory_mb = psutil.Process().memory_info().rss / (1024**2)

    return memory_mb
