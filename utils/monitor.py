import torch
import wandb


import gc
import os


def log_memory(stage, rank=None):
    """Log memory usage at different stages."""
    if rank is None:
        rank = int(os.environ.get("LOCAL_RANK", 0))

    # Only log from rank 0 to avoid cluttering logs
    if rank == 0:
        # Force garbage collection first
        gc.collect()
        torch.cuda.empty_cache()

        # Get memory stats
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        max_allocated = torch.cuda.max_memory_allocated() / (1024**3)

        # Print memory stats
        print(f"\n[MEMORY - {stage}]")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved:  {reserved:.2f} GB")
        print(f"  Peak:      {max_allocated:.2f} GB")

        # Log to wandb if available
        if wandb.run is not None:
            wandb.log(
                {
                    f"memory/{stage}/allocated_gb": allocated,
                    f"memory/{stage}/reserved_gb": reserved,
                    f"memory/{stage}/peak_gb": max_allocated,
                }
            )
