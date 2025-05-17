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


def calculate_model_memory_breakdown(model, batch_size, seq_len, dtype=torch.float32):
    """Calculate the theoretical memory breakdown for the model."""

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Calculate memory requirements
    bytes_per_param = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1,
    }[dtype]

    # Model parameters
    params_memory_gb = total_params * bytes_per_param / (1024**3)

    # Optimizer states (Adam uses 2 states per parameter)
    optimizer_memory_gb = params_memory_gb * 2  # For Adam

    # Rough activation memory estimate (depends on model architecture)
    # This is a very rough estimate - actual usage varies by model
    hidden_size = (
        model.config.hidden_size if hasattr(model.config, "hidden_size") else 768
    )
    layers = (
        model.config.num_hidden_layers
        if hasattr(model.config, "num_hidden_layers")
        else 12
    )
    activation_memory_gb = (
        batch_size * seq_len * hidden_size * layers * bytes_per_param
    ) / (1024**3)

    # Gradients
    gradients_memory_gb = params_memory_gb

    # Total theoretical memory
    total_theoretical_gb = (
        params_memory_gb
        + optimizer_memory_gb
        + activation_memory_gb
        + gradients_memory_gb
    )

    return {
        "parameters_gb": params_memory_gb,
        "optimizer_states_gb": optimizer_memory_gb,
        "activations_estimate_gb": activation_memory_gb,
        "gradients_gb": gradients_memory_gb,
        "total_theoretical_gb": total_theoretical_gb,
        "total_params": total_params,
        "params_in_billions": total_params / 1e9,
    }
