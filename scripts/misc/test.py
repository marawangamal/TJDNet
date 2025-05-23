# Test script with dummy data
from utils.helpers import get_model_and_tokenizer

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import os

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from functools import partial
from transformers.models.gptj.modeling_gptj import (
    GPTJBlock,
)

from utils.arguments_hf import parse_args  # Update this import based on your model


def setup_process_group():
    """Initialize the process group for distributed training."""
    # Initialize process group
    dist.init_process_group(
        backend="nccl",  # Use nccl for GPU training
        init_method="env://",
    )
    # Set the device to current process's visible device
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    print(
        f"Initialized process group: rank={dist.get_rank()}, world_size={dist.get_world_size()}"
    )


def get_transformer_layer_cls_for_model(model):
    """Determine the transformer layer classes based on model architecture."""
    model_class_name = model.__class__.__name__.lower()
    print(f"Inspecting model: {model_class_name}")

    # For custom TJDHuggingFace model, we need to deeply inspect the backbone
    if "tjd" in model_class_name:
        # Print the model to see its structure
        print("Model:", model)

        # Based on the printed structure, we can see we need to go:
        # model.backbone.base_model.model.layers[0].__class__
        if (
            hasattr(model, "backbone")
            and hasattr(model.backbone, "base_model")
            and hasattr(model.backbone.base_model, "model")
        ):
            base_model = model.backbone.base_model.model

            # Now check what kind of layers this model has
            if hasattr(base_model, "layers") and len(base_model.layers) > 0:
                layer_class = base_model.layers[0].__class__
                print(f"Found layer class: {layer_class.__name__}")
                return {layer_class}

            # Try other common layer names if "layers" doesn't exist
            elif hasattr(base_model, "h") and len(base_model.h) > 0:
                layer_class = base_model.h[0].__class__
                print(f"Found layer class: {layer_class.__name__}")
                return {layer_class}

    # If we still can't find transformer layers, raise an error with more diagnostic info
    raise ValueError(
        f"Couldn't determine transformer layer class for model: {model_class_name}. "
        f"Please specify the transformer layer class manually. "
        f"Model structure: {model.__class__}"
    )


if __name__ == "__main__":
    # Initialize process group first
    setup_process_group()

    args = parse_args()
    model, _ = get_model_and_tokenizer(args)

    # Then in your main code:
    transformer_layer_cls = get_transformer_layer_cls_for_model(model)
    wrap_policy = partial(
        transformer_auto_wrap_policy, transformer_layer_cls=transformer_layer_cls
    )

    # Define FSDP config parameters explicitly to see what happens
    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,  # Use the partial function with layer class specified
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # Your config uses 1 = FULL_SHARD
        backward_prefetch=None,  # Your config uses NONE
        use_orig_params=True,  # Required for LoRA
        device_id=torch.cuda.current_device(),
    )

    # Print model structure with FSDP wrapping
    print("===== FSDP Wrapped Model Structure =====")
    print(model)

    # Print additional information about the FSDP wrapping
    print("\n===== FSDP Wrapping Summary =====")

    def count_fsdp_modules(module):
        count = 0
        fsdp_submodules = []
        for child in module.children():
            if isinstance(child, FSDP):
                count += 1
                fsdp_submodules.append(child)
                count += count_fsdp_modules(child)[0]
        return count, fsdp_submodules

    fsdp_count, fsdp_modules = count_fsdp_modules(model)
    print(f"Total FSDP wrapped modules: {fsdp_count + 1}")  # +1 for the root module

    # Clean up
    dist.destroy_process_group()
