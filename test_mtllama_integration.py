#!/usr/bin/env python3
"""
Test script to verify mtllama integration with LModel.
"""

import torch
from transformers import AutoTokenizer
from utils.lmodules import LModel, LDataModule


def test_mtllama_integration():
    print("Testing mtllama integration with LModel...")

    # Create a simple LModel instance with mtllama
    model = LModel(
        model="distilbert/distilgpt2",  # Use a smaller model for testing
        horizon=2,
        dataset="stemp",  # Use a simple dataset
        seq_len=64,  # Shorter sequence length for testing
        max_new_tokens=16,
        debug=True,
    )

    print(f"Model created successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create a simple data module
    data_module = LDataModule(
        model="distilbert/distilgpt2",
        batch_size=2,
        seq_len=64,
        dataset="stemp",
        max_num_samples=10,  # Small dataset for testing
    )

    print("Setting up data module...")
    data_module.setup()

    # Test with a single batch
    print("Testing with a single batch...")
    batch = next(iter(data_module.train_dataloader()))

    # Test forward pass
    with torch.no_grad():
        outputs = model.model(**batch)
        print(f"Forward pass successful! Loss: {outputs.loss.item():.4f}")

    # Test generation
    print("Testing generation...")
    generated = model.model.generate(
        batch["input_ids"][:1],  # Use first example
        max_new_tokens=5,
        do_sample=False,
    )
    print(f"Generation successful! Shape: {generated.shape}")

    print("✅ mtllama integration test passed!")


if __name__ == "__main__":
    test_mtllama_integration()
