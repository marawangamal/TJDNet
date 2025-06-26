#!/usr/bin/env python3
"""
Test script for the MultiTokenLlama model.
"""

import torch
from transformers import AutoTokenizer
from modeling_hf_tjdsimple import MultiTokenLlamaConfig, MultiTokenLlama


def test_model():
    # Create config
    config = MultiTokenLlamaConfig(model_name="meta-llama/Llama-2-7b-hf", horizon=2)

    # Create model
    model = MultiTokenLlama(config)
    print(
        f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token

    # Test data
    text = "Hello world, this is a test"
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Test forward pass (training mode)
    print("\n=== Training Mode ===")
    outputs = model(input_ids=inputs["input_ids"], labels=inputs["input_ids"])
    print(f"Loss: {outputs['loss'].item():.4f}")
    print(f"Logits shape: {outputs['logits'].shape}")

    # Test forward pass (inference mode)
    print("\n=== Inference Mode ===")
    outputs = model(input_ids=inputs["input_ids"])
    print(f"Logits shape: {outputs['logits'].shape}")

    # Test save/load
    print("\n=== Save/Load Test ===")
    model.save_pretrained("./test_model")
    loaded_model = MultiTokenLlama.from_pretrained("./test_model")
    print("Model saved and loaded successfully!")

    # Test with different config
    print("\n=== Different Config Test ===")
    config2 = MultiTokenLlamaConfig(model_name="meta-llama/Llama-2-7b-hf", horizon=3)
    model2 = MultiTokenLlama(config2)
    print(
        f"Model2 created with {sum(p.numel() for p in model2.parameters()):,} parameters"
    )

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_model()
