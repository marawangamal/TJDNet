#!/usr/bin/env python3
"""
Test script for the MultiTokenLlama model.
"""

import torch
from transformers import AutoTokenizer
from modeling_mtllama import MultiTokenLlamaConfig, MultiTokenLlama


def test_model():
    # Create model
    config = MultiTokenLlamaConfig(model_name="distilbert/distilgpt2", horizon=2)
    model = MultiTokenLlama(config)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Test data
    inputs = tokenizer("Hello", return_tensors="pt")

    # Test forward pass
    outputs = model(input_ids=inputs["input_ids"], labels=inputs["input_ids"])
    print(f"Loss: {outputs['loss'].item():.4f}")

    # Test generation
    generated = model.generate(inputs["input_ids"], max_length=10)
    print(f"Generated: {tokenizer.decode(generated[0])}")

    # Test save/load
    model.save_pretrained("./test_model")
    loaded_model = MultiTokenLlama.from_pretrained("./test_model")
    print("Save/load: OK")

    print("✅ All tests passed!")


if __name__ == "__main__":
    test_model()
