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
    inputs = tokenizer("Hello world this is a test sentence", return_tensors="pt")
    print(f"Input shape: {inputs['input_ids'].shape}")
    print(f"Input tokens: {inputs['input_ids']}")
    print(f"Decoded input: {tokenizer.decode(inputs['input_ids'][0])}")

    # Test forward pass with debugging
    with torch.no_grad():
        outputs = model(input_ids=inputs["input_ids"], labels=inputs["input_ids"])
        print(f"Loss: {outputs['loss'].item():.4f}")
        print(f"Logits shape: {outputs['logits'].shape}")

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
