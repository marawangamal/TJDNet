import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="distilbert/distilgpt2")
    args = parser.parse_args()

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)

    # Print details
    print(f"Model: {args.model}")
    print(f"Params: {sum(p.numel() for p in model.parameters())}")
    print(f"Size (GB): {sum(p.numel() for p in model.parameters()) * 4 / 1e9:.2f}")
    print(f"Vocab size: {len(tokenizer.get_vocab())}")


if __name__ == "__main__":
    main()
