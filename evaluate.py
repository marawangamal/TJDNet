import os
import os.path as osp
import argparse
import torch
from helpers import get_model_and_tokenizer, load_args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Path to checkpoint directory"
    )
    return parser.parse_args()


def find_latest_checkpoint(ckpt_dir):
    """Find the latest checkpoint subdirectory in the given directory."""
    subdirs = [d for d in os.listdir(ckpt_dir) if d.startswith("checkpoint-")]
    if not subdirs:
        raise FileNotFoundError(f"No checkpoint directories found in {ckpt_dir}.")
    # Sort subdirectories by the step number (e.g., "checkpoint-5000")
    subdirs.sort(key=lambda x: int(x.split("-")[-1]), reverse=True)
    return osp.join(ckpt_dir, subdirs[0])


def main():
    args = parse_args()
    # Load saved args
    saved_args = load_args(args.ckpt)

    # Find the latest checkpoint directory
    latest_ckpt_dir = find_latest_checkpoint(args.ckpt)
    ckpt_path = osp.join(latest_ckpt_dir, "pytorch_model.bin")

    if not osp.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found at {ckpt_path}")

    # Model and tokenizer
    model, tokenizer = get_model_and_tokenizer(argparse.Namespace(**saved_args))

    # Load model state dict
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    print(f"Model loaded successfully from {latest_ckpt_dir}!")


if __name__ == "__main__":
    main()
