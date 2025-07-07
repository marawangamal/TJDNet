#!/usr/bin/env python3
"""
Example usage of main-cli.py with different argument combinations.

This script demonstrates how to use the updated main-cli.py with various
configurations for training TJDNet models.
"""

import subprocess
import sys


def run_command(cmd, description):
    """Run a command and print the result."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd, shell=True, check=True, capture_output=True, text=True
        )
        print("✅ Success!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("❌ Error!")
        print(f"Return code: {e.returncode}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False
    return True


def main():
    """Run example commands."""

    # Example 1: Basic training with default settings
    cmd1 = "python main-cli.py --dataset stemp --max_epochs 2"
    run_command(cmd1, "Basic training with default settings")

    # Example 2: Training with custom model and parameters
    cmd2 = "python main-cli.py --model gpt2 --model_head cp --horizon 2 --rank 8 --lr 1e-4 --max_epochs 2"
    run_command(cmd2, "Training with custom model and CP head")

    # Example 3: Training with auto learning rate finder
    cmd3 = "python main-cli.py --dataset stemp --auto_lr_find --max_epochs 2"
    run_command(cmd3, "Training with auto learning rate finder")

    # Example 4: Training with test after fit
    cmd4 = "python main-cli.py --dataset stemp --test_after_fit --max_epochs 2"
    run_command(cmd4, "Training with test after fit")

    # Example 5: Training with custom batch size and sequence length
    cmd5 = (
        "python main-cli.py --dataset stemp --batch_size 16 --seq_len 64 --max_epochs 2"
    )
    run_command(cmd5, "Training with custom batch size and sequence length")

    # Example 6: Training with sampling enabled
    cmd6 = "python main-cli.py --dataset stemp --do_sample --top_k 50 --max_epochs 2"
    run_command(cmd6, "Training with sampling enabled")

    # Example 7: Training with WandB disabled
    cmd7 = "python main-cli.py --dataset stemp --disable_wandb --max_epochs 2"
    run_command(cmd7, "Training with WandB disabled")

    # Example 8: Training with custom WandB project
    cmd8 = "python main-cli.py --dataset stemp --wandb_project my-custom-project --max_epochs 2"
    run_command(cmd8, "Training with custom WandB project")

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
