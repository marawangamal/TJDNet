# Main CLI Usage Guide

This document describes the updated `main-cli.py` script, which now includes experiment naming logic and comprehensive argument parsing.

## Overview

The `main-cli.py` script provides a simple way to train TJDNet models using PyTorch Lightning without the complexity of the Lightning CLI. It includes:

- **Experiment naming**: Automatic generation of experiment names based on configuration
- **Argument parsing**: Comprehensive command-line argument support
- **Auto-resume**: Automatic checkpoint resumption
- **WandB integration**: Optional logging with Weights & Biases
- **Learning rate finder**: Automatic learning rate discovery

## Key Features

### 1. Experiment Naming

The script automatically generates experiment names using the same logic as `main.py`. Experiment names are created from the configuration parameters and stored in the `experiments/` directory.

### 2. Auto-Resume

The script automatically checks for existing checkpoints and resumes training:
- `experiments/<run_name>/last.ckpt` - Resume from last checkpoint
- `experiments/<run_name>/best.ckpt` - Resume from best checkpoint

### 3. Comprehensive Argument Support

All model, data, and trainer parameters can be configured via command-line arguments.

## Usage Examples

### Basic Training

```bash
# Train with default settings
python main-cli.py --dataset stemp --max_epochs 10

# Train with custom model and parameters
python main-cli.py --model gpt2 --model_head cp --horizon 2 --rank 8 --lr 1e-4
```

### Advanced Training

```bash
# Auto learning rate finder
python main-cli.py --dataset stemp --auto_lr_find --max_epochs 10

# Test after training
python main-cli.py --dataset stemp --test_after_fit --max_epochs 10

# Custom batch size and sequence length
python main-cli.py --dataset stemp --batch_size 16 --seq_len 64 --max_epochs 10
```

### WandB Configuration

```bash
# Disable WandB logging
python main-cli.py --dataset stemp --disable_wandb --max_epochs 10

# Custom WandB project
python main-cli.py --dataset stemp --wandb_project my-project --max_epochs 10

# Custom WandB entity
python main-cli.py --dataset stemp --wandb_entity my-team --max_epochs 10
```

## Available Arguments

### Model Arguments

- `--model`: Model name or path (default: "gpt2")
- `--train_mode`: Training mode - "lora" or "full" (default: "lora")
- `--lora_rank`: LoRA rank (default: 32)
- `--model_head`: Model head type (default: "cp")
- `--horizon`: Horizon (default: 1)
- `--rank`: Rank (default: 1)
- `--positivity_func`: Positivity function (default: "sigmoid")
- `--lr`: Learning rate (default: 1e-3)
- `--warmup_steps`: Warmup steps (default: 100)
- `--grad_clip_val`: Gradient clipping value (default: 1.0)
- `--max_new_tokens`: Max new tokens for generation (default: 128)
- `--do_sample`: Enable sampling during generation (default: False)
- `--top_k`: Top-k for generation (default: 200)
- `--seq_len`: Sequence length (default: 128)
- `--dataset`: Dataset name (default: "stemp")
- `--debug`: Enable debug mode (default: False)
- `--gen_mode`: Generation mode (default: "draft")
- `--framework`: Framework (default: "tjd")

### Data Arguments

- `--batch_size`: Batch size (default: 32)
- `--max_num_samples`: Max number of samples (default: None)
- `--max_test_samples`: Max test samples (default: None)
- `--max_tokens`: Max tokens (default: None)
- `--num_workers`: Number of workers (default: 4)
- `--template_mode`: Template mode (default: "0_shot")
- `--domain_shift`: Domain shift (default: "in")

### Trainer Arguments

- `--max_epochs`: Max epochs (default: 10)
- `--accelerator`: Accelerator (default: "auto")
- `--devices`: Devices (default: "auto")
- `--precision`: Precision (default: 32)
- `--log_every_n_steps`: Log every n steps (default: 10)
- `--val_check_interval`: Validation check interval (default: 0.25)

### Experiment Arguments

- `--test_after_fit`: Run test after fit (default: False)
- `--auto_lr_find`: Auto find learning rate (default: False)
- `--wandb_project`: WandB project name (default: "tjdnet-minimal")
- `--wandb_entity`: WandB entity (default: None)
- `--disable_wandb`: Disable WandB logging (default: False)

## Experiment Directory Structure

```
experiments/
├── <experiment_name_1>/
│   ├── best.ckpt
│   ├── last.ckpt
│   └── wandb/
├── <experiment_name_2>/
│   ├── best.ckpt
│   ├── last.ckpt
│   └── wandb/
└── ...
```

## Differences from main.py

1. **Simpler interface**: No Lightning CLI complexity
2. **Direct argument parsing**: All arguments are command-line flags
3. **Same experiment naming**: Uses identical experiment naming logic
4. **Same auto-resume**: Identical checkpoint resumption behavior
5. **Same WandB integration**: Compatible WandB logging

## Running Examples

See `example_usage.py` for comprehensive examples of different usage patterns.

```bash
# Run all examples
python example_usage.py

# Or run individual commands
python main-cli.py --help  # See all available options
``` 