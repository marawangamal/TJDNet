"""
Minimal PyTorch Lightning training script without CLI.

This script provides a simple way to train models using PyTorch Lightning
without the complexity of the Lightning CLI.
"""

import os
import os.path as osp
import argparse
from datetime import timedelta
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import Callback
from transformers import AutoTokenizer

from utils.lmodules import LModel, LDataModule
from utils.lightning_callbacks import GenerateCallback
from utils.experiment_naming import get_experiment_name
from dataloaders import DATASETS

EXPERIMENTS_DIR = "experiments"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train TJDNet models with PyTorch Lightning"
    )

    # Model arguments
    parser.add_argument("--model", type=str, default="gpt2", help="Model name or path")
    parser.add_argument(
        "--train_mode",
        type=str,
        default="lora",
        choices=["lora", "full"],
        help="Training mode",
    )
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--model_head", type=str, default="cp", help="Model head type")
    parser.add_argument("--horizon", type=int, default=1, help="Horizon")
    parser.add_argument("--rank", type=int, default=1, help="Rank")
    parser.add_argument(
        "--positivity_func", type=str, default="sigmoid", help="Positivity function"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument(
        "--gradient_clip_val", type=float, default=1.0, help="Gradient clipping value"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=128, help="Max new tokens for generation"
    )
    parser.add_argument(
        "--do_sample", action="store_true", help="Enable sampling during generation"
    )
    parser.add_argument("--top_k", type=int, default=200, help="Top-k for generation")
    parser.add_argument("--seq_len", type=int, default=128, help="Sequence length")
    parser.add_argument("--dataset", type=str, default="stemp", help="Dataset name")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--gen_mode", type=str, default="draft", help="Generation mode")
    parser.add_argument("--framework", type=str, default="tjd", help="Framework")

    # Data arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--max_num_samples", type=int, default=None, help="Max number of samples"
    )
    parser.add_argument(
        "--max_test_samples", type=int, default=None, help="Max test samples"
    )
    parser.add_argument("--max_tokens", type=int, default=None, help="Max tokens")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument(
        "--template_mode", type=str, default="0_shot", help="Template mode"
    )
    parser.add_argument("--domain_shift", type=str, default="in", help="Domain shift")

    # Trainer arguments
    parser.add_argument("--max_epochs", type=int, default=10, help="Max epochs")
    parser.add_argument("--accelerator", type=str, default="auto", help="Accelerator")
    parser.add_argument("--devices", type=str, default="auto", help="Devices")
    parser.add_argument("--precision", type=int, default=32, help="Precision")
    parser.add_argument(
        "--log_every_n_steps", type=int, default=10, help="Log every n steps"
    )
    parser.add_argument(
        "--val_check_interval",
        type=float,
        default=0.25,
        help="Validation check interval",
    )

    # Experiment arguments
    parser.add_argument(
        "--test_after_fit", action="store_true", help="Run test after fit"
    )
    parser.add_argument(
        "--auto_lr_find", action="store_true", help="Auto find learning rate"
    )
    parser.add_argument(
        "--disable_wandb", action="store_true", help="Disable WandB logging"
    )

    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()

    # Set random seed for reproducibility
    L.seed_everything(42)

    # Model configuration
    model_config = {
        "model": args.model,
        "train_mode": args.train_mode,
        "lora_rank": args.lora_rank,
        "model_head": args.model_head,
        "horizon": args.horizon,
        "rank": args.rank,
        "positivity_func": args.positivity_func,
        "lr": args.lr,
        "warmup_steps": args.warmup_steps,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "top_k": args.top_k,
        "seq_len": args.seq_len,
        "dataset": args.dataset,
        "debug": args.debug,
        "gen_mode": args.gen_mode,
        "framework": args.framework,
    }

    # Data configuration
    data_config = {
        "model": args.model,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "dataset": args.dataset,
        "max_num_samples": args.max_num_samples,
        "max_test_samples": args.max_test_samples,
        "max_tokens": args.max_tokens,
        "num_workers": args.num_workers,
        "template_mode": args.template_mode,
        "domain_shift": args.domain_shift,
    }

    # Trainer configuration
    trainer_config = {
        "max_epochs": args.max_epochs,
        "accelerator": args.accelerator,
        "devices": args.devices,
        "precision": args.precision,
        "gradient_clip_val": args.gradient_clip_val,
        "log_every_n_steps": args.log_every_n_steps,
        "val_check_interval": args.val_check_interval,
    }

    # Generate experiment name
    experiment_config = {
        **{
            k: trainer_config[k]
            for k in ["max_epochs", "gradient_clip_val", "precision"]
        },
        **model_config,
        **data_config,
    }
    run_name = get_experiment_name(experiment_config)
    run_dir = osp.join(EXPERIMENTS_DIR, run_name)

    # Set default root dir for trainer
    trainer_config["default_root_dir"] = run_dir

    # Auto-resume if checkpoint exists
    last_ckpt_path = osp.join(EXPERIMENTS_DIR, run_name, "last.ckpt")
    best_ckpt_path = osp.join(EXPERIMENTS_DIR, run_name, "best.ckpt")
    ckpt_path = None
    if osp.exists(last_ckpt_path):
        print(f"[INFO] Resuming from last checkpoint: {last_ckpt_path}")
        ckpt_path = last_ckpt_path
    elif osp.exists(best_ckpt_path):
        print(f"[INFO] Resuming from best checkpoint: {best_ckpt_path}")
        ckpt_path = best_ckpt_path

    # Initialize model and data module
    model = LModel(**model_config)
    datamodule = LDataModule(**data_config)

    # Set up callbacks
    callbacks: list[Callback] = [
        ModelCheckpoint(
            dirpath=osp.join(EXPERIMENTS_DIR, run_name),
            filename="best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        ),
        ModelCheckpoint(
            dirpath=osp.join(EXPERIMENTS_DIR, run_name),
            filename="last",
            every_n_train_steps=None,
            train_time_interval=timedelta(minutes=30),  # Save every 30 minutes
        ),
    ]

    # Add generation callback if dataset supports it
    if model_config["dataset"] in DATASETS:
        tokenizer = AutoTokenizer.from_pretrained(model_config["model"])
        dataset_instance = DATASETS[model_config["dataset"]](tokenizer=tokenizer)
        if hasattr(dataset_instance, "get_sample_prompt"):
            generate_callback = GenerateCallback(
                prompt=dataset_instance.get_sample_prompt(),
            )
            callbacks.append(generate_callback)

    # Set up logger
    logger = None
    if not args.disable_wandb and os.getenv("WANDB_MODE") != "disabled":
        try:
            logger = WandbLogger(
                project="mtp",
                name=run_name,
                save_dir=osp.join(EXPERIMENTS_DIR, run_name),
            )
        except Exception as e:
            print(f"Warning: Could not initialize wandb logger: {e}")
            logger = None

    # Initialize trainer
    trainer = L.Trainer(
        **trainer_config,
        callbacks=callbacks,
        logger=logger,
    )

    # Auto LR finder if requested
    if args.auto_lr_find and trainer.global_rank == 0 and ckpt_path is None:
        print("🔍 Running learning rate finder...")
        from lightning.pytorch.tuner.tuning import Tuner

        lr_trainer = L.Trainer(
            accelerator="gpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            precision=trainer_config.get("precision", 32),
        )
        tuner = Tuner(lr_trainer)
        lr_finder = tuner.lr_find(
            model,
            datamodule=datamodule,
            min_lr=1e-6,
            max_lr=1e-3,
            num_training=50,
        )
        if lr_finder:
            suggested_lr = lr_finder.suggestion()
            print(f"  ↳ suggested lr = {suggested_lr:.2e}")
            # Update the model's learning rate
            model.hparams["lr"] = suggested_lr

    # Train the model
    print(f"Starting training for experiment: {run_name}")
    trainer.fit(model, datamodule, ckpt_path=ckpt_path)

    # Test the model if requested
    if args.test_after_fit:
        print("Running test...")
        trainer.test(model, datamodule)

    print("Training completed!")


if __name__ == "__main__":
    main()
