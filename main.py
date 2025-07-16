"""
Main entry point for the Lightning CLI application.

This script sets up the Lightning CLI for training and testing models, including
experiment management, learning rate finding, and integration with Weights & Biases (wandb).

Example usage:
    python main.py fit --model.model distilbert/distilgpt2 --model.model_head cp_condl --model.rank 1 --model.horizon 2 --trainer.max_epochs 1 --data.batch_size 1 --data.seq_len 10 --data.max_num_samples 10 --trainer.max_epochs 1

"""

import os

os.environ["WANDB_MODE"] = "disabled"


import os.path as osp
from datetime import timedelta

import lightning as L
from lightning.pytorch.tuner.tuning import Tuner
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from transformers import AutoTokenizer
import yaml

from dataloaders import DATASETS
from utils.experiment_naming import get_experiment_name
from utils.lmodules import LModel, LDataModule
from utils.lightning_callbacks import GenerateCallback

EXPERIMENTS_DIR = "experiments"


# utils/wandb_helpers.py
import wandb
from typing import Optional


# TODO: save/load to a <EXPERIMENTS_DIR>/<run_name>/wandb_id.txt file
def lookup_wandb_id(
    project_name: str, run_name: str, entity: Optional[str] = None
) -> str:
    """Lookup the WandB run ID for a given project and run name.

    Args:
        project_name (str): The name of the WandB project.
        run_name (str): The name of the run to look up.
        entity (Optional[str], optional): The WandB entity (team or user). Defaults to None.

    Returns:
        str: The WandB run ID.
    """
    try:
        if not osp.exists(osp.join(EXPERIMENTS_DIR, run_name)):
            print(f"[wandb] No existing run found for {run_name} in {EXPERIMENTS_DIR}.")
            return wandb.util.generate_id()  # type: ignore

        api = wandb.Api(timeout=15)
        # "entity/project" or just "project" if entity=None
        path = f"{entity}/{project_name}" if entity else project_name
        for run in api.runs(path):
            if run.name == run_name:
                print(f"[wandb] Found existing run: {run.id} ({run.name})")
                return run.id
    except Exception as e:  # network error, project not found, etc.
        print(f"[wandb] lookup failed ({e}); creating a new run id.")

    # No existing run → make a fresh ID (8 chars, collision-safe)
    return wandb.util.generate_id()  # type: ignore


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("model.model", "data.model")
        parser.link_arguments("data.dataset", "model.dataset")
        parser.add_argument("--test_after_fit", action="store_true")
        parser.add_argument("--auto_lr_find", action="store_true", default=False)

    def after_fit(self):
        if self.config.fit.get("test_after_fit"):
            print("[INFO] Running test after fit...")
            self.trainer.test(ckpt_path="best")

    def before_fit(self):
        # Only run LR finder on the main process
        if not self.config.fit["auto_lr_find"]:
            print("🔍 Skipping LR finder as per configuration")
            return

        if self.trainer.global_rank == 0 and self.trainer.ckpt_path is None:
            print("🔍 Running learning rate finder...")
            lr_trainer = L.Trainer(
                accelerator="gpu",
                devices=1,
                logger=False,
                enable_checkpointing=False,
                precision=self.config.fit.trainer.get("precision", 32),
            )

            tuner = Tuner(lr_trainer)
            lr_finder = tuner.lr_find(
                self.model,
                datamodule=self.datamodule,
                min_lr=1e-6,
                max_lr=1e-3,
                num_training=50,
            )

            if lr_finder:
                suggested_lr = lr_finder.suggestion()
                print(f"  ↳ suggested lr = {suggested_lr:.2e}")
                self.model.hparams.lr = suggested_lr

    # TODO: if test, parse the config
    def before_instantiate_classes(self):
        cfg = self.config

        if "test" in cfg:
            self._load_model_config(cfg.test.ckpt_path)
            return

        # 1. Set root dir
        run_name = get_experiment_name(
            {
                **{
                    k: cfg.fit["trainer"][k]
                    for k in ["max_epochs", "gradient_clip_val", "precision"]
                },
                **vars(cfg.fit.model),
                **vars(cfg.fit.data),
            }
        )
        run_dir = osp.join(EXPERIMENTS_DIR, run_name)
        cfg.fit.trainer.default_root_dir = run_dir

        # 2. Auto-resume if <EXPERIMENTS_DIR>/<run_name> exists
        last_ckpt_path = osp.join(EXPERIMENTS_DIR, run_name, "last.ckpt")
        best_ckpt_path = osp.join(EXPERIMENTS_DIR, run_name, "best.ckpt")
        if osp.exists(last_ckpt_path):
            print(f"[INFO] Resuming from last checkpoint: {last_ckpt_path}")
            cfg.fit.ckpt_path = last_ckpt_path
        elif osp.exists(best_ckpt_path):
            print(f"[INFO] Resuming from best checkpoint: {best_ckpt_path}")
            cfg.fit.ckpt_path = best_ckpt_path

        # 3. Add callbacks
        ckpt_best_cb = ModelCheckpoint(
            dirpath=osp.join(EXPERIMENTS_DIR, run_name),
            filename="best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )
        generate_cb = GenerateCallback(
            prompt=DATASETS[cfg.fit.data.dataset](
                tokenizer=AutoTokenizer.from_pretrained(cfg.fit.model.model)
            ).get_sample_prompt(),
        )
        ckpt_time_cb = ModelCheckpoint(
            dirpath=osp.join(EXPERIMENTS_DIR, run_name),
            filename="last",
            every_n_train_steps=None,
            train_time_interval=timedelta(minutes=30),  # Save every 30 minutes
        )

        if cfg.fit.trainer.callbacks is None:
            cfg.fit.trainer.callbacks = []
        cfg.fit.trainer.callbacks.extend([ckpt_best_cb, generate_cb, ckpt_time_cb])

        # 4. Add logger
        project_name = "mtp"
        wandb_logger = WandbLogger(
            project=project_name,
            name=run_name,
            id=lookup_wandb_id(project_name, run_name),
            resume="allow",
            save_dir=osp.join(EXPERIMENTS_DIR, run_name),
        )
        cfg.fit.trainer.logger = wandb_logger

    def _load_model_config(self, ckpt_path: str):
        """Load the model config from yaml file."""
        print(f"[INFO] Loading model config from {ckpt_path}")
        config_path = osp.join(osp.dirname(ckpt_path), "config.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Set model configuration using the parser
        for key, value in config["model"].items():
            setattr(self.config.test.model, key, value)


def cli_main() -> None:
    L.seed_everything(42)
    MyLightningCLI(LModel, LDataModule, save_config_kwargs={"overwrite": True})


def main() -> None:
    cli_main()


if __name__ == "__main__":
    main()
