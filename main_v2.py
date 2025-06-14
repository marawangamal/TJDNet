import os.path as osp

import lightning as L
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from transformers import AutoTokenizer

from dataloaders import DATASETS
from utils.experiment_naming import get_experiment_name
from utils.lmodules_v2 import LModel, LDataModule
from utils.lightning_callbacks import GenerateCallback

EXPERIMENTS_DIR = "experiments"


# utils/wandb_helpers.py
import wandb
from typing import Optional


def lookup_wandb_id(
    project_name: str, run_name: str, entity: Optional[str] = None
) -> str:
    try:
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
    return wandb.util.generate_id()


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("model.model", "data.model")
        parser.link_arguments("model.dataset", "data.dataset")

    # def before_fit(self):
    #     # Only run LR finder on the main process
    #     if self.trainer.global_rank == 0:
    #         print("🔍 Running learning rate finder...")
    #         lr_trainer = L.Trainer(
    #             accelerator="gpu",
    #             devices=1,
    #             logger=False,
    #             enable_checkpointing=False,
    #         )

    #         tuner = Tuner(lr_trainer)
    #         lr_finder = tuner.lr_find(self.model, datamodule=self.datamodule)

    #         if lr_finder:
    #             suggested_lr = lr_finder.suggestion()
    #             print(f"🎯 Suggested learning rate: {suggested_lr}")
    #             self.model.hparams.lr = suggested_lr
    #     else:
    #         print("🔍 Skipping LR finder on non-main process")

    #     # Wait for all processes to sync
    #     if hasattr(self.trainer.strategy, "barrier"):
    #         self.trainer.strategy.barrier()

    def before_instantiate_classes(self):
        cfg = self.config

        if "test" in cfg:
            generate_cb = GenerateCallback(
                tokenizer=AutoTokenizer.from_pretrained("gpt2"),
            )
            cfg.test.trainer.callbacks = [generate_cb]
            return

        # 1. Experiment naming
        run_name = get_experiment_name(
            {
                **{k: cfg.fit["trainer"][k] for k in ["max_epochs"]},
                **vars(cfg.fit.model),
                **vars(cfg.fit.data),
            }
        )
        run_dir = osp.join(EXPERIMENTS_DIR, run_name)
        cfg.fit.trainer.default_root_dir = run_dir

        # 2. Auto-resume
        ckpt_path = osp.join(EXPERIMENTS_DIR, run_name, "best.ckpt")
        cfg.fit.ckpt_path = ckpt_path if osp.exists(ckpt_path) else None

        # 4. Callbacks
        ckpt_best_cb = ModelCheckpoint(
            dirpath=osp.join(EXPERIMENTS_DIR, run_name),
            filename="best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )
        generate_cb = GenerateCallback(
            tokenizer=AutoTokenizer.from_pretrained(cfg.fit.model.model),
            prompt=DATASETS[cfg.fit.data.dataset](
                tokenizer=AutoTokenizer.from_pretrained(cfg.fit.model.model)
            ).get_sample_prompt(),
        )
        cfg.fit.trainer.callbacks = [ckpt_best_cb, generate_cb]

        # 5. Logger
        project_name = "mtp"
        wandb_logger = WandbLogger(
            project=project_name,
            name=run_name,
            id=lookup_wandb_id(project_name, run_name),
            resume="allow",
            save_dir=osp.join(EXPERIMENTS_DIR, run_name),
        )
        cfg.fit.trainer.logger = wandb_logger


def cli_main() -> None:
    L.seed_everything(42)
    MyLightningCLI(LModel, LDataModule, save_config_kwargs={"overwrite": True})


def main() -> None:
    cli_main()


if __name__ == "__main__":
    main()
