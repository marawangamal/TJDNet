import os.path as osp

import lightning as L
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from utils.experiment_naming import get_experiment_name
from utils.lmodules_v2 import LModel, LDataModule
from lightning.pytorch.cli import LightningCLI


EXPERIMENTS_DIR = "experiments_v2"

# def get_experiment_name(config):
#     # Takes model config, data config and select trainer config


class MyLightningCLI(LightningCLI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Create config-based experiment name
        self.experiment_name = get_experiment_name(
            {
                **{k: self.config["trainer"][k] for k in ["max_epochs"]},
                **self.config["model"],
                **self.config["data"],
            }
        )
        checkpoint_cb = ModelCheckpoint(
            dirpath=osp.join(EXPERIMENTS_DIR, self.experiment_name),
            filename="best",
            monitor="eval_loss",
            mode="min",
            save_top_k=1,
        )
        self.trainer.callbacks.append(checkpoint_cb)

        # # Always add WandB logger if not already set
        # if self.trainer.logger is None or not isinstance(
        #     self.trainer.logger, WandbLogger
        # ):
        #     wandb_logger = WandbLogger(
        #         project="tjdnet-dev",  # Change to your project name
        #         name=None,  # Auto-generate run name
        #         save_dir="./wandb_logs",
        #     )
        #     self.trainer.logger = wandb_logger
        #     print("📊 Added WandB logging automatically")

    def add_arguments_to_parser(self, parser):
        parser.link_arguments("model.model", "data.model")

    def before_fit(self):
        print("🔍 Running learning rate finder...")
        # Create a fresh trainer just for LR finding
        lr_trainer = L.Trainer(
            accelerator="gpu",
            devices=1,  # Single GPU to avoid distributed issues
            logger=False,
            enable_checkpointing=False,
        )

        tuner = Tuner(lr_trainer)
        lr_finder = tuner.lr_find(self.model, datamodule=self.datamodule)

        if lr_finder:
            suggested_lr = lr_finder.suggestion()
            print(f"🎯 Suggested learning rate: {suggested_lr}")
            self.model.hparams.lr = suggested_lr


def cli_main():
    cli = MyLightningCLI(LModel, LDataModule)


def main() -> None:
    cli_main()


if __name__ == "__main__":
    main()
