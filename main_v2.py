import os
import lightning as L
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import SaveConfigCallback
from utils.experiment_naming import get_experiment_name
from utils.lmodules_v2 import LModel, LDataModule
from lightning.pytorch.cli import LightningCLI


EXPERIMENTS_DIR = "experiments_v2"


class MyLightningCLI(LightningCLI):
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

        # Configure checkpointing
        # print("🔧 Configuring checkpoint callback...")
        # self.experiment_name = get_experiment_name(
        #     {
        #         **{k: self.config["fit"]["trainer"][k] for k in ["max_epochs"]},
        #         **vars(self.config["fit"]["model"]),
        #         **vars(self.config["fit"]["data"]),
        #     }
        # )
        # ckpt_dir = os.path.join(EXPERIMENTS_DIR, self.experiment_name)
        # os.makedirs(ckpt_dir, exist_ok=True)
        # print(f"📂 Checkpoints will be saved to: {ckpt_dir}")

        # # 1. Model Checkpoints
        # checkpoint_cb = ModelCheckpoint(
        #     dirpath=ckpt_dir,
        #     filename="best",
        #     monitor="eval_loss",
        #     mode="min",
        #     save_top_k=1,
        # )
        # self.trainer.callbacks.append(checkpoint_cb)

        # # 2. Config
        # self.trainer.callbacks.append(config_callback)


def cli_main():
    # Setup
    L.seed_everything(42)  # For reproducibility
    cli = MyLightningCLI(LModel, LDataModule)


def main() -> None:
    cli_main()


if __name__ == "__main__":
    main()
