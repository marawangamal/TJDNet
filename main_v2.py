import lightning as L
from lightning.pytorch.tuner import Tuner
from utils.lmodules_v2 import LModel, LDataModule
from lightning.pytorch.cli import LightningCLI


class MyLightningCLI(LightningCLI):
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
