"""
Simplified Lightning CLI script following official documentation recommendations.
"""

import lightning as L
from lightning.pytorch.cli import LightningCLI

from utils.lmodules import LModel, LDataModule
from jsonargparse import namespace_to_dict  # ships with Lightning


class SimpleLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # Link model and data arguments (same as main.py)
        parser.link_arguments("model.model", "data.model")
        parser.link_arguments("data.dataset", "model.dataset")


def main():
    """Main entry point with minimal configuration."""
    L.seed_everything(42)  # Same as main.py
    # cli = SimpleLightningCLI(
    #     LModel,
    #     LDataModule,
    #     save_config_kwargs={"overwrite": True},
    #     load_from_checkpoint_support=True,
    # )
    cli = SimpleLightningCLI(LModel, LDataModule, run=False)  # True by default
    cli.trainer.fit(cli.model)
    cli.trainer.test(cli.model)


if __name__ == "__main__":
    main()
