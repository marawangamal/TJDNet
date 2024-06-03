from typing import List
import argparse
import logging
import os.path as osp
import shutil

import torch
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset

from littjdnet import LitTJDNet
from utils.utils import get_experiment_name


logging.basicConfig(
    format="%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


class SequenceDataset(Dataset):
    """Dataset wrapping lists of sequences."""

    def __init__(
        self,
        non_zero_prob_indices: List[List[int]] = [
            [0, 1, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 1, 0],
        ],
        num_samples: int = 100,
    ) -> None:
        """Initialize the dataset.
        Args:
            non_zero_prob_indices (List[List[int]], optional): List of sequences with non-zero probability. Defaults to [[0, 1, 0, 1], [0, 0, 1, 1], [0, 1, 1, 0]].
            num_samples (int, optional): Number of samples to generate. Defaults to 100.
        """
        self.sequences = [
            non_zero_prob_indices[np.random.randint(len(non_zero_prob_indices))]
            for _ in range(num_samples)
        ]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {"label_ids": self.sequences[idx]}


def collate_fn(batch):
    return {
        "label_ids": torch.tensor([item["label_ids"] for item in batch]),
    }


def main(
    model_name: str,
    lr: float = 5e-5,
    epochs: int = 20,
    batch_size: int = 4,
    checkpoint_dir: str = "checkpoints",
    overwrite: bool = True,
):

    # 0. Create a unique experiment name
    experiment_config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "model_name": model_name,
    }
    experiment_name = get_experiment_name(experiment_config)
    logger.info(f"Experiment configuration\n: {experiment_config}")
    logger.info(f"Checkpoints will be saved to: {checkpoint_dir}/{experiment_name}")

    if osp.exists(osp.join(checkpoint_dir, experiment_name)) and overwrite:
        logger.info("Overwriting existing checkpoints...")
        shutil.rmtree(osp.join(checkpoint_dir, experiment_name))

    # 1. Load data
    dataset = SequenceDataset()
    model_params = {
        "rank": 2,
        "vocab_size": 3,
    }
    train_dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    eval_dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    lit_model = LitTJDNet(model_params=model_params, model_name=model_name, lr=lr)

    tb_logger = TensorBoardLogger(
        osp.join(checkpoint_dir, experiment_name), name="", version=""
    )

    trainer = Trainer(
        max_epochs=epochs,
        log_every_n_steps=1,
        logger=tb_logger,
        gradient_clip_val=1.0,
    )
    trainer.fit(lit_model, train_dataloader, eval_dataloader, ckpt_path="last")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_epochs", type=int, default=50, help="Number of training epochs"
    )

    parser.add_argument(
        "-c",
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints",
    )

    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")

    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    args = parser.parse_args()

    main(
        model_name="basic-tjd-layer",
        epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
    )
