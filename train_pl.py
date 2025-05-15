"""Train a TJD model using PyTorch Lightning.

Example:
    python train_pl.py --model distilbert/distilgpt2 --batch_size 1 --seq_len 8

"""

from argparse import Namespace
import os.path as osp

import lightning as L
from torch import optim
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

from dataloaders import DATASET_LOADERS
from tjdnet.models.tjd import TJD
from utils.helpers import (
    get_auto_tokenizer,
    get_model_and_tokenizer,
    get_model_and_tokenizer_nowrap,
    parse_args,
)
from utils.utils import get_experiment_name

EXPERIMENTS_DIR = "experiments"
SILENT_ARGS = [
    "slurm_job_id",
    "cache_dir",
    "disable_wandb",
    "compute_acc",
    "generate_strategy",
    "generate_steps",
    "logging_strategy",
    "logging_steps",
    "eval_strategy",
    "eval_steps",
    "wandb_project",
    # Evaluation args
    "do_sample",
    "max_new_tokens",
    "top_k",
    "gen_mode",
]


# define the LightningModule
class LModel(L.LightningModule):
    def __init__(self, model: TJD):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        # dict_keys(['input_ids', 'attention_mask', 'labels'])
        output = self.model(**batch)
        self.log("train_loss", output["loss"])
        return output["loss"]

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def main(args):
    # Setup
    L.seed_everything(42)
    filtered_args = Namespace(
        **{k: v for k, v in vars(args).items() if k not in SILENT_ARGS}
    )
    # exp name does not include silent args
    exp_name = get_experiment_name(vars(filtered_args))

    # Model
    model, tokenizer = get_model_and_tokenizer_nowrap(args)
    lmodel = LModel(model=model)

    # Data
    lm_dataset = DATASET_LOADERS[args.dataset](
        tokenizer=tokenizer,
        input_seq_len=args.seq_len,
        max_num_samples=args.max_num_samples,
    )

    # No pad token needed since all samples are of same length
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # we’re doing causal-LM, not masked-LM
        return_tensors="pt",
    )

    train_dataloader = DataLoader(
        lm_dataset["train"],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,  # ← HERE
    )

    # Trainer
    # trainer = L.Trainer(accelerator="cuda", devices=2, strategy=FSDPStrategy())
    trainer = L.Trainer(
        limit_train_batches=100,
        max_epochs=1,
        default_root_dir=osp.join(EXPERIMENTS_DIR, exp_name),
    )
    trainer.fit(lmodel, train_dataloader)


if __name__ == "__main__":
    args = parse_args()
    main(args)
