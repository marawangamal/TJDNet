"""Train a TJD model using PyTorch Lightning.

Example:
    python train_pl.py --model distilbert/distilgpt2 --batch_size 1 --seq_len 8 --max_num_samples 10
    python train_pl.py --dataset gsm8k --model meta-llama/Llama-3.2-3B-Instruct --epochs 5 --batch_size 1 --seq_len 8 --lr 5e-5 --model_head base --horizon 1 --rank 1

"""

import os
import os.path as osp
from argparse import Namespace

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

import torch
from torch import optim
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

from dataloaders import DATASET_LOADERS
from tjdnet.distributions._tjdist import TJDist
from utils.helpers import (
    get_auto_tokenizer,
    get_git_info,
    get_model_and_tokenizer,
    get_model_and_tokenizer_nowrap,
)
from utils.arguments_v2 import parse_args
from utils.monitor import calculate_model_memory_breakdown
from utils.pl_callbacks import CUDAMemoryLogger
from utils.utils import get_experiment_name


os.environ["TOKENIZERS_PARALLELISM"] = "false"

EXPERIMENTS_DIR = "experiments"
SILENT_ARGS = [
    "disable_wandb",
    "compute_acc",
    # Evaluation args
    "do_sample",
    "max_new_tokens",
    "top_k",
    "gen_mode",
    # Accelerator
    "accel_strategy",
]


# define the LightningModule
class LModel(L.LightningModule):
    def __init__(self, args: Namespace):
        super().__init__()
        self.args = args
        self.tokenizer = get_auto_tokenizer(args.model)
        self.save_hyperparameters(args)

    def configure_model(self):
        # create all your layers here
        self.model, _ = get_model_and_tokenizer(args)
        self.model.train()

    def training_step(self, batch, batch_idx):
        output = self.model(**batch)
        self.log("train_loss", output["loss"], prog_bar=True, on_epoch=True)
        return output["loss"]

    def validation_step(self, batch, batch_idx):
        output = self.model(**batch)
        self.log("val_loss", output["loss"], prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def get_wandb_logger(exp_name: str):
    git_info = get_git_info()
    suffix = "main" if git_info.get("branch") == "main" else "dev"
    project_name = f"tjdnet-{suffix}"
    wandb_logger = WandbLogger(
        project=project_name,
        name=exp_name,
        resume="allow",
    )
    return wandb_logger


def printo(*args, **kwargs):
    """Print to stdout and stderr."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(*args, **kwargs)


def main(args):
    # Setup
    L.seed_everything(42)
    filtered_args = Namespace(
        **{k: v for k, v in vars(args).items() if k not in SILENT_ARGS}
    )
    exp_name = get_experiment_name(vars(filtered_args))
    ckpt_path = osp.join(EXPERIMENTS_DIR, exp_name, "last.ckpt")
    if not osp.exists(ckpt_path):
        ckpt_path = None
    print(
        "Training from scratch." if ckpt_path is None else f"Resuming from {ckpt_path}"
    )

    # Model
    lmodel = LModel(args=args)

    # Data
    lm_dataset = DATASET_LOADERS[args.dataset](
        tokenizer=lmodel.tokenizer,
        input_seq_len=args.seq_len,
        max_num_samples=args.max_num_samples,
    )

    # No pad token needed since all samples are of same length
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token

    collator = DataCollatorForLanguageModeling(
        tokenizer=lmodel.tokenizer,
        mlm=False,  # we’re doing causal-LM, not masked-LM
        return_tensors="pt",
    )

    train_dataloader = DataLoader(
        lm_dataset["train"],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=4,
    )
    eval_dataloader = DataLoader(
        lm_dataset["eval"],
        batch_size=args.batch_size,
        collate_fn=collator,
        num_workers=4,
    )

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=osp.join(EXPERIMENTS_DIR, exp_name),
        filename="ckpt-{epoch}-{val_loss:.2f}",  # template for *best* files
        monitor="val_loss",  # metric to track
        mode="min",  # "max" for accuracy / "min" for loss
        save_top_k=1,  # keep only the single best model
        save_last=True,  # ALSO keep a rolling 'last.ckpt'
    )
    memory_cb = CUDAMemoryLogger()

    # Trainer
    # trainer = L.Trainer(accelerator="cuda", devices=2, strategy=FSDPStrategy())
    policy = {LlamaDecoderLayer}
    strategy = {"auto": "auto", "fsdp": FSDPStrategy(auto_wrap_policy=policy)}[
        args.accel_strategy
    ]
    trainer = L.Trainer(
        # fast_dev_run=args.fast_dev_run,
        # overfit_batches=1,
        # accelerator=args.accel,
        strategy=strategy,
        max_epochs=args.epochs + 2,
        default_root_dir=osp.join(EXPERIMENTS_DIR, exp_name),
        callbacks=[checkpoint_cb, memory_cb],
        logger=get_wandb_logger(exp_name),
    )

    # Memory breakdown
    params = sum(p.numel() for p in lmodel.parameters())
    params_memory_gb = params * 4 / (1024**3)
    printo("\n===== MEMORY BREAKDOWN =====")
    printo(f"Params: {params / 1e9:.3f} B parameters │  {params_memory_gb:.2f} GB ")
    printo("==============================\n")

    trainer.fit(
        lmodel,
        train_dataloader,
        eval_dataloader,
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
