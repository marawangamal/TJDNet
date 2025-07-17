"""
Main entry point for the Lightning CLI application.

This script sets up the Lightning CLI for training and testing models, including
experiment management, learning rate finding, and integration with Weights & Biases (wandb).

Example usage:
    python main.py fit --model.model HuggingFaceTB/SmolLM-135M --model.model_head cp_condl --model.rank 1 --model.horizon 2 --trainer.max_epochs 1 --data.batch_size 1 --data.seq_len 10 --data.max_num_samples 10 --trainer.max_epochs 1

"""

# import os

# os.environ["WANDB_MODE"] = "disabled"


import argparse
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


# # TODO: save/load to a <EXPERIMENTS_DIR>/<run_name>/wandb_id.txt file
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


#     def _load_model_config(self, ckpt_path: str):
#         """Load the model config from yaml file."""
#         print(f"[INFO] Loading model config from {ckpt_path}")
#         config_path = osp.join(osp.dirname(ckpt_path), "config.yaml")
#         with open(config_path, "r") as f:
#             config = yaml.safe_load(f)

#         # Set model configuration using the parser
#         for key, value in config["model"].items():
#             setattr(self.config.test.model, key, value)


def main() -> None:
    p = argparse.ArgumentParser()
    # model
    p.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM-135M")
    p.add_argument("--model_head", type=str, default="cp_condl")
    p.add_argument("--rank", type=int, default=1)
    p.add_argument("--horizon", type=int, default=2)
    p.add_argument("--lr", type=float, default=None)
    # trainer
    p.add_argument("--max_epochs", type=int, default=10)
    p.add_argument("--gradient_clip_val", type=float, default=1.0)
    p.add_argument("--ckpt_path", type=str, default=None)
    # data
    p.add_argument("--dataset", type=str, default="countdown")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_num_samples", type=int, default=10000)
    p.add_argument("--seq_len", type=int, default=10)
    # test/eval
    p.add_argument("--test", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=128)

    args = p.parse_args()
    L.seed_everything(42)

    # setup
    run_name = get_experiment_name(vars(args))

    # data
    lit_data = LDataModule(
        model=args.model,  # used for tokenizer
        dataset=args.dataset,
        batch_size=args.batch_size,
        max_num_samples=args.max_num_samples,
        seq_len=args.seq_len,
    )

    # model
    if args.ckpt_path is not None:
        # BUG: will this load the config too?
        lit_model = LModel.load_from_checkpoint(args.ckpt_path)
    else:
        lit_model = LModel(
            model=args.model,
            model_head=args.model_head,
            rank=args.rank,
            horizon=args.horizon,
            dataset=args.dataset,  # used for parsing/chat_template
        )

    # 2. Resume from ckpt
    # priority:
    # - if ckpt_path specified, use it
    # - if <EXPERIMENTS_DIR>/<run_name> exists, use the last or best checkpoint
    if args.ckpt_path is not None:
        ckpt_path_for_resume = args.ckpt_path
        print(f"[INFO] Resuming from specified checkpoint: {args.ckpt_path}")
    else:
        last_ckpt_path = osp.join(EXPERIMENTS_DIR, run_name, "last.ckpt")
        best_ckpt_path = osp.join(EXPERIMENTS_DIR, run_name, "best.ckpt")
        if osp.exists(last_ckpt_path):
            print(f"[INFO] Resuming from last checkpoint: {last_ckpt_path}")
            ckpt_path_for_resume = last_ckpt_path
        elif osp.exists(best_ckpt_path):
            print(f"[INFO] Resuming from best checkpoint: {best_ckpt_path}")
            ckpt_path_for_resume = best_ckpt_path

    # 3. Add callbacks
    ckpt_best_cb = ModelCheckpoint(
        dirpath=osp.join(EXPERIMENTS_DIR, run_name),
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    ckpt_time_cb = ModelCheckpoint(
        dirpath=osp.join(EXPERIMENTS_DIR, run_name),
        filename="last",
        every_n_train_steps=None,
        train_time_interval=timedelta(minutes=30),  # Save every 30 minutes
    )

    # sample from model on occassion
    generate_cb = GenerateCallback(
        prompt=DATASETS[args.dataset](
            tokenizer=AutoTokenizer.from_pretrained(args.model)
        ).get_sample_prompt(),
    )

    # trainer
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=[ckpt_best_cb, generate_cb, ckpt_time_cb],
        logger=WandbLogger(
            project="mtp",
            name=run_name,
            id=lookup_wandb_id("mtp", run_name),
            resume="allow",
            save_dir=osp.join(EXPERIMENTS_DIR, run_name),
        ),
        auto_lr_find=True,
    )

    if args.test:
        # update other test args
        test_kwargs = {
            "horizon": args.horizon,
            "dataset": args.dataset,
        }
        lit_model.hparams.update(test_kwargs)
        trainer.test(lit_model, lit_data, ckpt_path=ckpt_path_for_resume)
    else:
        if args.lr is None:
            trainer.tune(lit_model)
        trainer.fit(lit_model, lit_data, ckpt_path=ckpt_path_for_resume)
        trainer.test(lit_model, lit_data, ckpt_path=ckpt_path_for_resume)


if __name__ == "__main__":
    main()
