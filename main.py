"""Train a TJD model using PyTorch Lightning.

Example:
    python main.py train  --model distilbert/distilgpt2 --batch_size 1 --seq_len 8 --max_num_samples 10 --gen_mode mixed --epochs 1
    python main.py train --accel_strategy fsdp --dataset gsm8k --model meta-llama/Llama-3.2-3B-Instruct --epochs 5 --max_num_samples 10 --batch_size 1 --seq_len 8 --lr 5e-5 --model_head base --horizon 1 --rank 1

"""

from ast import Name
from collections import defaultdict
import os
import os.path as osp
from argparse import Namespace
import subprocess
from typing import List, Literal, Optional, Union

import torch
from torch import optim
from torch.utils.data import DataLoader
from wandb.util import generate_id

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers import DataCollatorForLanguageModeling
from transformers import get_linear_schedule_with_warmup


from dataloaders import DATASETS
from tjdnet.distributions._tjdist import TJDist
from tjdnet.models.tjd import TJD, TJDGenerationConfig
from utils.average_meter import AverageMeter
from utils.helpers import get_auto_tokenizer, get_git_info, get_model_and_tokenizer
from utils.lightning_callbacks.generate import GenerateCallback
from utils.experiment_naming import get_experiment_name
from utils.arguments import parse_args

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
    "test",
    "cmd",
    "wandb_id",
    "experiment_name",
    "slurm_job_id",
    "group_id",
    "group_level",
    "extend",
    "lookup",
    "delete_ckpt",
]

PROSPECT_FLAG_FILENAME = ".prospect"
BEST_FLAG_FILENAME = ".best"
TEST_FILENAME = "test_results.txt"


#################################################################
#                       Lightning Classes                       #
#################################################################


# define the LightningModule
class LModel(L.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.args = Namespace(**kwargs)
        self.tokenizer = get_auto_tokenizer(self.args.model)
        self.model: TJD = None  # type: ignore
        self.dataset = DATASETS[self.args.dataset](tokenizer=self.tokenizer)
        self.eos = self.tokenizer.eos_token
        self.test_ameter = AverageMeter()
        self.save_hyperparameters(kwargs)

    def configure_model(self):
        # IMPORTANT: This function must be idempotent (i.e., calling it multiple times should not change self.model)
        if self.model is None:  # Model might be already created in load_from_checkpoint
            # self.model, _ = get_model_and_tokenizer(self.args)
            self.model, _ = get_model_and_tokenizer(self.args)

    def training_step(self, batch, batch_idx):
        # check if any ids are negative
        output = self.model(**batch)
        # === TJD model
        loss = output["loss"]
        # === HF model
        # loss = output.loss
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            output = self.model(**batch)
        # === TJD model
        loss = output["loss"]
        # === HF model
        # loss = output.loss
        self.log("eval_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):

        gen_modes: List[Literal["draft", "base", "speculative"]] = (
            [self.args.gen_mode]
            if self.args.gen_mode in ["base", "draft", "speculative"]
            else ["base", "draft", "speculative"]
        )

        for gen_mode in gen_modes:
            outputs, ardict = self.model.generate(
                generation_config=TJDGenerationConfig(
                    max_new_tokens=self.args.max_new_tokens,
                    do_sample=self.args.do_sample,
                    top_k=self.args.top_k,
                    eos_token_id=int(self.tokenizer.eos_token_id),  # type: ignore
                    gen_mode=gen_mode,
                ),
                **batch,
            )

            # Compute accuracy
            y_pred_str = self.tokenizer.batch_decode(outputs)
            y_pred = torch.tensor(
                [self.dataset.parse_answer(y) for y in y_pred_str],  # type: ignore
                device=outputs.device,
            )
            corr = (y_pred == batch["labels"]).float().sum()
            self.test_ameter.update(corr, len(batch["input_ids"]))
            self.log(f"test_acc_{gen_mode}", corr, prog_bar=True)
            if gen_mode == "speculative":
                tokens_accepted = ardict["tokens_accepted"]
                tokens_generated = ardict["tokens_generated"]
                self.test_ameter.update(
                    tokens_accepted / tokens_generated if tokens_generated > 0 else 0,
                    ardict["tokens_generated"],
                )

    def on_test_epoch_end(self):
        self.log("test_acceptance_rate", self.test_ameter.avg, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.args.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return [optimizer], [scheduler]

    def configure_gradient_clipping(
        self,
        optimizer,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[str] = None,
    ) -> None:
        assert gradient_clip_algorithm in ("norm", None), gradient_clip_algorithm
        if gradient_clip_val is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip_val)

    # === Debug (memory) ===
    def on_before_zero_grad(self, *args, **kwargs):
        self._log_memory("before_zero_grad")
        return super().on_before_zero_grad(*args, **kwargs)

    def _log_memory(self, phase: str):
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**3
            # reserved = torch.cuda.memory_reserved() / 1024**3
            peak = torch.cuda.max_memory_allocated() / 1024**3
            dct = {"allocated": alloc, "peak": peak}
            dct = {f"{phase}_{k}": v for k, v in dct.items()}
            for k, v in dct.items():
                self.log(k, v, prog_bar=True)
            torch.cuda.reset_peak_memory_stats()


class LDataModule(L.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.tokenizer = get_auto_tokenizer(kwargs["model"])
        self.batch_size = kwargs.get("batch_size", 1)
        self.seq_len = kwargs.get("seq_len", 8)
        self.max_num_samples = kwargs.get("max_num_samples", None)
        self.ds_name = kwargs.get("dataset", "stemp")

    def setup(self, stage: str):
        self.lm_dataset = DATASETS[self.ds_name](
            tokenizer=self.tokenizer,
            seq_len=self.seq_len,
            max_num_samples=self.max_num_samples,
        ).load_data()
        self.train_ds, self.eval_ds, self.test_ds = (
            self.lm_dataset["train"],
            self.lm_dataset["eval"],
            self.lm_dataset["test"],
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,  # type: ignore
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collator_train(),
            num_workers=0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.eval_ds,  # type: ignore
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collator_train(),
            num_workers=0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,  # type: ignore
            batch_size=1,
            num_workers=0,
            collate_fn=self._collator_test(),
        )

    def _collator_test(self):
        # return batch[0]
        # stack all tensors across keys
        def collator(batch):
            collated_batch = {}
            for key in batch[0].keys():
                collated_batch[key] = torch.stack([torch.tensor(b[key]) for b in batch])
            return collated_batch

        return collator

    def _collator_train(self):
        collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # we’re doing causal-LM, not masked-LM
            return_tensors="pt",
        )
        return collator


class SafeModelCheckpoint(ModelCheckpoint):
    """ModelCheckpoint that converts all metrics to CPU tensors to avoid device conflicts."""

    def check_monitor_top_k(self, trainer, current):  # type: ignore
        # Convert current to CPU tensor (not float - Lightning expects tensors)
        if isinstance(current, torch.Tensor):
            current = current.detach().cpu()
        else:
            current = torch.tensor(float(current))  # type: ignore

        # Convert stored scores to CPU tensors
        for path in list(self.best_k_models.keys()):
            val = self.best_k_models[path]
            if isinstance(val, torch.Tensor):
                self.best_k_models[path] = val.detach().cpu()
            else:
                self.best_k_models[path] = torch.tensor(float(val))

        return super().check_monitor_top_k(trainer, current)


#################################################################
#                       Helpers                                 #
#################################################################


@rank_zero_only
def get_wandb_logger(exp_name: str, wandb_id=None):
    git_info = get_git_info()
    suffix = "main" if git_info.get("branch") == "main" else "dev"
    project_name = f"tjdnet-{suffix}"
    wandb_logger = WandbLogger(
        project=project_name,
        name=exp_name,
        id=wandb_id,  # Add this line to specify the run ID
        resume="allow",
    )
    return wandb_logger


def printo(*args, **kwargs):
    """Print to stdout and stderr."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(*args, **kwargs)


@rank_zero_only
def generate_wandb_id():
    wandb_id = generate_id()
    print(f"Generated new wandb id: {wandb_id}")
    return wandb_id


# Define a simple identity collator for single samples
def identity_collator(batch):
    # return batch[0]
    # stack all tensors across keys
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = torch.stack([torch.tensor(b[key]) for b in batch])
    return collated_batch


def filter_kwargs(**kwargs):
    # Filter out args that are not needed for the experiment
    return {k: v for k, v in kwargs.items() if k not in SILENT_ARGS}


def print_args(args):
    # Print all args
    line = "=" * 40
    printo(f"{line}\nArgs:\n{line}")
    for k, v in vars(args).items():
        if k not in SILENT_ARGS:
            printo(f"  - {k}: {v}")
    printo(f"{line}\n")


def maybe_update_args(args, exp_name: str):
    # save args to meta_path if it exists
    meta_path = get_meta_path(exp_name)
    if osp.exists(meta_path):
        meta_ckpt = torch.load(meta_path, map_location="cpu")
        meta_ckpt["hyper_parameters"].update(vars(args))
        torch.save(meta_ckpt, meta_path)


#################################################################
#                       CKPT HELPERS                            #
#################################################################


def get_meta_path(exp_name: str):
    # Get the meta path for the experiment
    ckpt_path = osp.join(EXPERIMENTS_DIR, exp_name, "best.ckpt")
    if osp.isdir(ckpt_path):
        meta_path = osp.join(ckpt_path, "meta.pt")
    else:
        meta_path = ckpt_path
    return meta_path


def get_hyper_parameters(exp_name: str):
    # Get the hyper parameters for the experiment
    meta_path = get_meta_path(exp_name)
    if osp.exists(meta_path):
        hparams = torch.load(meta_path, map_location="cpu")["hyper_parameters"]
    else:
        raise ValueError(f"Meta file not found: {meta_path}")
    return hparams


def get_ckpt_file_paths(exp_name: str):
    ckpt_paths = []
    ckpt_path = osp.join(EXPERIMENTS_DIR, exp_name, "best.ckpt")
    if osp.isdir(ckpt_path):
        for item in os.listdir(ckpt_path):
            if item.endswith(".distcp"):
                distcp_path = osp.join(ckpt_path, item)
                ckpt_paths.append(distcp_path)

    elif osp.exists(ckpt_path):
        ckpt_paths.append(ckpt_path)

    if osp.exists(ckpt_path + ".consolidated"):
        ckpt_paths.append(ckpt_path + ".consolidated")

    return ckpt_paths

    # # Get the checkpoint path for the experiment
    # ckpt_path = osp.join(EXPERIMENTS_DIR, exp_name, "best.ckpt")
    # ckpt_path_consolidated = None
    # if osp.isdir(ckpt_path):
    #     if osp.exists(ckpt_path + ".consolidated"):
    #         ckpt_path_consolidated = ckpt_path + ".consolidated"
    #     else:
    #         # Convert
    # subprocess.run(
    #     [
    #         "python",
    #         "-m",
    #         "lightning.pytorch.utilities.consolidate_checkpoint",
    #         str(ckpt_path),
    #     ],
    #     capture_output=True,
    # )
    #     ckpt_path_consolidated = ckpt_path + ".consolidated"
    # return ckpt_path, ckpt_path_consolidated


def make_consolidated_ckpt(exp_name: str):
    ckpt_path = osp.join(EXPERIMENTS_DIR, exp_name, "best.ckpt")
    if osp.isdir(ckpt_path):
        if osp.exists(ckpt_path + ".consolidated"):
            printo(f"Consolidated checkpoint already exists: {ckpt_path}.consolidated")
            return ckpt_path + ".consolidated"
        else:
            # Convert
            printo(f"Consolidating checkpoint: {ckpt_path}")
            subprocess.run(
                [
                    "python",
                    "-m",
                    "lightning.pytorch.utilities.consolidate_checkpoint",
                    str(ckpt_path),
                ],
                capture_output=True,
            )
            return ckpt_path + ".consolidated"
    return ckpt_path


def get_exp_eval_score(exp: str):
    """Load experiment eval_loss and hyperparams."""
    meta_path = get_meta_path(exp)
    meta_ckpt = torch.load(meta_path, map_location="cpu")
    for key, cb in meta_ckpt.get("callbacks", {}).items():
        if "ModelCheckpoint" in key and "eval_loss" in key:
            return cb["best_model_score"]
    return None


def remove_exp_ckpts(exp: str):
    """Remove all checkpoints for the experiment."""
    try:
        ckpt_paths = get_ckpt_file_paths(exp)
        for ckpt_path in ckpt_paths:
            os.remove(ckpt_path)
            printo(f"Deleted {ckpt_path}")
    except Exception as e:
        printo(f"Error deleting checkpoints for {exp}: {e}")
        printo("Make sure the experiment name is correct and the checkpoints exist.")


def lookup_experiments_by_group_id(
    group_id: str, group_level: int = 0, flag_filename=None
) -> List[str]:
    """Find all best checkpoints matching group_id at the specified group level."""
    filtered_exps = []

    for exp in os.listdir(EXPERIMENTS_DIR):
        # Filter: skip if not .best
        if flag_filename is not None:
            flag_file = osp.join(EXPERIMENTS_DIR, exp, flag_filename)
            if not osp.exists(flag_file):
                continue

        # Filter: apply group_id, group_level
        hparams = get_hyper_parameters(exp)
        if "group_id" in hparams:
            exp_group = hparams["group_id"].split("-")[group_level]
            target_group = group_id.split("-")[group_level]

            if exp_group == target_group:
                filtered_exps.append(exp)

    printo(
        f"Found {len(filtered_exps)} checkpoints for group_id: {group_id} @ level {group_level}"
    )
    printo(f"Checkpoints:")
    for exp in filtered_exps:
        printo(f"  - {exp}")

    return filtered_exps


#################################################################
#                       train/test/tag                          #
#################################################################


def train(args, flag_filename=None):
    ##### Setup
    printo("Training model...")
    print_args(args)
    L.seed_everything(42)
    exp_name_filtered = get_experiment_name(filter_kwargs(**vars(args)))
    ckpt_path = osp.join(EXPERIMENTS_DIR, exp_name_filtered, "best.ckpt")
    wandb_id = None

    # If experiment already exists, check if it has completed.
    meta_path = get_meta_path(exp_name_filtered)
    if osp.exists(meta_path):
        meta_ckpt = torch.load(meta_path, map_location="cpu")
        if meta_ckpt.get("epoch", 0) >= args.epochs - 1:
            printo(
                f"Experiment {exp_name_filtered} already completed {meta_ckpt.get('epochs', 0)} epochs. Skipping."
            )
            return

    if len(get_ckpt_file_paths(exp_name_filtered)) > 0:
        wandb_id = torch.load(exp_name_filtered, map_location="cpu")[
            "hyper_parameters"
        ]["wandb_id"]
        printo(f"Found checkpoint @ {ckpt_path}, wandb ID: {wandb_id}")
    else:
        ckpt_path = None
        wandb_id = generate_wandb_id()
        args.wandb_id = wandb_id
        args.experiment_name = exp_name_filtered
        printo("Training from scratch.")

    maybe_update_args(args, exp_name_filtered)

    ##### End of Setup

    printo(f"GROUP ID: {args.group_id}")
    wandb_logger = get_wandb_logger(exp_name_filtered, wandb_id=wandb_id)

    # Model
    lmodel = LModel(**vars(args))

    # Data
    ldata = LDataModule(**vars(args))

    # Callbacks
    checkpoint_cb = SafeModelCheckpoint(
        dirpath=osp.join(EXPERIMENTS_DIR, exp_name_filtered),
        filename="best",
        monitor="eval_loss",
        mode="min",
        save_top_k=1,
    )
    generate_cb = GenerateCallback(prompt=DATASETS[args.dataset].get_sample_prompt())

    # Memory breakdown
    params = sum(p.numel() for p in lmodel.parameters())
    params_memory_gb = params * 4 / (1024**3)
    printo("\n===== MEMORY BREAKDOWN =====")
    printo(f"Params: {params / 1e9:.3f} B parameters │  {params_memory_gb:.2f} GB ")
    printo("==============================\n")

    # Train
    policy = {LlamaDecoderLayer, GPT2Block, TJDist}
    strategy = {
        "auto": "auto",
        "ddp": "ddp",
        "fsdp": FSDPStrategy(
            auto_wrap_policy=policy,
            sharding_strategy="FULL_SHARD",
            # mixed_precision=MixedPrecision(param_dtype=torch.bfloat16),
            # cpu_offload=True,
            # activation_checkpointing_policy={TJDist},
            # limit_all_gathers=True,  # Important for evaluation
            state_dict_type="sharded",
        ),
    }[args.accel_strategy]

    trainer = L.Trainer(
        fast_dev_run=args.fast_dev_run,  # for debugging
        strategy=strategy,
        max_epochs=args.epochs,
        default_root_dir=osp.join(EXPERIMENTS_DIR, exp_name_filtered),
        callbacks=(
            [checkpoint_cb, generate_cb]
            if args.accel_strategy != "fsdp"
            else checkpoint_cb
        ),
        logger=wandb_logger,
        # precision="bf16-mixed",
        accumulate_grad_batches=args.accum_grad_batches,
        gradient_clip_val=1,
        precision=args.precision,
    )

    trainer.fit(
        lmodel,
        datamodule=ldata,
        ckpt_path=ckpt_path,
    )
    if not args.accel_strategy == "fsdp":
        trainer.test(
            ckpt_path="best",
            datamodule=ldata,
        )
    if torch.cuda.is_available():
        trainer.print(torch.cuda.memory_summary())

    # Save the model
    if flag_filename:
        flag_path = osp.join(EXPERIMENTS_DIR, exp_name_filtered, flag_filename)
        with open(flag_path, "w") as f:
            f.write(f"Best extended model")
        printo(f"Saved flag @ {flag_path}")
    else:
        printo("No flag filename provided, skipping flag save.")

    if args.delete_ckpt:
        # Delete model ckpt to free disk space
        remove_exp_ckpts(exp_name_filtered)
        printo(f"Deleted checkpoints for {exp_name_filtered}")


def test(exp_name: str, remove_ckpt=True, test_filename=TEST_FILENAME, **kwargs):
    # Check if the experiment exists
    if osp.exists(osp.join(EXPERIMENTS_DIR, exp_name, test_filename)):
        printo(f"Test results already exist for {exp_name}. Skipping.")
        return

    printo(f"\n=== Testing {exp_name} ===")
    ckpt_path = make_consolidated_ckpt(exp_name)

    # Load model and setup  # cfg=exp_name
    lmodel = LModel.load_from_checkpoint(ckpt_path)
    exp_args = Namespace(**lmodel.hparams)

    # Setup trainer
    generate_cb = GenerateCallback(
        prompt=DATASETS[exp_args.dataset].get_sample_prompt()
    )

    logger = get_wandb_logger(exp_args.experiment_name, wandb_id=exp_args.wandb_id)
    trainer = L.Trainer(
        accelerator="gpu", devices=1, callbacks=[generate_cb], logger=logger
    )

    # Test
    overrideable_args = [
        "max_new_tokens",
        "do_sample",
        "top_k",
        "gen_mode",
    ]
    kwargs = {k: v for k, v in kwargs.items() if k in overrideable_args}
    ekwargs = {**vars(exp_args), **kwargs}
    ldata = LDataModule(**ekwargs)
    test_results = trainer.test(lmodel, datamodule=ldata)

    # Save test results
    test_results_path = osp.join(EXPERIMENTS_DIR, exp_name, test_filename)
    with open(test_results_path, "w") as f:
        f.write(f"Test results for {exp_name}:\n")
        for key, value in test_results[0].items():
            f.write(f"{key}: {value}\n")

    if remove_ckpt:
        remove_exp_ckpts(exp_name)
        printo(f"Deleted checkpoints for {exp_name}")


def tag(args, flag_filename=PROSPECT_FLAG_FILENAME):
    """Tag the best model in each group."""
    exps = lookup_experiments_by_group_id(args.group_id, args.group_level)

    # Get exp losses
    exp_losses = dict()
    for exp_name in exps:
        exp_loss = get_exp_eval_score(exp_name)
        hparams = get_hyper_parameters(exp_name)
        if exp_losses is None:
            printo(f"Skipping {exp_name} - no eval_loss")
            continue
        exp_losses[exp_name] = (exp_loss, hparams)

    # Apply grouping
    groups = defaultdict(dict)
    if args.group_by:
        for exp_name, (loss, hparams) in exp_losses.items():
            key_parts = [
                f"{param}={hparams.get(param, 'unknown')}" for param in args.group_by
            ]
            key = ", ".join(key_parts)
            groups[key][exp_name] = loss
    else:
        # If no grouping specified, treat all experiments as one group
        groups["all"] = {exp_name: loss for _, (loss, _) in exp_losses.items()}

    # Best within each group
    best_exps = set()
    for group_name, group_exps in groups.items():

        if not group_exps:
            printo(f"No experiments in group {group_name}")
            continue

        # Find experiment with minimum loss (assuming lower is better)
        best_exp = min(group_exps.items(), key=lambda x: x[1])[0]

        best_exps.add(best_exp)
        printo(f"Best model in group {group_name}: {best_exp}")

        # Tag best model (i.e., create .best file)
        best_path = osp.join(EXPERIMENTS_DIR, best_exp, flag_filename)
        with open(best_path, "w") as f:
            f.write(f"Best model in group {group_name}")

    # Remove .best files from other models
    for exp_name in exps:
        if exp_name not in best_exps:
            best_path = osp.join(EXPERIMENTS_DIR, exp_name, flag_filename)
            if osp.exists(best_path):
                os.remove(best_path)
                printo(f"Removed .best file from {exp_name}")


if __name__ == "__main__":
    args = parse_args()

    if args.cmd == "train":
        if args.lookup:
            # Lookup prospects
            exps = lookup_experiments_by_group_id(
                args.group_id, args.group_level, flag_filename=PROSPECT_FLAG_FILENAME
            )
            # Train first experiment that is not already trained
            for prosepect_exp_name in exps:
                meta_path = get_meta_path(prosepect_exp_name)
                exp_kwargs = torch.load(meta_path, map_location="cpu")[
                    "hyper_parameters"
                ]
                exp_kwargs["epochs"] = args.epochs  # Override epochs
                exp_kwargs["delete_ckpt"] = False  # Don't delete new ckpt
                new_exp_name = get_experiment_name(filter_kwargs(**exp_kwargs))
                if not osp.exists(
                    osp.join(EXPERIMENTS_DIR, new_exp_name, BEST_FLAG_FILENAME)
                ):
                    train(Namespace(**exp_kwargs), flag_filename=BEST_FLAG_FILENAME)
                    break
        else:
            train(args)
    elif args.cmd == "test":
        if args.lookup:
            # Lookup best experiments
            exps = lookup_experiments_by_group_id(
                args.group_id, args.group_level, flag_filename=BEST_FLAG_FILENAME
            )
            # Test first experiment that is not already tested
            for prosepect_exp_name in exps:
                if not osp.exists(
                    osp.join(EXPERIMENTS_DIR, prosepect_exp_name, TEST_FILENAME)
                ):
                    test(
                        prosepect_exp_name,
                        remove_ckpt=args.delete_ckpt,
                        test_filename=TEST_FILENAME,
                    )
                    break
        else:
            test(args.experiment_name, remove_ckpt=args.delete_ckpt)

    elif args.cmd == "tag":
        tag(args)
