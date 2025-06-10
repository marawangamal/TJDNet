"""Train and evaluate TJD models using PyTorch Lightning.

Supports distributed training, experiment tracking with WandB, and automated
checkpoint management. Organizes experiments by groups for easy comparison.

Commands:
    train   Train a TJD model
    test    Evaluate trained models on test data
    tag     Mark best performing models within groups

Examples:
    # Basic training
    python main.py train --model distilbert/distilgpt2 --batch_size 1 --seq_len 8

    # Distributed training
    python main.py train --accel_strategy fsdp --dataset gsm8k --model meta-llama/Llama-3.2-3B-Instruct

    # Test
    python main.py test --experiment_name my_experiment

    # Tag -- tags best model within a `group_id`
    python main.py tag --group_id xxx-xxx-xxx-xxx

    # Test (w/ lookup) -- test tagged experiments
    python main.py test --lookup --group_id xxx-xxx-xxx-xxx
"""

from ast import Name
from collections import defaultdict
import os
import os.path as osp
import logging
from datetime import datetime

from argparse import Namespace
import subprocess
import sys
import time
from typing import List

import torch
from torch import optim
from wandb.util import generate_id
import torchmetrics as tm


import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_lightning.utilities import rank_zero_only
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block


from dataloaders import DATASETS
from tjdnet.distributions._tjdist import TJDist
from utils.helpers import get_auto_tokenizer, get_git_info, get_model_and_tokenizer
from utils.lightning_callbacks.generate import GenerateCallback
from utils.experiment_naming import get_experiment_name
from utils.arguments import parse_args
from utils.lmodules import LModel, LDataModule, SafeModelCheckpoint

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",  # Much simpler format
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,  # Force to stdout instead of stderr
)
logger = logging.getLogger(__name__)

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
    "idx",
]

PROSPECT_FLAG_FILENAME = ".prospect"
BEST_FLAG_FILENAME = ".best"
TEST_FILENAME = "test_results.txt"


#################################################################
#                       Helpers                                 #
#################################################################


@rank_zero_only
def get_wandb_logger(exp_name: str, wandb_id=None):
    logger.info(f"Setting up WandB logger for experiment: {exp_name}")
    git_info = get_git_info()
    suffix = "main" if git_info.get("branch") == "main" else "dev"
    project_name = f"tjdnet-{suffix}"
    wandb_logger = WandbLogger(
        project=project_name,
        name=exp_name,
        id=wandb_id,
        resume="allow",
    )
    logger.info(f"WandB logger configured - Project: {project_name}, ID: {wandb_id}")
    return wandb_logger


def printo(*args, **kwargs):
    """Print to stdout and stderr - now uses proper logging."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        logger.info(" ".join(str(arg) for arg in args))


@rank_zero_only
def generate_wandb_id():
    wandb_id = generate_id()
    logger.info(f"Generated new WandB ID: {wandb_id}")
    return wandb_id


def filter_kwargs(**kwargs):
    # Filter out args that are not needed for the experiment
    return {k: v for k, v in kwargs.items() if k not in SILENT_ARGS}


def print_args(args):
    """Print arguments in a structured format."""
    logger.info("=" * 50)
    logger.info("EXPERIMENT CONFIGURATION")
    logger.info("=" * 50)

    # Group args by category for better readability
    model_args = {}
    training_args = {}
    data_args = {}
    other_args = {}
    silent_args = {}

    for k, v in vars(args).items():
        if k in ["model", "horizon", "precision"]:
            model_args[k] = v
        elif k in [
            "lr",
            "epochs",
            "warmup_steps",
            "batch_size",
            "accum_grad_batches",
        ]:
            training_args[k] = v
        elif k in ["dataset", "seq_len", "max_num_samples"]:
            data_args[k] = v
        elif k in SILENT_ARGS:
            silent_args[k] = v
        else:
            other_args[k] = v

    if model_args:
        logger.info("Model Configuration:")
        for k, v in model_args.items():
            logger.info(f"  {k}: {v}")

    if training_args:
        logger.info("Training Configuration:")
        for k, v in training_args.items():
            logger.info(f"  {k}: {v}")

    if data_args:
        logger.info("Data Configuration:")
        for k, v in data_args.items():
            logger.info(f"  {k}: {v}")

    if silent_args:
        logger.info("Silent Configuration:")
        for k, v in silent_args.items():
            logger.info(f"  {k}: {v}")

    if other_args:
        logger.info("Other Configuration:")
        for k, v in other_args.items():
            logger.info(f"  {k}: {v}")

    logger.info("=" * 50)


def maybe_update_args(args, exp_name: str):
    """Update args in meta file with retry logic."""
    meta_path = get_meta_path(exp_name)
    attempts = 3
    max_retries = 10
    retry_delay = 2  # seconds

    if osp.exists(meta_path):
        for attempt in range(attempts):
            try:
                meta_ckpt = torch.load(
                    meta_path, map_location="cpu", weights_only=False
                )
                meta_ckpt["hyper_parameters"].update(vars(args))
                torch.save(meta_ckpt, meta_path)
                logger.info(f"Successfully updated args in meta file: {meta_path}")
                return
            except (RuntimeError, OSError) as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Failed to update args (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error(
                        f"Failed to update args after {max_retries} attempts: {e}"
                    )


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


def make_consolidated_ckpt(exp_name: str):
    ckpt_path = osp.join(EXPERIMENTS_DIR, exp_name, "best.ckpt")
    if osp.isdir(ckpt_path):
        if osp.exists(ckpt_path + ".consolidated"):
            logger.info(
                f"Using existing consolidated checkpoint: {ckpt_path}.consolidated"
            )
            return ckpt_path + ".consolidated"
        else:
            logger.info(f"Consolidating checkpoint: {ckpt_path}")
            result = subprocess.run(
                [
                    "python",
                    "-m",
                    "lightning.pytorch.utilities.consolidate_checkpoint",
                    str(ckpt_path),
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.error(f"Failed to consolidate checkpoint: {result.stderr}")
            else:
                logger.info("Checkpoint consolidation completed successfully")
            return ckpt_path + ".consolidated"
    return ckpt_path


def get_exp_eval_score(exp: str):
    """Load experiment eval_loss and hyperparams."""
    try:
        meta_path = get_meta_path(exp)
        meta_ckpt = torch.load(meta_path, map_location="cpu")
        for key, cb in meta_ckpt.get("callbacks", {}).items():
            if "ModelCheckpoint" in key and "eval_loss" in key:
                return cb["best_model_score"]
        return None
    except Exception as e:
        logger.error(f"Failed to get eval score for {exp}: {e}")
        return None


def remove_exp_ckpts(exp: str):
    """Remove all checkpoints for the experiment."""
    try:
        ckpt_paths = get_ckpt_file_paths(exp)
        for ckpt_path in ckpt_paths:
            if ckpt_path.endswith("best.ckpt"):
                # remove only the params
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                if "state_dict" in ckpt:
                    del ckpt["state_dict"]
                    torch.save(ckpt, ckpt_path)
                    logger.info(f"Removed state_dict from checkpoint: {ckpt_path}")
            elif osp.exists(ckpt_path):
                os.remove(ckpt_path)
                logger.info(f"Deleted checkpoint: {ckpt_path}")
        logger.info(f"Successfully removed all checkpoints for experiment: {exp}")
    except Exception as e:
        logger.error(f"Error deleting checkpoints for {exp}: {e}")


def lookup_experiments_by_group_id(
    group_id: str, group_level: int = 0, flag_filename=None
) -> List[str]:
    """Find all best checkpoints matching group_id at the specified group level."""
    logger.info(
        f"Looking up experiments for group_id: {group_id} at level {group_level}"
    )
    filtered_exps = []

    for exp in os.listdir(EXPERIMENTS_DIR):
        # Filter: skip if not .best
        if flag_filename is not None:
            flag_file = osp.join(EXPERIMENTS_DIR, exp, flag_filename)
            if not osp.exists(flag_file):
                continue

        # Filter: apply group_id, group_level
        try:
            hparams = get_hyper_parameters(exp)
            if "group_id" in hparams and hparams["group_id"] is not None:
                exp_group = hparams["group_id"].split("-")[group_level]
                target_group = group_id.split("-")[group_level]

                if exp_group == target_group:
                    filtered_exps.append(exp)
        except Exception as e:
            logger.warning(f"Could not process experiment {exp}: {e}")
            continue

    logger.info(f"Found {len(filtered_exps)} experiments matching criteria")
    for exp in filtered_exps:
        logger.debug(f"  - {exp}")

    return filtered_exps


#################################################################
#                       train/test/tag                          #
#################################################################


def train(args, flag_filename=None):
    """Train a model with improved logging."""
    logger.info("Starting training process")
    print_args(args)

    L.seed_everything(42)
    exp_name_filtered = get_experiment_name(filter_kwargs(**vars(args)))
    ckpt_path = osp.join(EXPERIMENTS_DIR, exp_name_filtered, "best.ckpt")
    wandb_id = None

    # Check if experiment already completed
    meta_path = get_meta_path(exp_name_filtered)
    if osp.exists(meta_path):
        completed_flag_path = osp.join(EXPERIMENTS_DIR, exp_name_filtered, ".completed")
        if osp.exists(completed_flag_path):
            logger.info(f"Experiment {exp_name_filtered} already completed - Skipping")
            maybe_update_args(args, exp_name_filtered)
            return

    # Handle existing checkpoints
    if len(get_ckpt_file_paths(exp_name_filtered)) > 0:
        try:
            existing_ckpt = torch.load(
                get_ckpt_file_paths(exp_name_filtered)[0], map_location="cpu"
            )
            wandb_id = existing_ckpt["hyper_parameters"]["wandb_id"]
            logger.info(f"Resuming from existing checkpoint - WandB ID: {wandb_id}")
        except Exception as e:
            logger.warning(f"Could not load existing checkpoint: {e}")
            ckpt_path = None
    else:
        ckpt_path = None
        wandb_id = generate_wandb_id()
        args.wandb_id = wandb_id
        args.experiment_name = exp_name_filtered
        logger.info("Starting training from scratch")

    logger.info(f"Experiment name: {exp_name_filtered}")
    logger.info(f"Group ID: {args.group_id}")

    # Setup logging and model
    wandb_logger = get_wandb_logger(exp_name_filtered, wandb_id=wandb_id)
    lmodel = LModel(**vars(args))
    ldata = LDataModule(**vars(args))

    # Callbacks
    checkpoint_cb = SafeModelCheckpoint(
        dirpath=osp.join(EXPERIMENTS_DIR, exp_name_filtered),
        filename="best",
        monitor="eval_loss",
        mode="min",
        save_top_k=1,
    )
    tokenizer = get_auto_tokenizer(args.model)
    generate_cb = GenerateCallback(
        prompt=DATASETS[args.dataset](tokenizer=tokenizer).get_sample_prompt()
    )
    early_stopping_cb = EarlyStopping(
        monitor="eval_loss",
        patience=3,
        mode="min",
        check_finite=True,
    )
    callbacks = [checkpoint_cb, early_stopping_cb]
    if not args.accel_strategy == "fsdp":
        callbacks.append(generate_cb)

    # Memory analysis
    params = sum(p.numel() for p in lmodel.parameters())
    params_memory_gb = params * 4 / (1024**3)
    logger.info("=" * 50)
    logger.info("MEMORY ANALYSIS")
    logger.info("=" * 50)
    logger.info(f"Model parameters: {params / 1e9:.3f}B ({params_memory_gb:.2f} GB)")
    logger.info("=" * 50)

    # Training strategy
    policy = {LlamaDecoderLayer, GPT2Block, TJDist}
    strategy = {
        "auto": "auto",
        "ddp": "ddp",
        "fsdp": FSDPStrategy(
            auto_wrap_policy=policy,
            sharding_strategy="FULL_SHARD",
            state_dict_type="sharded",
        ),
    }[args.accel_strategy]

    logger.info(f"Using acceleration strategy: {args.accel_strategy}")

    # Create trainer
    trainer = L.Trainer(
        fast_dev_run=args.fast_dev_run,
        strategy=strategy,
        max_epochs=args.epochs,
        default_root_dir=osp.join(EXPERIMENTS_DIR, exp_name_filtered),
        callbacks=callbacks,
        logger=wandb_logger,
        accumulate_grad_batches=args.accum_grad_batches,
        gradient_clip_val=1,
        precision=args.precision,
    )

    # Start training
    logger.info("Starting training...")
    start_time = datetime.now()

    trainer.fit(lmodel, datamodule=ldata, ckpt_path=ckpt_path)

    training_time = datetime.now() - start_time
    logger.info(f"Training completed in {training_time}")

    # Testing
    if not args.accel_strategy == "fsdp" and args.compute_acc:
        logger.info("Starting evaluation on test set...")
        trainer.test(ckpt_path="best", datamodule=ldata)

    # Memory summary
    if torch.cuda.is_available():
        memory_summary = torch.cuda.memory_summary()
        logger.info(f"Final memory summary:\n{memory_summary}")

    # Save flag file
    if flag_filename:
        flag_path = osp.join(EXPERIMENTS_DIR, exp_name_filtered, flag_filename)
        with open(flag_path, "w") as f:
            f.write(f"Training completed at {datetime.now()}")
        logger.info(f"Flag file saved: {flag_path}")

    # Cleanup
    if args.delete_ckpt:
        remove_exp_ckpts(exp_name_filtered)
        logger.info(f"Deleted checkpoints for experiment: {exp_name_filtered}")

    # Final update
    time.sleep(5)  # Allow file operations to complete
    maybe_update_args(args, exp_name_filtered)

    # Add completed flag
    completed_flag_path = osp.join(EXPERIMENTS_DIR, exp_name_filtered, ".completed")
    with open(completed_flag_path, "w") as f:
        f.write(f"Training completed successfully at {datetime.now()}\n")
    logger.info(f"Training completed successfully - Flag saved: {completed_flag_path}")


def test(exp_name: str, remove_ckpt=True, test_filename=TEST_FILENAME, **kwargs):
    """Test a trained model with improved logging."""
    logger.info(f"Starting test for experiment: {exp_name}")

    # Check if already tested
    test_results_path = osp.join(EXPERIMENTS_DIR, exp_name, test_filename)
    if osp.exists(test_results_path):
        logger.info(f"Test results already exist for {exp_name} - Skipping")
        return

    # Load and test model
    logger.info("Loading model checkpoint...")
    try:

        # Filter kwargs to only include overrideable args
        overrideable_args = [
            "max_new_tokens",
            "do_sample",
            "top_k",
            "gen_mode",
            "dataset",
            "template_mode",
        ]
        kwargs = {
            k: v for k, v in kwargs.items() if k in overrideable_args and v is not None
        }
        logger.info(f"Overriding test arguments: {kwargs}")

        ckpt_path = make_consolidated_ckpt(exp_name)
        lmodel = LModel.load_from_checkpoint(ckpt_path, **kwargs)
        exp_args = Namespace(**lmodel.hparams)
        logger.info(f"Model loaded successfully from: {ckpt_path}")
    except Exception as e:
        logger.error(f"Failed to load model checkpoint: {e}")
        return

    # Setup trainer
    tokenizer = get_auto_tokenizer(exp_args.model)
    generate_cb = GenerateCallback(
        prompt=DATASETS[exp_args.dataset](tokenizer=tokenizer).get_sample_prompt()
    )

    logger.info("Setting up WandB logger for testing...")
    wandb_logger = get_wandb_logger(
        exp_args.experiment_name, wandb_id=exp_args.wandb_id
    )
    trainer = L.Trainer(
        accelerator="gpu", devices=1, callbacks=[generate_cb], logger=wandb_logger
    )

    if kwargs:
        logger.info(f"Overriding test arguments: {kwargs}")

    logger.info("Setting up test data...")
    ldata = LDataModule(**lmodel.hparams)

    # Run test
    logger.info("Starting model evaluation...")
    start_time = datetime.now()
    test_results = trainer.test(lmodel, datamodule=ldata)
    test_time = datetime.now() - start_time

    logger.info(f"Testing completed in {test_time}")

    # Save test results to file
    with open(test_results_path, "w") as f:
        f.write(f"Test results for {exp_name}:\n")
        f.write(f"Test completed at: {datetime.now()}\n")
        f.write(f"Test duration: {test_time}\n\n")
        for key, value in test_results[0].items():
            f.write(f"{key}: {value}\n")

    logger.info(f"Test results saved to: {test_results_path}")

    # Cleanup
    if remove_ckpt:
        remove_exp_ckpts(exp_name)
    logger.info(f"Removed checkpoints for experiment: {exp_name}")

    logger.info("Testing process completed successfully")


def tag(args, flag_filename=PROSPECT_FLAG_FILENAME):
    """Tag the best model in each group with improved logging."""
    logger.info(f"Starting tagging process for group_id: {args.group_id}")
    logger.info(f"Group level: {args.group_level}, Flag: {flag_filename}")

    # Lookup experiments
    exps = lookup_experiments_by_group_id(args.group_id, args.group_level)

    if not exps:
        logger.warning("No experiments found matching the criteria")
        return

    # Get experiment losses
    logger.info("Evaluating experiment performance...")
    exp_losses = dict()
    failed_exps = []

    for exp_name in exps:
        try:
            exp_loss = get_exp_eval_score(exp_name)
            hparams = get_hyper_parameters(exp_name)

            if exp_loss is None:
                logger.warning(f"Skipping {exp_name} - no eval_loss available")
                failed_exps.append(exp_name)
                continue

            exp_losses[exp_name] = (exp_loss, hparams)
            logger.debug(f"{exp_name}: eval_loss = {exp_loss:.4f}")

        except Exception as e:
            logger.error(f"Failed to process experiment {exp_name}: {e}")
            failed_exps.append(exp_name)

    if failed_exps:
        logger.warning(
            f"Failed to process {len(failed_exps)} experiments: {failed_exps}"
        )

    if not exp_losses:
        logger.error("No valid experiments found with eval_loss")
        return

    # Apply grouping
    logger.info("Grouping experiments...")
    groups = defaultdict(dict)

    if args.group_by:
        logger.info(f"Grouping by parameters: {args.group_by}")
        for exp_name, (loss, hparams) in exp_losses.items():
            key_parts = [
                f"{param}={hparams.get(param, 'unknown')}" for param in args.group_by
            ]
            key = ", ".join(key_parts)
            groups[key][exp_name] = loss
    else:
        # If no grouping specified, treat all experiments as one group
        logger.info("No grouping specified - treating all experiments as one group")
        groups["all"] = {exp_name: loss for exp_name, (loss, _) in exp_losses.items()}

    # Find best within each group
    logger.info("Finding best models in each group...")
    best_exps = set()
    removed_flags = []

    for group_name, group_exps in groups.items():
        if not group_exps:
            logger.warning(f"No experiments in group: {group_name}")
            continue

        # Find experiment with minimum loss (assuming lower is better)
        best_exp, best_loss = min(group_exps.items(), key=lambda x: x[1])
        best_exps.add(best_exp)

        logger.info(
            f"Best in group '{group_name}': {best_exp} (eval_loss: {best_loss:.4f})"
        )

        # Tag best model
        best_path = osp.join(EXPERIMENTS_DIR, best_exp, flag_filename)
        with open(best_path, "w") as f:
            f.write(f"Best model in group: {group_name}\n")
            f.write(f"Tagged at: {datetime.now()}\n")
            f.write(f"Eval loss: {best_loss}\n")

        logger.info(f"Tagged {best_exp} as best in group '{group_name}'")

    # Remove flags from non-best models
    logger.info("Removing flags from non-best models...")
    for exp_name in exps:
        if exp_name not in best_exps:
            flag_path = osp.join(EXPERIMENTS_DIR, exp_name, flag_filename)
            if osp.exists(flag_path):
                os.remove(flag_path)
                removed_flags.append(exp_name)

    if removed_flags:
        logger.info(
            f"Removed flags from {len(removed_flags)} experiments: {removed_flags}"
        )

    # Summary
    logger.info("=" * 50)
    logger.info("TAGGING SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total experiments processed: {len(exps)}")
    logger.info(f"Valid experiments: {len(exp_losses)}")
    logger.info(f"Groups created: {len(groups)}")
    logger.info(f"Best models tagged: {len(best_exps)}")
    logger.info(f"Flags removed: {len(removed_flags)}")
    logger.info("=" * 50)

    logger.info("Tagging process completed successfully")


if __name__ == "__main__":
    args = parse_args()

    logger.info(f"Starting TJD training script - Command: {args.cmd}")

    if args.cmd == "train":
        if args.lookup:
            logger.info(
                "Training mode: Lookup prospects and train first untrained experiment"
            )
            # Lookup prospects
            exps = lookup_experiments_by_group_id(
                args.group_id, args.group_level, flag_filename=PROSPECT_FLAG_FILENAME
            )

            if not exps:
                logger.warning("No prospect experiments found")
            else:
                # Train first experiment that is not already trained
                for i, prospect_exp_name in enumerate(exps):
                    # If args.idx provided and valid limit to only that idx
                    if args.idx and len(exps) > args.idx and i != args.idx:
                        continue
                    meta_path = get_meta_path(prospect_exp_name)
                    exp_kwargs = torch.load(meta_path, map_location="cpu")[
                        "hyper_parameters"
                    ]
                    overrideable_args = [
                        "epochs",
                        "max_num_samples",
                    ]
                    okwargs = {
                        k: v
                        for k, v in vars(args).items()
                        if k in overrideable_args and v is not None
                    }
                    logger.info(f"Overriding prospect experiment args: {okwargs}")
                    exp_kwargs.update(okwargs)

                    exp_kwargs["delete_ckpt"] = False  # Don't delete new ckpt

                    new_exp_name = get_experiment_name(filter_kwargs(**exp_kwargs))
                    best_flag_path = osp.join(
                        EXPERIMENTS_DIR, new_exp_name, BEST_FLAG_FILENAME
                    )

                    if not osp.exists(best_flag_path):
                        logger.info(
                            f"Training prospect experiment: {prospect_exp_name} -> {new_exp_name}"
                        )
                        train(
                            Namespace(**exp_kwargs),
                            flag_filename=BEST_FLAG_FILENAME,
                        )
                        # only train one experiment
                        break
                    else:
                        logger.info(
                            f"Experiment {new_exp_name} already trained - Skipping"
                        )

        else:
            logger.info("Training mode: Direct training")
            train(args)

    elif args.cmd == "test":
        if args.lookup:
            logger.info("Test mode: Lookup best experiments and test first untested")
            # Lookup best experiments
            exps = lookup_experiments_by_group_id(
                args.group_id, args.group_level, flag_filename=BEST_FLAG_FILENAME
            )

            if not exps:
                logger.warning("No best experiments found")
            else:
                # Test first experiment that is not already tested
                for i, best_exp_name in enumerate(exps):
                    # If args.idx provided and valid limit to only that idx
                    if args.idx and len(exps) > args.idx and i != args.idx:
                        continue
                    test_file_path = osp.join(
                        EXPERIMENTS_DIR, best_exp_name, TEST_FILENAME
                    )
                    if not osp.exists(test_file_path):
                        logger.info(f"Testing experiment: {best_exp_name}")
                        test(
                            best_exp_name,
                            remove_ckpt=args.delete_ckpt,
                            test_filename=TEST_FILENAME,
                            **vars(args),
                        )
                        break
                    else:
                        logger.info(
                            f"Experiment {best_exp_name} already tested - Skipping"
                        )

        else:
            logger.info(f"Test mode: Direct testing of {args.experiment_name}")
            test(args.experiment_name, remove_ckpt=args.delete_ckpt, **vars(args))

    elif args.cmd == "tag":
        logger.info("Tag mode: Finding and tagging best models")
        tag(args)

    else:
        raise ValueError(f"Unknown command: {args.cmd}")
