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

import os
import logging

from argparse import Namespace
import sys
from typing import List, Literal, Optional, Union

import torch
from torch import optim
from torch.utils.data import DataLoader
import torchmetrics as tm


import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from transformers import (
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)


from dataloaders import DATASETS
from tjdnet.models.tjd import TJD, TJDGenerationConfig
from utils.helpers import get_auto_tokenizer, get_model_and_tokenizer

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
#                       Lightning Classes                       #
#################################################################


# define the LightningModule
class LModel(L.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model: TJD = None  # type: ignore
        self.tokenizer = get_auto_tokenizer(self.hparams["model"])
        self.dataset = DATASETS[self.hparams["dataset"]](tokenizer=self.tokenizer)

        # Configure metrics
        def make_meter_dict():
            return torch.nn.ModuleDict(
                {
                    k: tm.MeanMetric()
                    for k in ("draft", "base", "speculative", "acceptance_rate")
                }
            )

        self.metrics = torch.nn.ModuleDict(
            {
                "H": make_meter_dict(),
                "1": make_meter_dict(),
            }
        )

        logger.info(
            f"Initialized LModel with dataset: {self.hparams['dataset']}, model: {self.hparams['model']}"
        )

    # ==== Configuration
    def configure_model(self):
        # IMPORTANT: This function must be idempotent (i.e., calling it multiple times should not change self.model)
        if self.model is None:  # Model might be already created in load_from_checkpoint
            logger.info("Configuring model...")
            self.model, _ = get_model_and_tokenizer(Namespace(**self.hparams))
            logger.info(f"Model configured: {type(self.model).__name__}")

    def configure_optimizers(self):
        logger.info(
            f"Configuring optimizer with lr={self.hparams['lr']}, warmup_steps={self.hparams['warmup_steps']}"
        )
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams["lr"])
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams["warmup_steps"],
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

    # ==== Training / Evaluation

    def training_step(self, batch, batch_idx):
        output = self.model(**batch)
        loss = output["loss"]
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            output = self.model(**batch)
        loss = output["loss"]
        self.log("eval_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    # ===== Testing

    def _get_hlabels(self) -> List[Literal["H", "1"]]:
        opts: dict[str, List[Literal["H", "1"]]] = {
            "mixed": ["H", "1"],
            "draft_multi_horizon": ["H", "1"],
            "base_multi_horizon": ["H", "1"],
        }
        return opts.get(self.hparams["gen_mode"], ["H"])

    def _get_gen_modes(self) -> List[Literal["draft", "base", "speculative"]]:
        opts: dict[str, List[Literal["draft", "base", "speculative"]]] = {
            "draft_multi_horizon": ["draft"],
            "base_multi_horizon": ["base"],
            "mixed": ["draft", "base", "speculative"],
        }
        return opts.get(self.hparams["gen_mode"], [self.hparams["gen_mode"]])

    def _generate_and_test(
        self, batch, gen_mode: Literal["draft", "base", "speculative"], horizon: int
    ):
        outputs, ardict = self.model.generate(
            generation_config=TJDGenerationConfig(
                max_new_tokens=self.hparams["max_new_tokens"],
                do_sample=self.hparams["do_sample"],
                top_k=self.hparams["top_k"],
                eos_token_id=int(self.tokenizer.eos_token_id),  # type: ignore
                gen_mode=gen_mode,
                horizon=horizon,
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
        return {
            "corr": corr,
            "tokens_accepted": ardict["tokens_accepted"],
            "tokens_generated": ardict["tokens_generated"],
        }

    def test_step(self, batch, batch_idx):
        for hlabel in self._get_hlabels():
            for gmode in self._get_gen_modes():
                horizon = 1 if hlabel == "1" else self.hparams["horizon"]
                out = self._generate_and_test(batch, gmode, horizon)

                # accuracy
                self.metrics[hlabel][gmode].update(  # type: ignore
                    out["corr"] / len(batch["input_ids"]), len(batch["input_ids"])
                )

                # acceptance
                if gmode == "speculative":
                    denom = out["tokens_generated"]
                    ar = (out["tokens_accepted"] / denom) if denom else 0.0
                    self.metrics[hlabel]["acceptance_rate"].update(ar, denom)  # type: ignore

    def on_test_epoch_end(self):
        # Lightning handles metric logging to progress bar and WandB
        logger.info("Test epoch completed")
        for hlabel in self._get_hlabels():
            for gmode in self._get_gen_modes():

                # accuracy
                metric_value = self.metrics[hlabel][gmode].compute()  # type: ignore
                metric_name = f"test_h{hlabel}_{gmode}_acc"
                self.log(metric_name, metric_value, prog_bar=True)

                # acceptance rate
                if gmode == "speculative":
                    metric_name = f"test_h{hlabel}_acceptance_rate"
                    metric_value = self.metrics[hlabel]["acceptance_rate"].compute()  # type: ignore
                    self.log(metric_name, metric_value, prog_bar=True)

    # === Memory Logging ===

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
    def __init__(self, model, **kwargs):
        super().__init__()
        self.tokenizer = get_auto_tokenizer(model)
        self.batch_size = kwargs.get("batch_size", 1)
        self.seq_len = kwargs.get("seq_len", 8)
        self.max_num_samples = kwargs.get("max_num_samples", None)
        self.ds_name = kwargs.get("dataset", "stemp")
        self.ds_ttype = kwargs.get("template_mode", "0_shot")
        logger.info(
            f"Initialized DataModule - dataset: {self.ds_name}, batch_size: {self.batch_size}"
        )

    def setup(self, stage: str):
        logger.info(f"Setting up data for stage: {stage}")
        self.lm_dataset = DATASETS[self.ds_name](
            tokenizer=self.tokenizer,
            seq_len=self.seq_len,
            max_num_samples=self.max_num_samples,
            template_mode=self.ds_ttype,
        ).load_data()
        self.train_ds, self.eval_ds, self.test_ds = (
            self.lm_dataset["train"],
            self.lm_dataset["eval"],
            self.lm_dataset["test"],
        )
        logger.info(
            f"Data loaded - Train: {len(self.train_ds)}, Eval: {len(self.eval_ds)}, Test: {len(self.test_ds)}"
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
        def collator(batch):
            collated_batch = {}
            for key in batch[0].keys():
                collated_batch[key] = torch.stack([torch.tensor(b[key]) for b in batch])
            return collated_batch

        return collator

    def _collator_train(self):
        collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # we're doing causal-LM, not masked-LM
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
