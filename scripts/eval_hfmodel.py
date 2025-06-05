# Evaluate Hugging Face model on various datasets.
import argparse
import logging
import sys

import lightning as L
import torch
from transformers import AutoModelForCausalLM

from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

from dataloaders import DATASETS
from utils.helpers import get_auto_tokenizer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",  # Much simpler format
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,  # Force to stdout instead of stderr
)
logger = logging.getLogger(__name__)


########################################################
#                 Lightning modules                    #
########################################################


class LModel(L.LightningModule):
    def __init__(
        self,
        model,
        tokenizer,
        dataset,
        max_new_tokens=256,
        do_sample=True,
        top_k=50,
        **kwargs,
    ):
        """Initialize the Lightning model for evaluation."""
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = DATASETS[dataset](tokenizer=tokenizer)

        # save hyperparameters
        self.save_hyperparameters()

    def _generate_and_test(self, batch):
        output = self.model.generate(
            **batch,
            max_new_tokens=self.hparams["max_new_tokens"],
            do_sample=self.hparams["do_sample"],
            top_k=self.hparams["top_k"],
        )

        # Compute accuracy
        y_pred_str = self.tokenizer.batch_decode(output)
        y_pred = torch.tensor(
            [self.dataset.parse_answer(y) for y in y_pred_str],  # type: ignore
            device=output.device,
        )
        corr = (y_pred == batch["labels"]).float().sum()
        return {"loss": 0, "accuracy": corr / len(batch["labels"])}

    def test_step(self, batch, batch_idx):
        results = self._generate_and_test(batch)
        self.log_dict(
            {f"test/{k}": v for k, v in results.items()},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return results


class LDataModule(L.LightningDataModule):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.tokenizer = get_auto_tokenizer(model)
        self.batch_size = kwargs.get("batch_size", 1)
        self.seq_len = kwargs.get("seq_len", 8)
        self.max_num_samples = kwargs.get("max_num_samples", None)
        self.ds_name = kwargs.get("dataset", "stemp")
        logger.info(
            f"Initialized DataModule - dataset: {self.ds_name}, batch_size: {self.batch_size}"
        )

    def setup(self, stage: str):
        logger.info(f"Setting up data for stage: {stage}")
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


########################################################
#                 Main script                          #
########################################################


def main(model_name, ds_name, **kwargs):
    """Evaluate a Hugging Face model on a specified dataset."""

    # Init model & tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    tokenizer = get_auto_tokenizer(model_name)

    # Init lmodules
    lmodel = LModel(
        model=model,
        tokenizer=tokenizer,
        dataset=ds_name,
        max_new_tokens=kwargs.get("max_new_tokens", 256),
        do_sample=kwargs.get("do_sample", True),
        top_k=kwargs.get("top_k", 50),
    )
    ldata = LDataModule(model=model_name, dataset=ds_name)

    # Test
    trainer = L.Trainer(
        logger=False,
        devices="auto",
        max_epochs=1,
        enable_progress_bar=True,
    )
    trainer.test(model=lmodel, datamodule=ldata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Hugging Face model on various datasets."
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Name of the Hugging Face model to evaluate.",
        default="gpt2",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="stemp",
    )
    args = parser.parse_args()

    main(args.model, args.dataset)
