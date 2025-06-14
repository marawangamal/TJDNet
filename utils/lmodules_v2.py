from typing import Literal, Union
import lightning as L
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

from dataloaders import DATASETS

from peft import LoraConfig, TaskType


class LModel(L.LightningModule):
    def __init__(
        self,
        model: str = "gpt2",
        lr: float = 1e-3,
        train_mode: Literal["full", "lora"] = "lora",
        lora_rank: int = 8,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize model
        self.strict_loading = False
        self.model = AutoModelForCausalLM.from_pretrained(model)
        if self.hparams["train_mode"] == "lora":
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=self.hparams["lora_rank"],
            )
            self.model.add_adapter(peft_config, adapter_name="lora_1")

        # Randomize lm_head weights
        if hasattr(self.model, "lm_head"):
            self.model.lm_head.weight.data.normal_(mean=0.0, std=0.02)
            print("Resetting lm_head weights to random values.")

    def training_step(self, batch, batch_idx):
        out = self.model(**batch)
        loss = out.loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.model(**batch)
        loss = out.loss
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams["lr"])

    def on_save_checkpoint(self, checkpoint):
        state = checkpoint["state_dict"]
        for name in list(state.keys()):
            if not any([l in name for l in ["lora", "lm_head"]]):
                state.pop(name)
        return state


class LDataModule(L.LightningDataModule):
    def __init__(
        self,
        model: str = "gpt2",
        batch_size: int = 1,
        seq_len: int = 128,
        dataset: str = "stemp",
        max_num_samples: Union[int, None] = None,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.dataset_name = dataset
        self.max_num_samples = max_num_samples

    def setup(self, stage=None):
        dataset = DATASETS[self.dataset_name](
            tokenizer=self.tokenizer,
            seq_len=self.seq_len,
            max_num_samples=self.max_num_samples,
        ).load_data()

        self.train_ds = dataset["train"]
        self.val_ds = dataset.get("eval")
        self.test_ds = dataset.get("test")

    def train_dataloader(self):
        collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False, return_tensors="pt"
        )
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True, collate_fn=collator  # type: ignore
        )

    def val_dataloader(self):
        collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False, return_tensors="pt"
        )
        return DataLoader(self.val_ds, batch_size=self.batch_size, collate_fn=collator)  # type: ignore

    def test_dataloader(self):
        collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False, return_tensors="pt"
        )
        return DataLoader(self.test_ds, batch_size=self.batch_size, collate_fn=collator)  # type: ignore
