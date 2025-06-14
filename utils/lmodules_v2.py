from typing import Literal, Optional, Union
import lightning as L
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)


from dataloaders import DATASETS

from peft import LoraConfig, TaskType

from tjdnet.distributions._tjdist import BaseDistConfig
from tjdnet.distributions.tpnet import TensorParamNetConfig
from tjdnet.models.tjd import TJDConfig
from tjdnet.models.tjdhf import TJDHuggingFace


class LModel(L.LightningModule):
    def __init__(
        self,
        model: str = "gpt2",
        lr: float = 1e-3,
        train_mode: Literal["full", "lora"] = "lora",
        lora_rank: int = 32,
        warmup_steps: int = 100,
        # sampling parameters
        max_new_tokens: int = 128,
        do_sample: bool = True,
        top_k: int = 200,
        seq_len: int = 128,
        dataset: str = "stemp",
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.dataset = DATASETS[dataset](tokenizer=self.tokenizer)

        # Initialize model
        self.strict_loading = False
        # ==== HF >>>>>>
        self.model = AutoModelForCausalLM.from_pretrained(model)
        # if self.hparams["train_mode"] == "lora":
        #     peft_config = LoraConfig(
        #         task_type=TaskType.FEATURE_EXTRACTION,
        #         inference_mode=False,
        #         r=self.hparams["lora_rank"],
        #         lora_alpha=32,
        #         lora_dropout=0.1,
        #     )
        #     self.model.add_adapter(peft_config, adapter_name="lora_1")
        # =========
        # self.model = TJDHuggingFace(
        #     auto_model_kwargs={"pretrained_model_name_or_path": model},
        #     train_mode=self.hparams["train_mode"],
        #     lora_rank=self.hparams["lora_rank"],
        #     config=TJDConfig(
        #         model_head="base",
        #         model_head_config=BaseDistConfig(
        #             vocab_size=-1,
        #             horizon=1,
        #             rank=1,
        #             param_net=TensorParamNetConfig(),
        #         ),
        #     ),
        # )
        # ==== TJD >>>>>>

        # # Randomize lm_head weights
        # if hasattr(self.model, "lm_head"):
        #     self.model.lm_head.weight.data.normal_(mean=0.0, std=0.02)
        #     print("Resetting lm_head weights to random values.")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams["lr"])
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

    def test_step(self, batch, batch_idx):
        results = self._run_test(
            batch, gen_mode="draft", horizon=self.hparams["seq_len"]
        )
        self.log("test_acc", results["corr"], prog_bar=True)

    def _run_test(
        self, batch, gen_mode: Literal["draft", "base", "speculative"], horizon: int
    ):
        outputs = self.model.generate(
            max_new_tokens=self.hparams["max_new_tokens"],
            do_sample=self.hparams["do_sample"],
            top_k=self.hparams["top_k"],
            eos_token_id=int(self.tokenizer.eos_token_id),  # type: ignore
            pad_token_id=int(self.tokenizer.pad_token_id),  # type: ignore
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
            "corr": corr.item(),
        }

    # def on_save_checkpoint(self, checkpoint):
    #     state = checkpoint["state_dict"]
    #     for name in list(state.keys()):
    #         if not any([l in name for l in ["lora", "lm_head"]]):
    #             state.pop(name)
    #     return state


class LDataModule(L.LightningDataModule):
    def __init__(
        self,
        model: str = "gpt2",
        batch_size: int = 32,
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
        collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer, return_tensors="pt"
        )
        return DataLoader(self.test_ds, batch_size=1, collate_fn=collator)  # type: ignore
