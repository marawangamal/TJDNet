from typing import Literal, Optional, Union
import lightning as L
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.data.data_collator import (
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
)
from lightning.pytorch.loggers import WandbLogger

from tjdnet.models.tjdsimple import TJDSimple, TJDSimpleConfig
from tjdnet.types import PositivityFuncType, ModelHeadType
from dataloaders import DATASETS


class LModel(L.LightningModule):
    def __init__(
        self,
        # model
        model: str = "gpt2",
        train_mode: Literal["full", "lora"] = "lora",
        lora_rank: int = 32,
        # tjdist parameters
        model_head: ModelHeadType = "cp",
        horizon: int = 1,
        rank: int = 1,
        positivity_func: PositivityFuncType = "safe_exp",
        # trainer
        lr: float = 1e-3,
        warmup_steps: int = 100,
        grad_clip_val: Optional[float] = None,
        # sampling parameters
        max_new_tokens: int = 128,
        do_sample: bool = False,
        top_k: int = 200,
        seq_len: int = 128,
        dataset: str = "stemp",
        debug: bool = False,
        gen_mode: Literal["draft", "base", "speculative"] = "draft",
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.dataset = DATASETS[dataset](tokenizer=self.tokenizer)

        # Initialize TJDSimple model
        self.model = TJDSimple(
            config=TJDSimpleConfig(
                model_name=model,
                model_head=model_head,
                horizon=horizon,
                rank=rank,
                train_mode=train_mode,
                lora_rank=lora_rank,
                positivity_func=positivity_func,
            )
        )

        # self.automatic_optimization = False

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams["lr"])
        return optimizer

    def training_step(self, batch, batch_idx):
        out = self.model(**batch)
        loss = out.loss
        self.log("train_loss", loss, prog_bar=True)
        return loss
        # opt = self.optimizers()
        # opt.zero_grad()
        # out = self.model.forward_backward(**batch)
        # loss = out.loss
        # if self.hparams["grad_clip_val"] is not None:
        #     self.clip_gradients(
        #         opt,
        #         gradient_clip_val=self.hparams["grad_clip_val"],
        #         gradient_clip_algorithm="norm",
        #     )
        # opt.step()
        # self.log("train_loss", loss, prog_bar=True)
        # return loss

    def validation_step(self, batch, batch_idx):
        out = self.model(**batch)
        self.log("val_loss", out.loss, prog_bar=True)
        self.log("val_nll", out.nll, prog_bar=True)
        return out.loss

    def test_step(self, batch, batch_idx):
        results = self._run_test(batch)
        self.log("test_acc", results["corr"], prog_bar=True)
        return results["corr"]

    def _run_test(self, batch):
        outputs = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_new_tokens=self.hparams["max_new_tokens"],
            do_sample=self.hparams["do_sample"],
            top_k=self.hparams["top_k"],
            eos_token_id=int(self.tokenizer.eos_token_id),  # type: ignore
        )

        # Compute accuracy
        y_pred_str = self.tokenizer.batch_decode(outputs)
        y_pred_str = [y.split(self.tokenizer.eos_token)[0] for y in y_pred_str]  # type: ignore
        y_pred = torch.tensor(
            [self.dataset.parse_answer(y) for y in y_pred_str],  # type: ignore
            device=outputs.device,
        )
        corr = (y_pred == batch["labels"]).float().sum()

        if self.hparams["debug"]:
            print("inputs:", self.tokenizer.batch_decode(batch["input_ids"]))
            print("outputs:", y_pred_str)
            print("labels:", batch["labels"])
            print("corr:", corr.item())

        return {"corr": corr.item()}

    def on_fit_start(self):
        if isinstance(self.logger, WandbLogger):
            print(">> Watching model with wandb")
            self.logger.experiment.watch(self, log="all", log_freq=100)
        else:
            print(">> No logger found")

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
        max_test_samples: Union[int, None] = None,
        max_tokens: Union[int, None] = None,
        num_workers: int = 4,
        template_mode: Literal["0_shot", "few_shot", "few_shot:standard"] = "0_shot",
        domain_shift: Literal["in", "mild", "hard"] = "in",
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.dataset_name = dataset
        self.max_num_samples = max_num_samples
        self.max_tokens = max_tokens
        self.num_workers = num_workers
        self.template_mode = template_mode
        self.max_test_samples = max_test_samples
        self.domain_shift = domain_shift
        print("Using template mode:", self.template_mode)

    def setup(self, stage=None):
        dataset = DATASETS[self.dataset_name](
            tokenizer=self.tokenizer,
            seq_len=self.seq_len,
            max_num_samples=self.max_num_samples,
            max_tokens=self.max_tokens,
            template_mode=self.template_mode,  # type: ignore
            max_test_samples=self.max_test_samples,
            domain_shift=self.domain_shift,  # type: ignore
        ).load_data()

        self.train_ds = dataset["train"]
        self.val_ds = dataset.get("eval")
        self.test_ds = dataset.get("test")

    def train_dataloader(self):
        collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False, return_tensors="pt"
        )
        return DataLoader(
            self.train_ds,  # type: ignore
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False, return_tensors="pt"
        )
        return DataLoader(
            self.val_ds,  # type: ignore
            batch_size=self.batch_size,
            collate_fn=collator,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer, return_tensors="pt"
        )
        return DataLoader(self.test_ds, batch_size=1, collate_fn=collator)  # type: ignore
