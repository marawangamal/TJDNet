"""Train a TJD model using PyTorch Lightning.

Example:
    python train_pl.py --model distilbert/distilgpt2 --batch_size 1 --seq_len 8 --max_num_samples 10
    python train_pl.py --dataset gsm8k --model meta-llama/Llama-3.2-3B-Instruct --epochs 5 --batch_size 1 --seq_len 8 --lr 5e-5 --model_head base --horizon 1 --rank 1

"""

import os
import os.path as osp
from argparse import Namespace

import torch
from torch import optim
from torch.utils.data import DataLoader

from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers import DataCollatorForLanguageModeling
from transformers import get_linear_schedule_with_warmup
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint

from dataloaders import DATASETS
from tjdnet.distributions._tjdist import TJDist
from tjdnet.models.tjd import TJD, TJDGenerationConfig
from utils.average_meter import AverageMeter
from utils.helpers import (
    get_auto_tokenizer,
    get_git_info,
    get_model_and_tokenizer,
    get_model_and_tokenizer_nowrap,
)
from utils.arguments import parse_args
from utils.lightning_callbacks.generate import GenerateCallback
from utils.experiment_naming import get_experiment_name


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
]


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
        outputs, ardict = self.model.generate(
            generation_config=TJDGenerationConfig(
                max_new_tokens=self.args.max_new_tokens,
                do_sample=self.args.do_sample,
                top_k=self.args.top_k,
                eos_token_id=int(self.tokenizer.eos_token_id),  # type: ignore
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
        self.log(
            "test_acc", self.test_ameter.avg, prog_bar=True, on_step=True, on_epoch=True
        )

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.args.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return [optimizer], [scheduler]

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


# Define a simple identity collator for single samples
def identity_collator(batch):
    # return batch[0]
    # stack all tensors across keys
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = torch.stack([torch.tensor(b[key]) for b in batch])
    return collated_batch


def train(args):
    # Setup
    L.seed_everything(42)
    filtered_args = Namespace(
        **{k: v for k, v in vars(args).items() if k not in SILENT_ARGS}
    )
    exp_name = get_experiment_name(vars(filtered_args))
    ckpt_path = osp.join(EXPERIMENTS_DIR, exp_name, "best.ckpt")
    if not osp.exists(ckpt_path):
        ckpt_path = None
    printo(
        "Training from scratch."
        if ckpt_path is None
        else f"Found checkpoint @ {ckpt_path}"
    )

    # Model
    lmodel = LModel(**vars(args))

    # Data
    lm_dataset = DATASETS[args.dataset](
        tokenizer=lmodel.tokenizer,
        seq_len=args.seq_len,
        max_num_samples=args.max_num_samples,
    ).load_data()

    # NOTE: no pad token needed since all samples are of same length
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token

    collator = DataCollatorForLanguageModeling(
        tokenizer=lmodel.tokenizer,
        mlm=False,  # we’re doing causal-LM, not masked-LM
        return_tensors="pt",
    )
    train_dataloader = DataLoader(
        lm_dataset["train"],  # type: ignore
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        # NOTE: persistent_workers=True is not supported in FSDP
        # persistent_workers=True,
    )
    eval_dataloader = DataLoader(
        lm_dataset["eval"],  # type: ignore
        batch_size=max(args.batch_size // 2, 1),
        collate_fn=collator,
        num_workers=0,
        # persistent_workers=True,
    )

    test_dataloader = DataLoader(
        lm_dataset["test"],  # type: ignore
        batch_size=1,
        num_workers=0,
        collate_fn=identity_collator,
    )

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=osp.join(EXPERIMENTS_DIR, exp_name),
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
        default_root_dir=osp.join(EXPERIMENTS_DIR, exp_name),
        callbacks=(
            [checkpoint_cb, generate_cb]
            if args.accel_strategy != "fsdp"
            else checkpoint_cb
        ),
        logger=get_wandb_logger(exp_name),
        precision="bf16-true",
        accumulate_grad_batches=args.accum_grad_batches,
    )

    trainer.fit(
        lmodel,
        train_dataloader,
        eval_dataloader,
        ckpt_path=ckpt_path,
    )
    if not args.accel_strategy == "fsdp":
        trainer.test(
            ckpt_path="best",
            dataloaders=test_dataloader,
        )
    if torch.cuda.is_available():
        trainer.print(torch.cuda.memory_summary())


def eval(args):
    printo(f"Testing model...")
    # Load args from checkpoint
    lmodel = LModel.load_from_checkpoint(args.ckpt)
    exp_args = Namespace(**lmodel.hparams)
    generate_cb = GenerateCallback(
        prompt=DATASETS[exp_args.dataset].get_sample_prompt()
    )
    trainer = L.Trainer(strategy="auto", callbacks=[generate_cb])
    ldata = LDataModule(**vars(exp_args))
    trainer.test(lmodel, dataloaders=ldata)
    return


if __name__ == "__main__":
    args = parse_args()

    if args.cmd == "train":
        train(args)  # ← your existing train() function
    elif args.cmd == "test":
        eval(args)  # ← your existing eval() / test() function


# accelerate launch:
# [MEMORY - batch 10 (before forward)]
#   Allocated: 13.45 GB
#   Reserved:  13.49 GB
#   Peak:      30.25 GB
#   0%|         | 18/113792 [00:29<49:57:19,  1.58s/it]^CW0517 11:28:38.256250 3339902

# lightning:
#   0%|         | 24/28448 [01:10<23:11:35,  0.34it/s, v_num=ph0c, before_forward_allocated=13.50, before_forward_peak=22.40, train_loss_step=84.40]
