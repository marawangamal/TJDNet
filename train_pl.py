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
from torch.distributed.fsdp import MixedPrecision

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
from utils.helpers import get_auto_tokenizer, get_git_info, get_model_and_tokenizer
from utils.arguments_v2 import parse_args
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
            self.model, _ = get_model_and_tokenizer(args)

    def training_step(self, batch, batch_idx):
        # check if any ids are negative
        output = self.model(**batch)
        self.log(
            "train_loss", output["loss"], prog_bar=True, on_epoch=True, sync_dist=True
        )
        return output["loss"]

    def validation_step(self, batch, batch_idx):
        output = self.model(**batch)
        self.log(
            "eval_loss", output["loss"], prog_bar=True, on_epoch=True, sync_dist=True
        )
        return output["loss"]

    def test_step(self, batch, batch_idx):
        outputs, ardict = self.model.generate(
            generation_config=TJDGenerationConfig(
                max_new_tokens=self.args.max_new_tokens,
                do_sample=self.args.do_sample,
                top_k=self.args.top_k,
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
        self.log("test_acc", self.test_ameter.avg, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.args.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return [optimizer], [scheduler]

    # === Debug (memory) ===
    def on_train_batch_start(self, batch, batch_idx):
        # print memory usage
        self._log_memory("before_forwardt")
        return super().on_train_batch_start(batch, batch_idx)

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
        self._log_memory("before_forwarde")
        return super().on_validation_batch_start(batch, batch_idx, dataloader_idx)

    # Max mem happens after backward and before optimizer step
    def on_after_backward(self):
        self._log_memory("after_backward")
        return super().on_after_backward()

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
    lmodel = LModel(**vars(args))

    # Data
    # lm_dataset = DATASET_LOADERS[args.dataset](
    #     tokenizer=lmodel.tokenizer,
    #     input_seq_len=args.seq_len,
    #     max_num_samples=args.max_num_samples,
    # )

    # New data:
    lm_dataset = DATASETS[args.dataset](
        tokenizer=lmodel.tokenizer,
        seq_len=args.seq_len,
        max_num_samples=args.max_num_samples,
    ).load_data()

    # No pad token needed since all samples are of same length
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
        # num_workers=4,
        # persistent_workers=True,
    )
    eval_dataloader = DataLoader(
        lm_dataset["eval"],  # type: ignore
        batch_size=args.batch_size,
        collate_fn=collator,
        # num_workers=4,
        # persistent_workers=True,
    )

    test_dataloader = DataLoader(
        lm_dataset["test"],  # type: ignore
        batch_size=1,
        # num_workers=4,
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

    # Test only
    if args.test:
        printo(f"Testing model...")
        best_ckpt = osp.join(EXPERIMENTS_DIR, exp_name, "best.ckpt")
        test_trainer = L.Trainer(strategy="auto", callbacks=[generate_cb])
        test_model = LModel.load_from_checkpoint(best_ckpt)
        test_trainer.test(test_model, dataloaders=test_dataloader)
        return

    # Train
    policy = {LlamaDecoderLayer, GPT2Block, TJDist}
    strategy = {
        "auto": "auto",
        "ddp": "ddp",
        "fsdp": FSDPStrategy(
            auto_wrap_policy=policy,
            sharding_strategy="FULL_SHARD",
            mixed_precision=MixedPrecision(param_dtype=torch.bfloat16),
            # cpu_offload=True,
            activation_checkpointing_policy={TJDist},
        ),
    }[args.accel_strategy]

    trainer = L.Trainer(
        strategy=strategy,
        max_epochs=args.epochs,
        default_root_dir=osp.join(EXPERIMENTS_DIR, exp_name),
        callbacks=(
            [checkpoint_cb, generate_cb]
            if args.accel_strategy != "fsdp"
            else checkpoint_cb
        ),
        logger=get_wandb_logger(exp_name),
        precision="bf16",
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


if __name__ == "__main__":
    args = parse_args()
    main(args)


# accelerate launch:
# [MEMORY - batch 10 (before forward)]
#   Allocated: 13.45 GB
#   Reserved:  13.49 GB
#   Peak:      30.25 GB
#   0%|         | 18/113792 [00:29<49:57:19,  1.58s/it]^CW0517 11:28:38.256250 3339902

# lightning:
#   0%|         | 24/28448 [01:10<23:11:35,  0.34it/s, v_num=ph0c, before_forward_allocated=13.50, before_forward_peak=22.40, train_loss_step=84.40]
