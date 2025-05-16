"""Train a TJD model using PyTorch Lightning.

Example:
    python train_pl.py --model distilbert/distilgpt2 --batch_size 1 --seq_len 8 --max_num_samples 10
    python train_pl.py --dataset gsm8k --model meta-llama/Llama-3.2-3B-Instruct --epochs 5 --batch_size 1 --seq_len 8 --lr 5e-5 --model_head base --horizon 1 --rank 1

"""

import os
import os.path as osp
from argparse import Namespace

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from torch import optim
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from transformers import get_linear_schedule_with_warmup


from dataloaders import CHAT_TEMPLATES, DATASET_LOADERS
from tjdnet.models.tjd import TJDGenerationConfig
from utils.helpers import get_auto_tokenizer, get_git_info, get_model_and_tokenizer
from utils.arguments_v2 import parse_args
from utils.lightning_callbacks.generate import GenerateCallback
from utils.lightning_callbacks.memory_logger import CUDAMemoryLogger
from utils.utils import AverageMeter, get_experiment_name


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
]


# define the LightningModule
class LModel(L.LightningModule):
    def __init__(self, args: Namespace):
        super().__init__()
        self.args = args
        self.tokenizer = get_auto_tokenizer(args.model)
        self.ct = CHAT_TEMPLATES[args.dataset]
        self.eos = self.tokenizer.eos_token
        self.test_ameter = AverageMeter()
        self.save_hyperparameters(args)

    def configure_model(self):
        # create all your layers here
        self.model, _ = get_model_and_tokenizer(args)

    def training_step(self, batch, batch_idx):
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

    def test_step(self, batch, batch_idx):
        # Sample
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
            [self.ct.parse_answer(y, self.tokenizer.eos_token) for y in y_pred_str],
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
        if batch_idx in [0, 10, 20, 30, 40]:
            self._log_memory("before_forward")
        return super().on_train_batch_start(batch, batch_idx)

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
    lmodel = LModel(args=args)

    # Data
    lm_dataset = DATASET_LOADERS[args.dataset](
        tokenizer=lmodel.tokenizer,
        input_seq_len=args.seq_len,
        max_num_samples=args.max_num_samples,
    )

    # No pad token needed since all samples are of same length
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token

    collator = DataCollatorForLanguageModeling(
        tokenizer=lmodel.tokenizer,
        mlm=False,  # we’re doing causal-LM, not masked-LM
        return_tensors="pt",
    )
    train_dataloader = DataLoader(
        lm_dataset["train"],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=4,
        persistent_workers=True,
    )
    eval_dataloader = DataLoader(
        lm_dataset["eval"],
        batch_size=args.batch_size,
        collate_fn=collator,
        num_workers=4,
        persistent_workers=True,
    )

    test_dataloader = DataLoader(
        lm_dataset["test"],
        batch_size=1,
        num_workers=4,
        collate_fn=identity_collator,
    )

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=osp.join(EXPERIMENTS_DIR, exp_name),
        filename="ckpt-{epoch}-{val_loss:.2f}",  # template for *best* files
        monitor="eval_loss",  # metric to track
        mode="min",  # "max" for accuracy / "min" for loss
        save_top_k=1,  # keep only the single best model
        save_last=True,  # ALSO keep a rolling 'last.ckpt'
    )
    # memory_cb = CUDAMemoryLogger()
    generate_cb = GenerateCallback()

    # Trainer
    # trainer = L.Trainer(accelerator="cuda", devices=2, strategy=FSDPStrategy())
    policy = {LlamaDecoderLayer}
    strategy = {"auto": "auto", "fsdp": FSDPStrategy(auto_wrap_policy=policy)}[  # type: ignore
        args.accel_strategy
    ]
    trainer = L.Trainer(
        # fast_dev_run=True,
        # overfit_batches=1,
        # accelerator=args.accel,
        strategy=strategy,
        max_epochs=args.epochs,
        default_root_dir=osp.join(EXPERIMENTS_DIR, exp_name),
        callbacks=[checkpoint_cb, generate_cb],
        logger=get_wandb_logger(exp_name),
        gradient_clip_val=args.grad_clip_val,
    )

    # Memory breakdown
    params = sum(p.numel() for p in lmodel.parameters())
    params_memory_gb = params * 4 / (1024**3)
    printo("\n===== MEMORY BREAKDOWN =====")
    printo(f"Params: {params / 1e9:.3f} B parameters │  {params_memory_gb:.2f} GB ")
    printo("==============================\n")

    trainer.fit(
        lmodel,
        train_dataloader,
        eval_dataloader,
        ckpt_path=ckpt_path,
    )
    trainer.test(model=lmodel, dataloaders=test_dataloader)


if __name__ == "__main__":
    args = parse_args()
    main(args)
