# https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html

import os.path as osp
import os
from typing import Callable, Dict, List, Optional, Any, Tuple, Union

import torch
from transformers import (
    DataCollatorForLanguageModeling,
    AutoModelForCausalLM,
    AutoTokenizer,
)

# pyright: reportPrivateImportUsage=false
from peft import LoraConfig, TaskType, get_peft_model

from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm

from data.gsm8k import load_gsm8k_data
from data.shakespeare import load_shakespeare_data
from data.sharegptv2 import load_sharegptv2_data
from data.syn_number_bases import load_syn_num_base_data
from data.syn_numbers import load_syn_num_data
from data.syn_temp import load_syn_temp_data
from data.wikitext import load_wikitext_data
from utils import get_experiment_name
from utils.train_helpers import (
    get_chat_template,
    get_model_and_tokenizer,
    parse_args,
    save_args,
    set_seed,
)


def inner_train_loop(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    accelerator: Accelerator,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    epoch: Optional[int] = None,
    total_epochs: Optional[int] = None,
) -> Dict[str, Any]:
    # Create a tqdm progress bar that disappears after completion
    epoch_str = f"Epoch {epoch}/{total_epochs}" if epoch is not None else "Training"
    progress_bar = tqdm(
        total=len(train_dataloader),
        desc=epoch_str,
        position=0,
        leave=False,
        dynamic_ncols=True,
    )

    running_loss = 0.0
    model.train()

    for i, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        if scheduler:
            scheduler.step()

        # Update the progress bar with loss info
        running_loss += loss.item()
        avg_loss = running_loss / (i + 1)
        progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
        progress_bar.update(1)

    # Close the progress bar after the epoch
    progress_bar.close()

    # Create and return metrics dictionary
    metrics = {
        "train_loss": avg_loss,
        "epoch": epoch if epoch is not None else 1,
        "steps": len(train_dataloader),
        "lr": optimizer.param_groups[0]["lr"],
    }
    return metrics


def train_loop(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    accelerator: Accelerator,
    optimizer: torch.optim.Optimizer,
    epochs: int = 1,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    on_train_epoch_end: Optional[Callable] = None,
):
    for epoch in range(epochs):
        # Run inner training loop for this epoch
        inner_train_loop(
            model=model,
            train_dataloader=train_dataloader,
            accelerator=accelerator,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            total_epochs=epochs,
        )
        if on_train_epoch_end is not None:
            on_train_epoch_end(epoch + 1)


def evaluate(
    model: torch.nn.Module, eval_dataloader: DataLoader, accelerator: Accelerator
) -> float:
    model.eval()
    progress_bar = tqdm(
        total=len(eval_dataloader),
        desc="Evaluating",
        position=0,
        leave=False,
        dynamic_ncols=True,
    )

    eval_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(eval_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.item()

            avg_loss = eval_loss / (i + 1)
            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
            progress_bar.update(1)

    progress_bar.close()
    model.train()
    return eval_loss / len(eval_dataloader)


def main():
    # Configuration
    args = parse_args()
    exp_name = get_experiment_name(vars(args))
    # Add timestamp to exp_name
    # exp_name += f"_{int(time.time())}"  -- remove to facilitate resume_from_checkpoint
    ckpt_dir = osp.join("checkpoints", exp_name)
    has_checkpoint = osp.exists(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    set_seed(args.seed)
    save_args(args, ckpt_dir)

    # Model and tokenizer
    # model, tokenizer = get_model_and_tokenizer(args)
    # chat_template = get_chat_template(args)

    # meta-llama/Llama-2-7b-chat-hf
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf", low_cpu_mem_usage=True
    )
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=4,
            lora_alpha=32,
            lora_dropout=0.1,
        ),
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    # Print dict key value pairs
    print("Model parameters:")
    print(
        "# Params (Trainable):",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    print("# Params (Total):", sum(p.numel() for p in model.parameters()))

    # Datasets
    lm_dataset = {
        "shakespeare": load_shakespeare_data,
        "wikitext": load_wikitext_data,
        "sharegpt": load_sharegptv2_data,
        "gsm8k": load_gsm8k_data,
        "stemp": load_syn_temp_data,
        "snum": load_syn_num_data,
        "sbase": load_syn_num_base_data,
    }[args.dataset](tokenizer, args.seq_len, max_num_samples=args.max_num_samples)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_dataloader = DataLoader(
        lm_dataset["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.batch_size,
    )
    eval_dataloader = DataLoader(
        lm_dataset["eval"],
        collate_fn=data_collator,
        batch_size=args.batch_size,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Add a learning rate scheduler if needed
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_dataloader))
    accelerator = Accelerator()
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # Train with the callback
    train_loop(
        model=model,
        train_dataloader=train_dataloader,
        accelerator=accelerator,
        optimizer=optimizer,
        epochs=3,
    )

    print("Training complete!")


if __name__ == "__main__":
    main()
