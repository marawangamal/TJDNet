from collections import OrderedDict
import os.path as osp
import os
import time
from tqdm import tqdm

import torch
import wandb
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, get_scheduler

from helpers import (
    get_git_info,
    get_model_and_tokenizer,
    get_test_samples,
    parse_args,
    save_args,
    set_seed,
)
from data.shakespeare import load_shakespeare_data
from data.wikitext import load_wikitext_data
from data.sharegpt import load_sharegpt_data
from utils import get_experiment_name, AverageMeter


def evaluate(
    model: torch.nn.Module,
    eval_dataloader: DataLoader,
    horizon: int = 1,
):
    model.eval()
    loss_meter = AverageMeter()
    nll_meter = AverageMeter()
    device = next(model.parameters()).device
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            output_dict = model(horizon=horizon, **batch)
            loss, nll = output_dict["loss"], output_dict["nll"]
            loss_meter.update(loss.item())
            nll_meter.update(nll.item())
    return loss_meter.avg, nll_meter.avg


# Train function
def train(
    model,
    train_dataloader,
    eval_dataloader,
    num_epochs=5,
    lr=2e-5,
    warmup_steps=100,
    n_eval_samples=3,
    max_new_tokens=128,
    eval_before_training=True,
    save_dir="checkpoints",
    horizon_eval=1,
    grad_clip_val=None,
    use_loss_scale=False,
    wandb_run=None,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)  # type: ignore
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    epochs_range = range(num_epochs + 1) if eval_before_training else range(num_epochs)
    text_table = wandb.Table(columns=["epoch", "eval/nll", "text"])
    train_start_time = time.time()
    for epoch in epochs_range:
        epoch_start_time = time.time()
        train_loss_meter = AverageMeter()  # Create new meter each epoch
        train_nll_meter = AverageMeter()
        model.train()
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch}",
        )

        # Training loop
        for _, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}
            output_dict = model(**batch)
            loss, nll, loss_scale = (
                output_dict["loss"],  # (B, T) -> (1,)
                output_dict["nll"],
                output_dict["loss_scale"],
            )
            scaled_loss = loss * loss_scale if use_loss_scale else loss
            train_loss_meter.update(loss.item())
            train_nll_meter.update(nll.item())
            # Skip training first epoch if eval_before_training is True
            if eval_before_training and epoch == 0:
                continue
            scaled_loss.backward()
            if grad_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.set_postfix(
                OrderedDict(
                    [
                        ("epoch_time", f"{time.time() - epoch_start_time:.2f}"),
                        ("train_time", f"{time.time() - train_start_time:.2f}"),
                        ("loss", f"{loss.item():.3f}"),
                    ]
                )
            )

        eval_loss, eval_nll = evaluate(
            model=model, eval_dataloader=eval_dataloader, horizon=horizon_eval
        )
        # TODO: Align with Trainer class
        # # Save model checkpoint
        # if eval_nll <= best_eval_nll:
        #     print(f"Saving model checkpoint to {save_dir}")
        #     ckpt_dir = osp.join(save_dir, f"checkpoint-{epoch}")
        #     os.makedirs(ckpt_dir)
        #     torch.save(
        #         sta
        #     torch.save(
        #         {
        #             "state_dict": model.state_dict(),
        #             "train/epoch": epoch,
        #             "eval/nll": eval_nll,
        #             "eval/loss": eval_loss,
        #         },
        #         osp.join(save_dir, f"checkpoint-{epoch}.pt"),
        #     )

        # Log metrics to wandb
        elapsed_mins = (time.time() - train_start_time) / 60
        print(
            f"[Epoch {epoch + 1}] Train Loss: {train_loss_meter.avg:.2f} | Elapsed: {elapsed_mins} | Eval Loss: {eval_nll:.2f}"
        )
        wandb.log({"train/loss": train_loss_meter.avg, "train/epoch": epoch})
        wandb.log({"train/nll": train_nll_meter.avg, "train/epoch": epoch})
        wandb.log({"eval/loss": eval_loss, "train/epoch": epoch})
        wandb.log({"eval/nll": eval_nll, "train/epoch": epoch})

        sample = get_test_samples(
            model,
            tokenizer,
            max_new_tokens=max_new_tokens,
            horizon=horizon_eval,
            print_output=True,
        )
        text_table.add_data(epoch, eval_nll, sample)

    if wandb_run is not None:
        print("Number of rows in table:", len(text_table.data))
        wandb_run.log({"training_samples": text_table})


if __name__ == "__main__":

    # Configuration
    args = parse_args()
    exp_name = get_experiment_name(vars(args))
    ckpt_dir = osp.join("checkpoints", exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    set_seed(args.seed)
    save_args(args, ckpt_dir)

    # Model and tokenizer
    model, tokenizer = get_model_and_tokenizer(args)

    # Datasets
    lm_dataset = {
        "shakespeare": load_shakespeare_data,
        "wikitext": load_wikitext_data,
        "sharegpt": load_sharegpt_data,
    }[args.dataset](tokenizer, args.seq_len)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Data loaders
    train_dataloader = DataLoader(
        lm_dataset["train"], shuffle=True, batch_size=args.batch_size, collate_fn=data_collator  # type: ignore
    )
    eval_dataloader = DataLoader(
        lm_dataset["test"], batch_size=args.batch_size, collate_fn=data_collator  # type: ignore
    )

    wandb_run = wandb.init(
        project="tjdnet-shakepeare-debug",
        config={**vars(args), **get_git_info()},
        name=exp_name,
    )

    train(
        model,
        train_dataloader,
        eval_dataloader,
        num_epochs=args.epochs,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        max_new_tokens=args.max_new_tokens,
        save_dir=ckpt_dir,
        horizon_eval=args.horizon_eval,
        grad_clip_val=args.grad_clip_val,
        wandb_run=wandb_run,
    )

    # Generate a test sample
    final_sample = get_test_samples(model, tokenizer, print_output=False)
