"""
Fine-tune GPT-2 using TJDNet on Shakespeare dataset.

Example usage:
python train_clm_shakespeare_char_tjd.py --model cpgpt2 

Resources: 
https://huggingface.co/docs/transformers/tasks/language_modeling
https://github.com/dariush-bahrami/character-tokenizer/tree/master


Two options for data loading:

Given a dataset of sequences of different length {s1, s2, ..., s2}, we have two options for dataloading

1. Simple (preprocess_simple)
    - Convert each sequence to be of length `max_len` via padding or trunction 

2. Advanced (preprocess_function & group texts)
    - Combine to sinlge length string s = [s_1, s_2, ..., s_b], then split into chunks of size `max_len`. This is less 
    - Less wastefulness from truncation


"""

import os.path as osp
import os
import time
from tqdm import tqdm

import torch
import wandb
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, get_scheduler
from transformers import AutoTokenizer

from helpers import get_git_info, parse_args, set_seed
from models.tjdgpt2 import TJDGPT2
from ctokenizers.char_tokenizer import CharTokenizer
from data.shakespeare import load_shakespeare_data
from data.wikitext import load_wikitext_data
from utils import get_experiment_name, AverageMeter


def get_test_samples(
    model,
    tokenizer,
    prompt="\n",
    # max_new_tokens=8,
    # num_beams=5,
    # do_sample=True,
    # top_k=50,
    max_new_tokens=128,
    top_k=200,
    # temperature=0.8,
    num_beams=1,
    do_sample=True,
    horizon_eval=1,
    n_samples=1,
    print_output=True,
):
    # Inference
    model.eval()
    samples = []
    for i in range(n_samples):
        inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        outputs = model.generate(
            inputs,
            num_beams=num_beams,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            horizon=horizon_eval,
            top_k=top_k,
        )
        sample = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if n_samples == 1:
            samples.append(sample)
        else:
            samples.append(f"[{i+1}] {sample}")

    if print_output:
        print("\n---\n".join(samples) + "\n")
    return "\n".join(samples)


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
    model_config={},
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
    best_eval_nll = float("inf")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    epochs_range = range(num_epochs + 1) if eval_before_training else range(num_epochs)
    start_time = time.time()
    text_table = wandb.Table(columns=["epoch", "eval/nll", "text"])
    for epoch in epochs_range:
        train_loss_meter = AverageMeter()  # Create new meter each epoch
        train_nll_meter = AverageMeter()
        model.train()
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch}",
            bar_format="{l_bar}{bar}| [Duration: {elapsed}][{postfix}]",
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
            progress_bar.set_postfix(loss=f"{loss.item():.3f}")

        eval_loss, eval_nll = evaluate(
            model=model, eval_dataloader=eval_dataloader, horizon=horizon_eval
        )
        best_eval_nll = min(eval_nll, best_eval_nll)

        # Save model checkpoint
        if eval_nll <= best_eval_nll:
            print(f"Saving model checkpoint to {save_dir}")
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_config": model_config,
                    "train/epoch": epoch,
                    "eval/nll": eval_nll,
                    "eval/loss": eval_loss,
                },
                osp.join(save_dir, "best_model.pth"),
            )

        # Log metrics to wandb
        elapsed_mins = (time.time() - start_time) / 60
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
            horizon_eval=horizon_eval,
            print_output=True,
        )
        text_table.add_data(epoch, eval_nll, sample)

    if wandb_run is not None:
        print("Number of rows in table:", len(text_table.data))
        wandb_run.log({"training_samples": text_table})


if __name__ == "__main__":

    args = parse_args()
    git_info = get_git_info()
    set_seed(args.seed)

    # Configuration
    exp_name = get_experiment_name(vars(args))
    ckpt_dir = osp.join("checkpoints", exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Tokenizer
    tokenizer = (
        AutoTokenizer.from_pretrained("gpt2")
        if args.tokenizer_type == "word"
        else CharTokenizer(args.seq_len)
    )
    if args.tokenizer_type == "word":
        tokenizer.pad_token = tokenizer.eos_token

    # Datasets
    lm_dataset = {
        "shakespeare": load_shakespeare_data,
        "wikitext": load_wikitext_data,
    }[args.dataset](tokenizer, args.seq_len)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Data loaders
    train_dataloader = DataLoader(
        lm_dataset["train"], shuffle=True, batch_size=args.batch_size, collate_fn=data_collator  # type: ignore
    )
    eval_dataloader = DataLoader(
        lm_dataset["test"], batch_size=args.batch_size, collate_fn=data_collator  # type: ignore
    )

    # Model
    model_config = {
        "model_head": args.model_head,
        "vocab_size": (
            len(tokenizer.get_vocab())
            if hasattr(tokenizer, "get_vocab")
            else len(tokenizer)
        ),
        "n_embd": args.n_embd,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "dropout": args.dropout,
        "rank": args.rank,
        "horizon": args.horizon,
        "positivity_func": args.positivity_func,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "freeze_base_model": args.freeze_base_model,
    }
    model = TJDGPT2(**model_config)

    wandb_run = wandb.init(
        project="tjdnet-shakepeare-debug",
        config={**vars(args), **git_info},
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
        model_config=model_config,
        horizon_eval=args.horizon_eval,
        grad_clip_val=args.grad_clip_val,
        use_loss_scale=args.scale_loss,
        wandb_run=wandb_run,
    )

    # Generate a test sample
    final_sample = get_test_samples(model, tokenizer, print_output=False)
