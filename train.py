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
import string
import math
import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset
import wandb
from transformers import DataCollatorForLanguageModeling, get_scheduler


from models.tjdgpt2.tjdgpt2 import TJDGPT2
from models.tjdgpt2.tokenizer import Tokenizer
from utils import get_experiment_name


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on the ELI5 dataset.")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for training."
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Number of warmup steps for learning rate scheduler.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training and evaluation.",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=256,
        help="Block size for model input sequences.",
    )
    parser.add_argument(
        "--n_embd",
        type=int,
        default=384,
        help="Dimensionality of the model embeddings.",
    )
    parser.add_argument(
        "--n_layer",
        type=int,
        default=6,
        help="Number of hidden layers in the transformer model.",
    )
    parser.add_argument(
        "--n_head",
        type=int,
        default=6,
        help="Number of attention heads in the transformer model.",
    )
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate.")
    parser.add_argument(
        "--model",
        type=str,
        default="mps",
        help="Type of factorization to use for the model.",
        choices=[
            "cp",
            "mps",
            "full",
        ],
    )
    parser.add_argument(
        "--positivity_func",
        type=str,
        default="exp",
        choices=["sq", "abs", "exp"],
        help="Positivity function to use for MPSDist.",
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help="Rank of the tensor train decomposition.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=2,
        help="Block size for model input sequences.",
    )
    # Evaluation only arguments
    parser.add_argument(
        "--horizon_eval",
        type=int,
        default=1,
        help="Block size for model input sequences.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate during evaluation.",
    )
    return parser.parse_args()


def preprocess_shakespeare(examples):
    chars = list(examples["text"])
    return {"text": chars}


def tokenize(examples, tokenizer):
    return tokenizer(examples["text"], add_special_tokens=False)


def group_texts(examples, input_seq_len):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])  # type: ignore
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= input_seq_len:
        total_length = (total_length // input_seq_len) * input_seq_len
    # Split by chunks of input_seq_len.
    result = {
        k: [t[i : i + input_seq_len] for i in range(0, total_length, input_seq_len)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def load_shakespeare_data(tokenizer, input_seq_len, test_size=0.2):
    dataset = load_dataset("tiny_shakespeare", split="train")
    # d = d.map(preprocess_shakespeare)
    dataset = dataset.map(
        lambda x: tokenize(x, tokenizer),
        remove_columns=["text"],
    )
    dataset = dataset.map(lambda x: group_texts(x, input_seq_len), batched=True)
    dataset = dataset.train_test_split(test_size=test_size)  # type: ignore
    # DEBUG: print first example decoded
    # print(f"First example: \n{tokenizer.decode(dataset['train']['input_ids'][0])}")  # type: ignore
    return dataset


def get_test_sample(
    model,
    tokenizer,
    prompt="\n",
    max_new_tokens=8,
    # top_k=200,
    # temperature=0.8,
    num_beams=1,
    do_sample=False,
):
    # Inference
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    outputs = model.generate(
        inputs,
        num_beams=num_beams,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Train function
def train(
    model,
    train_dataloader,
    eval_dataloader,
    num_epochs=5,
    lr=2e-5,
    warmup_steps=100,
    n_eval_samples=3,
    max_new_tokens=8,
    eval_before_training=True,
    save_dir="checkpoints",
    model_config={},
    horizon_eval=1,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)  # type: ignore
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )
    best_eval_loss = float("inf")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    if eval_before_training:
        evaluate(model, eval_dataloader, epoch=0)

    for epoch in range(1, num_epochs + 1):
        for i in range(n_eval_samples):
            print(
                f"{get_test_sample(model, tokenizer, max_new_tokens=max_new_tokens)}\n-------------------\n"
            )
        model.train()
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch}",
            bar_format="{l_bar}{bar}| [Duration: {elapsed}][{postfix}]",
        )

        # Training loop
        for i, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            # Update progress bar with latest loss
            # Check if model has forward_avg_meter and loss_avg_meter
            model_fwd_time = 0
            model_loss_time = 0
            if hasattr(model, "forward_avg_meter") and hasattr(model, "loss_avg_meter"):
                model_fwd_time = model.forward_avg_meter.avg
                model_loss_time = model.loss_avg_meter.avg
            progress_bar.set_postfix(
                loss=f"{loss.item():.3f}",
                time_fwd_ms=f"{model_fwd_time:.3f}",
                time_loss_ms=f"{model_loss_time:.3f}",
            )
            if i % 100 == 0:
                print(
                    f"{get_test_sample(model, tokenizer, max_new_tokens=max_new_tokens)}\n-------------------\n"
                )

        # Evaluate model
        eval_loss = evaluate(model, eval_dataloader, epoch, horizon=horizon_eval)
        best_eval_loss = min(eval_loss, best_eval_loss)

        # Save model checkpoint
        if eval_loss <= best_eval_loss:
            print(f"Saving model checkpoint to {save_dir}")
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_config": model_config,
                    "epoch": epoch,
                    "eval_loss": eval_loss,
                },
                osp.join(save_dir, "best_model.pth"),
            )


def evaluate(
    model: torch.nn.Module,
    eval_dataloader: torch.utils.data.DataLoader,
    epoch: int,
    horizon: int = 1,
):
    model.eval()
    losses = []
    device = next(model.parameters()).device
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            loss = model(horizon=horizon, **batch)
        losses.append(loss.item())

    eval_loss = sum(losses) / len(losses)
    print(f"[Epoch {epoch + 1}] PPL: {math.exp(eval_loss):.2f} | Loss: {eval_loss:.2f}")
    wandb.log({"eval_loss": eval_loss, "epoch": epoch})
    return eval_loss


if __name__ == "__main__":

    args = parse_args()

    # Configuration
    exp_name = get_experiment_name(vars(args))
    # Make ckpt dir
    ckpt_dir = osp.join("checkpoints", exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Training

    characters = list(string.ascii_letters + string.digits + string.punctuation) + [
        "\n",
        " ",
        "\t",
    ]
    tokenizer = Tokenizer(characters, args.seq_len)

    # Sanity check tokenizer
    # print(f"Tokenizer test: {tokenizer.encode('Hello, my dog is cute.!')}")
    decoded = tokenizer.decode(
        tokenizer.encode("Hello, my dog is cute!"), skip_special_tokens=True
    )
    assert (
        decoded == "Hello, my dog is cute!"
    ), f"[FAIL] Tokenizer test failed: {decoded}"
    print(f"[PASS] Tokenizer test passed.")

    lm_dataset = load_shakespeare_data(tokenizer, args.seq_len)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Data loaders
    train_dataloader = DataLoader(
        lm_dataset["train"], shuffle=True, batch_size=args.batch_size, collate_fn=data_collator  # type: ignore
    )
    eval_dataloader = DataLoader(
        lm_dataset["test"], batch_size=args.batch_size, collate_fn=data_collator  # type: ignore
    )

    # Configuration for a smaller GPT2 model (baby GPT)
    model_config = {
        "model": args.model,
        "vocab_size": len(tokenizer),
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
    }

    model = TJDGPT2(**model_config)

    wandb.init(
        project="tjdnet-shakepeare-v2",
        config=vars(args),
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
    )

    # Generate a test sample
    sample_output = get_test_sample(model, tokenizer)
    print(sample_output)
