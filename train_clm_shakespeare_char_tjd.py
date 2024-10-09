"""
Fine-tune GPT-2 using TJDNet on Shakespeare dataset.

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

from typing import Literal
import string
import math
import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
)
import wandb
from transformers import DataCollatorForLanguageModeling, get_scheduler

from character_tokenizer import CharacterTokenizer
from TJDNet import MPSDistBase
from TJDNet.loss import get_entropy_loss_stable, get_entropy_loss_stable_debug
from TJDNet.utils import window_input_ids, AverageMeter


from utils import get_experiment_name


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on the ELI5 dataset.")
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
        "--num_epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training and evaluation.",
    )
    parser.add_argument(
        "--input_seq_len",
        type=int,
        default=256,
        help="Block size for model input sequences.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=2,
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
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help="Rank of the tensor train decomposition.",
    )
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate.")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Type of model to use (gpt2 or tgpt2).",
        choices=["gpt2", "tgpt2"],
    )
    parser.add_argument(
        "--positivity_func",
        type=str,
        default="exp",
        choices=["sq", "abs", "exp"],
        help="Positivity function to use for MPSDist.",
    )

    # Evaluation only arguments
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate during evaluation.",
    )
    return parser.parse_args()


class TGPT2(torch.nn.Module):
    def __init__(
        self,
        config: GPT2Config,
        rank: int = 2,
        eps: float = 1e-9,
        horizon: int = 8,
        tokenizer=None,
        positivity_func: str = "sq",
    ):
        super().__init__()
        self.model = GPT2LMHeadModel(config)
        self.rank = rank
        self.positivity_func = positivity_func
        self.vocab_size = config.vocab_size
        self.eps = eps
        self.custom_unembedding = torch.nn.Linear(
            config.n_embd, config.vocab_size, bias=False
        )
        self.tokenizer = tokenizer
        self.tensor_train_size = rank + (rank * config.vocab_size * rank) + rank
        self.seq2latents = torch.nn.Sequential(
            # Average pool the seq_len dimension
            torch.nn.Linear(config.n_embd, config.n_embd),
            torch.nn.ReLU(),
        )
        self.latent2tt = torch.nn.Linear(config.n_embd, self.tensor_train_size)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.horizon = horizon

        self.forward_avg_meter = AverageMeter()
        self.loss_avg_meter = AverageMeter()

    @property
    def device(self):
        return next(self.parameters()).device

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 8, **kwargs):
        """_summary_

        Args:
            input_ids (torch.Tensor): Input prompt tensor of shape (B, T).
            max_new_tokens (int, optional): Maximum number of tokens to generate. Defaults to 8.

        Returns:
            torch.Tensor: Generated tokens of shape (B, max_new_tokens).
        """

        batch_size, seq_len = input_ids.size()
        n_passes = max_new_tokens // self.horizon
        output_tens = torch.empty(batch_size, 0, dtype=torch.long).to(self.device)
        input_tens = input_ids

        for _ in range(n_passes):
            transformer_outputs = self.model.transformer(
                input_ids=input_tens,
            )
            hidden_states = transformer_outputs.last_hidden_state

            alpha, beta, core = self.get_tt_params(
                hidden_states[:, -1:, :]
            )  # (B, 1, R, D, R)
            _, seq_len_adj, rank, vocab_size, _ = core.size()

            # Forward pass:
            learned_mpsdist = MPSDistBase(
                alpha.reshape(batch_size * seq_len_adj, -1),
                beta.reshape(batch_size * seq_len_adj, -1),
                core.reshape(batch_size * seq_len_adj, rank, vocab_size, rank),
                positivity_func=self.positivity_func,
            )

            sample = learned_mpsdist.sample(max_len=self.horizon)  # (B, H)
            output_tens = torch.cat([output_tens, sample], dim=1)
            input_tens = torch.cat([input_tens, sample], dim=1)
        return output_tens

    # todo: use delta_core
    def get_tt_dist(self, input_ids: torch.Tensor, **kwargs):
        transformer_outputs = self.model.transformer(
            input_ids=input_ids,
            **kwargs,
        )

        hidden_states = transformer_outputs.last_hidden_state
        alpha, beta, core = self.get_tt_params(
            hidden_states[:, : -self.horizon, :]
        )  # (B, T-H, R), (B, T-H, R), (B, T-H, R, D, R)
        batch_size, seq_len_adj, rank, vocab_size, _ = core.size()

        # Forward pass:
        learned_mpsdist = MPSDistBase(
            alpha.reshape(batch_size * seq_len_adj, -1),
            beta.reshape(batch_size * seq_len_adj, -1),
            core.reshape(batch_size * seq_len_adj, rank, vocab_size, rank),
            positivity_func=self.positivity_func,
        )

        # 1. Window the `input_ids` to get targets: (B, T) => (B, T, H)
        #   each position should look H steps ahead
        input_ids_windowed = window_input_ids(input_ids, horizon=self.horizon)

        # 2. Make targets using windowed input_ids
        targets = input_ids_windowed[:, : -self.horizon]  # (B, T-H, H)
        targets = targets.reshape(-1, self.horizon)  # (B * (T-H), H)

        return learned_mpsdist, transformer_outputs, targets

    def get_tt_params(self, hidden_states: torch.Tensor):
        # Map with linear layer
        batch_size, seq_len, hidden_size = hidden_states.size()
        tt_latent = self.seq2latents(
            hidden_states
        )  # (batch_size, seq_len, hidden_size)
        tt_params = self.latent2tt(
            tt_latent
        )  # (batch_size, seq_len, tensor_train_size)
        alpha, core, beta = torch.split(
            tt_params,
            [self.rank, self.rank * self.vocab_size * self.rank, self.rank],
            dim=-1,
        )
        alpha = alpha.reshape(batch_size, seq_len, self.rank)
        beta = beta.reshape(batch_size, seq_len, self.rank)
        core = core.reshape(batch_size, seq_len, self.rank, self.vocab_size, self.rank)
        return alpha, beta, core

    def forward(self, input_ids, labels, *args, **kwargs):

        learned_mpsdist, transformer_outputs, targets = self.get_tt_dist(
            input_ids, **kwargs
        )
        loss = get_entropy_loss_stable(
            learned_mpsdist,
            targets=targets,
            eps=self.eps,
        )

        # DEBUG: entropy loss works so no issues with `alpha`, `beta`, `core` and sampling
        # probs_tilde = learned_mpsdist.materialize(
        #     n_core_repititions=self.horizon, normalize=False
        # )
        # loss = get_entropy_loss_stable_debug(
        #     probs_tilde,
        #     targets.flatten(),
        # )
        transformer_outputs.loss = loss

        torch.cuda.synchronize()
        return transformer_outputs


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

    for epoch in range(num_epochs):
        for i in range(n_eval_samples):
            print(
                f"{get_test_sample(model, tokenizer, max_new_tokens=max_new_tokens)}\n-------------------\n"
            )
        model.train()
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}",
            bar_format="{l_bar}{bar}| [Duration: {elapsed}][{postfix}]",
        )
        for i, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
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

        model.eval()
        losses = []
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(loss.item())

        eval_loss = sum(losses) / len(losses)
        print(
            f"[Epoch {epoch + 1}] PPL: {math.exp(eval_loss):.2f} | Loss: {eval_loss:.2f}"
        )
        wandb.log({"eval_loss": eval_loss, "epoch": epoch + 1})


if __name__ == "__main__":

    args = parse_args()

    # Configuration

    # Training
    lr = args.lr
    warmup_steps = args.warmup_steps
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    input_seq_len = args.input_seq_len

    # Model
    n_embd = args.n_embd
    n_layer = args.n_layer
    n_head = args.n_head
    dropout = args.dropout

    characters = list(string.ascii_letters + string.digits + string.punctuation) + [
        "\n",
        " ",
        "\t",
    ]
    tokenizer = CharacterTokenizer(characters, input_seq_len)

    # Sanity check tokenizer
    # print(f"Tokenizer test: {tokenizer.encode('Hello, my dog is cute.!')}")
    decoded = tokenizer.decode(
        tokenizer.encode("Hello, my dog is cute!"), skip_special_tokens=True
    )
    assert (
        decoded == "Hello, my dog is cute!"
    ), f"[FAIL] Tokenizer test failed: {decoded}"
    print(f"[PASS] Tokenizer test passed.")

    lm_dataset = load_shakespeare_data(tokenizer, input_seq_len)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Data loaders
    train_dataloader = DataLoader(
        lm_dataset["train"], shuffle=True, batch_size=batch_size, collate_fn=data_collator  # type: ignore
    )
    eval_dataloader = DataLoader(
        lm_dataset["test"], batch_size=batch_size, collate_fn=data_collator  # type: ignore
    )

    # Configuration for a smaller GPT2 model (baby GPT)
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        dropout=dropout,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = (
        TGPT2(
            config,
            rank=args.rank,
            horizon=args.horizon,
            positivity_func=args.positivity_func,
            tokenizer=tokenizer,
        )
        if args.model == "tgpt2"
        else GPT2LMHeadModel(config)
    )

    wandb.init(
        project="tjdnet-shakepeare",
        config=vars(args),
        name=get_experiment_name(vars(args)),
    )

    train(
        model,
        train_dataloader,
        eval_dataloader,
        num_epochs=num_epochs,
        lr=lr,
        warmup_steps=warmup_steps,
        max_new_tokens=args.max_new_tokens,
    )

    # Generate a test sample
    sample_output = get_test_sample(model, tokenizer)
    print(sample_output)
