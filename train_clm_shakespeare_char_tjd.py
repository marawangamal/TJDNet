"""
Fine-tune GPT-2 on ELI5 dataset.

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

import string
import math
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
)
from transformers import DataCollatorForLanguageModeling, get_scheduler

from character_tokenizer import CharacterTokenizer
from TJDNet.TJDLayer.TTDist import TTDist

from utils.utils import get_entropy_loss, get_preference_loss


class TGPT2(torch.nn.Module):
    def __init__(
        self,
        config: GPT2Config,
        rank: int = 2,
        norm_method: str = "abs",
        eps: float = 1e-9,
    ):
        super().__init__()
        self.model = GPT2LMHeadModel(config)
        self.rank = rank
        self.norm_method = norm_method
        self.vocab_size = config.vocab_size
        self.eps = eps
        self.custom_unembedding = torch.nn.Linear(
            config.n_embd, config.vocab_size, bias=False
        )
        self.tensor_train_size = rank + (rank * config.vocab_size * rank) + rank
        self.seq2latents = torch.nn.Sequential(
            # Average pool the seq_len dimension
            torch.nn.Linear(config.n_embd, config.n_embd),
            torch.nn.ReLU(),
        )
        self.latent2tt = torch.nn.Linear(config.n_embd, self.tensor_train_size)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    @property
    def device(self):
        return next(self.parameters()).device

    def generate(self, input_ids, *args, max_new_tokens=8, **kwargs):
        learned_ttdist, transformer_outputs = self.get_tt_dist(
            input_ids,
            seq_len=max_new_tokens,
        )
        sample = learned_ttdist.sample()
        return sample

    def get_tt_dist(self, input_ids: torch.Tensor, seq_len=None, **kwargs):
        transformer_outputs = self.model.transformer(
            input_ids=input_ids,
            **kwargs,
        )
        hidden_states = transformer_outputs.last_hidden_state
        seq_len = hidden_states.size(1) if seq_len is None else seq_len
        alpha, beta, core = self.get_tt_params(hidden_states)

        # Forward pass:
        learned_ttdist = TTDist(
            alpha,
            beta,
            core,
            n_core_repititions=seq_len,
            norm_method=self.norm_method,
            norm_method_alpha=self.norm_method,
            eps=0.0,
        )
        return learned_ttdist, transformer_outputs

    def get_tt_params(self, hidden_states: torch.Tensor):
        # Map with linear layer
        batch_size, seq_len, hidden_size = hidden_states.size()
        tt_latent = self.seq2latents(hidden_states).mean(dim=1)
        tt_params = self.latent2tt(tt_latent)
        alpha, core, beta = torch.split(
            tt_params,
            [self.rank, self.rank * self.vocab_size * self.rank, self.rank],
            dim=-1,
        )
        alpha = alpha.reshape(batch_size, self.rank)
        beta = beta.reshape(batch_size, self.rank)
        core = core.reshape(batch_size, self.rank, self.vocab_size, self.rank)
        return alpha, beta, core

    def forward(self, input_ids, labels, *args, **kwargs):
        learned_ttdist, transformer_outputs = self.get_tt_dist(input_ids, **kwargs)
        loss, _ = get_preference_loss(
            learned_ttdist, samples=input_ids, eps=self.eps, vocab_size=self.vocab_size
        )
        transformer_outputs.loss = loss
        return transformer_outputs


def preprocess_shakespeare(examples):
    chars = list(examples["text"])
    return {"text": chars}


def tokenize(examples, tokenizer):
    return tokenizer(examples["text"], add_special_tokens=False)


def group_texts(examples, block_size):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])  # type: ignore
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def load_shakespeare_data(tokenizer, block_size, test_size=0.2):
    dataset = load_dataset("tiny_shakespeare", split="train")
    # d = d.map(preprocess_shakespeare)
    dataset = dataset.map(
        lambda x: tokenize(x, tokenizer),
        remove_columns=["text"],
    )
    dataset = dataset.map(lambda x: group_texts(x, block_size), batched=True)
    dataset = dataset.train_test_split(test_size=test_size)  # type: ignore
    # DEBUG: print first example decoded
    print(f"First example: \n{tokenizer.decode(dataset['train']['input_ids'][0])}")  # type: ignore
    return dataset


def get_test_sample(
    model,
    tokenizer,
    prompt="\n",
    max_new_tokens=8,
    top_k=200,
    temperature=0.8,
):
    # Inference
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=top_k,
        temperature=temperature,
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
            print(f"{get_test_sample(model, tokenizer)}\n-------------------\n")
        model.train()
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}",
            bar_format="{l_bar}{bar}| [Duration: {elapsed}][Loss: {postfix}]",
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
            progress_bar.set_postfix(loss=f"{loss.item():.3f}")

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


if __name__ == "__main__":

    # Configuration

    # Training
    lr = 1e-3
    warmup_steps = 100
    num_epochs = 50
    batch_size = 64
    block_size = 256

    # Model
    n_embd = 384
    n_layer = 6
    n_head = 6
    dropout = 0.2

    characters = list(string.ascii_letters + string.digits + string.punctuation) + [
        "\n",
        " ",
        "\t",
    ]
    tokenizer = CharacterTokenizer(characters, block_size)

    # Sanity check tokenizer
    # print(f"Tokenizer test: {tokenizer.encode('Hello, my dog is cute.!')}")
    decoded = tokenizer.decode(
        tokenizer.encode("Hello, my dog is cute!"), skip_special_tokens=True
    )
    assert (
        decoded == "Hello, my dog is cute!"
    ), f"[FAIL] Tokenizer test failed: {decoded}"

    lm_dataset = load_shakespeare_data(tokenizer, block_size)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Data loaders
    train_dataloader = DataLoader(
        lm_dataset["train"], shuffle=True, batch_size=8, collate_fn=data_collator  # type: ignore
    )
    eval_dataloader = DataLoader(
        lm_dataset["test"], batch_size=8, collate_fn=data_collator  # type: ignore
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
    model = TGPT2(config)

    train(
        model,
        train_dataloader,
        eval_dataloader,
        num_epochs=num_epochs,
        lr=lr,
        warmup_steps=warmup_steps,
    )

    # Generate a test sample
    sample_output = get_test_sample(model, tokenizer)
    print(sample_output)
