from typing import Optional
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from tqdm import tqdm

import argparse
import wandb

from utils.data import get_wikitext2_dataloaders
from utils.utils import get_experiment_name


def count_batches(data_loader):
    count = 0
    for _ in data_loader:
        count += 1
    return count


class HFTransformerModel(nn.Module):
    def __init__(
        self,
    ):
        super(HFTransformerModel, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.unembedding = nn.Linear(
            self.model.config.n_embd, self.model.config.vocab_size
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 32,
    ):
        """Generate new tokens from the model.

        Args:
            input_ids (torch.Tensor): Input tensor of shape (batch_size, seq_len).
            attention_mask (torch.Tensor, optional): Attention mask tensor of shape (batch_size, seq_len). Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len).
        """
        out = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        inputs = [
            self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids
        ]
        decoded = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in out]
        decoded_raw = [
            dec[len(inp) :] if dec.startswith(inp) else dec
            for inp, dec in zip(inputs, decoded)
        ]
        return inputs, decoded_raw

    def forward(self, input_ids, labels, attention_mask=None):
        outputs = self.model(
            input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )
        return outputs.loss


def train(model, train_loader, device, optimizer, num_epochs):
    model.train()
    decoded_input = ""
    decoded_output = ""
    for epoch in range(num_epochs):
        total_loss = 0
        with tqdm(train_loader, unit="batch") as tepoch:
            for batch in tepoch:
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = model(**batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                if tepoch.n % 100 == 0:
                    decoded_inputs, decoded_outputs = model.generate(batch["input_ids"])

                tepoch.set_postfix(
                    loss=total_loss / (tepoch.n + 1),
                    # sin=decoded_inputs[0][:32],
                    # sout=decoded_outputs[0][:32],
                )

        print(f"Epoch {epoch+1}: Loss {total_loss / len(train_loader)}")
        print(f"Input: {decoded_inputs[0]}")
        print(f"Output: {decoded_outputs[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Transformer model on the Penn Treebank dataset at character level."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=20,
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--embed_size",
        type=int,
        default=512,
        help="size of the embeddings (default: 512)",
    )
    parser.add_argument(
        "--nhead",
        type=int,
        default=8,
        help="number of heads in the nn.Transformer (default: 8)",
    )
    parser.add_argument(
        "--ffn_hid_dim",
        type=int,
        default=2048,
        help="dimension of the feedforward network model (default: 2048)",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=3,
        help="number of nn.Transformer layers (default: 3)",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="./data",
        help="root directory for storing the dataset (default: ./data)",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=32,
        help="maximum length of the input sequence (default: 512)",
    )

    args = parser.parse_args()

    model = HFTransformerModel()
    tokenizer = model.tokenizer

    train_loader, valid_loader, test_loader = get_wikitext2_dataloaders(
        cache_dir=args.root_dir,
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    model.to(device)

    experiment_config = args.__dict__
    experiment_name = get_experiment_name(experiment_config)

    # Run training
    wandb.init(
        project="tjdnet-transformer",
        config=experiment_config,
        name=experiment_name,
    )
    print(f"Running experiment: {experiment_name}")
    train(
        model,
        train_loader,
        device,
        optimizer,
        args.num_epochs,
    )
    wandb.finish()
