import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchtext.datasets import PennTreebank
from torchtext.vocab import vocab
from collections import Counter, OrderedDict
from transformers import GPT2Model, GPT2Config
import torch.optim as optim
import argparse
from tqdm import tqdm
import wandb

from utils.utils import get_experiment_name


def build_vocab(data_iter):
    counter = Counter()
    for text in data_iter:
        counter.update(text)
    # Adding high count for '<unk>' and '<pad>'
    counter.update({"<unk>": 1e5, "<pad>": 1e5})
    sorted_vocab_items = sorted(counter.items(), key=lambda x: -x[1])
    vocab_obj = vocab(OrderedDict(sorted_vocab_items))
    vocab_obj.set_default_index(
        vocab_obj["<unk>"]
    )  # Set default index for unknown tokens
    return vocab_obj


def count_batches(data_loader):
    count = 0
    for _ in data_loader:
        count += 1
    return count


def decode_sequence(vocab_obj, sequence):
    decoded_string = "".join([vocab_obj.get_itos()[token_id] for token_id in sequence])
    return decoded_string


def collate_batch(batch, vocab_obj, max_len=512):
    # Function to handle padding of batch
    batch_out = [
        torch.tensor([vocab_obj[char] for char in list(data)], dtype=torch.long)
        for data in batch
    ]
    # Pad sequences in the batch
    batch_out = torch.nn.utils.rnn.pad_sequence(
        batch_out, padding_value=vocab_obj["<pad>"], batch_first=True
    )
    batch_out = batch_out[:, :max_len]
    return batch_out


class HFTransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers):
        super(HFTransformerModel, self).__init__()
        config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=embed_size,
            n_layer=num_layers,
            n_head=embed_size // 64,  # Assuming embed size is divisible by 64
        )
        self.transformer = GPT2Model(config)
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, src):
        outputs = self.transformer(src, return_dict=True)
        output = outputs.last_hidden_state
        return self.fc_out(output)


def train(
    model, train_loader, device, optimizer, criterion, num_epochs, total_batches=None
):
    total_batches = count_batches(train_loader)
    decoded_input = ""
    decoded_output = ""
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        with tqdm(train_loader, total=total_batches) as t:
            for i, data in enumerate(t):
                inputs, targets = (
                    data[:, :-1],
                    data[:, 1:],
                )  # Shifted by one for next character prediction
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                output = model(inputs)
                loss = criterion(
                    output.reshape(-1, model.fc_out.out_features), targets.reshape(-1)
                )
                # Clip gradients to avoid exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                # Log to W&B
                wandb.log(
                    {
                        "Loss": loss.item(),
                    }
                )
                if i % 50 == 0:
                    sample_output = model(inputs[0].unsqueeze(0))
                    decoded_input = decode_sequence(vocab_obj, inputs[0])
                    decoded_output = decode_sequence(
                        vocab_obj, sample_output.argmax(-1)[0]
                    )
                t.set_postfix(
                    loss=loss.item(), sin=decoded_input[:30], sout=decoded_output[:30]
                )
        print(
            f"[Epoch {epoch + 1}], Loss: {total_loss / total_batches}, Input: {decoded_input}, Output: {decoded_output}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Transformer model on the Penn Treebank dataset at character level."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
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

    args = parser.parse_args()

    # Initialize dataset and build vocabulary
    train_iter = PennTreebank(split="train", root=args.root_dir)
    vocab_obj = build_vocab(train_iter)

    # Reinitialize the dataset iterator for DataLoader
    train_iter = PennTreebank(split="train", root=args.root_dir)
    train_loader = DataLoader(
        train_iter,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, vocab_obj),
    )

    # Device and model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HFTransformerModel(len(vocab_obj), args.embed_size, args.num_layers).to(
        device
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

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
        criterion,
        args.num_epochs,
    )
    wandb.finish()
