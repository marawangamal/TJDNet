import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchtext.datasets import PennTreebank
from torchtext.vocab import vocab
from collections import Counter, OrderedDict
from torch.nn import Transformer
import torch.optim as optim
import argparse


def build_vocab(data_iter):
    # Tokenizer that splits text into a list of characters
    tokenizer = lambda x: list(x)
    # Build vocabulary from iterator
    counter = Counter()
    for text in data_iter:
        counter.update(tokenizer(text))
    counter.update(
        {"<unk>": max(counter.values()) + 1, "<pad>": 0}
    )  # Ensure padding token is there
    sorted_vocab_items = sorted(counter.items(), key=lambda x: -x[1])
    vocab_obj = vocab(OrderedDict(sorted_vocab_items))
    vocab_obj.set_default_index(
        vocab_obj["<unk>"]
    )  # Set default index for unknown tokens
    return vocab_obj


def collate_batch(batch, vocab_obj, max_len=512):
    # Function to handle padding of batch
    batch = [
        torch.tensor([vocab_obj[char] for char in list(data[1])], dtype=torch.long)
        for data in batch
    ]
    # Pad sequences in the batch
    batch = torch.nn.utils.rnn.pad_sequence(
        batch, padding_value=vocab_obj["<pad>"], batch_first=True
    )
    # Optionally truncate sequences
    batch = batch[:, :max_len]
    return batch


class PTBDataloader

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
        default=0.01,
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
