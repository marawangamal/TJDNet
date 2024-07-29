from typing import Iterable
from torchtext.vocab import vocab, Vocab

import torch
from collections import Counter, OrderedDict


def build_vocab(data_iter: Iterable[str]) -> Vocab:
    """Builds a vocabulary object from the data iterator

    Args:
        data_iter (Iterable[str]): Iterable of strings

    Returns:
        Vocab: PyTorchText Vocab object
    """
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


def decode_sequence(vocab_obj: Vocab, sequence: Iterable[int]) -> str:
    """Decodes a sequence of tokens into a string

    Args:
        vocab_obj (Vocab): PyTorchText Vocab object
        sequence (Iterable[int]): Sequence of token IDs

    Returns:
        str: Decoded string
    """
    decoded_string = "".join([vocab_obj.get_itos()[token_id] for token_id in sequence])
    return decoded_string


def decode_sequence_from_tokenizer(tokenizer, sequence: Iterable[int]) -> str:
    """Decodes a sequence of tokens into a string

    Args:
        vocab_obj (Vocab): PyTorchText Vocab object
        sequence (Iterable[int]): Sequence of token IDs

    Returns:
        str: Decoded string
    """
    decoded_string = tokenizer.decode(sequence)
    return decoded_string


def collate_batch_from_vocab(
    batch: Iterable[str], vocab_obj: Vocab, max_len: int = 100
) -> torch.Tensor:
    """Collates a batch of strings into a padded tensor

    Args:
        batch (Iterable[str]): Batch of strings
        vocab_obj (Vocab): PyTorchText Vocab object
        max_len (int, optional):  Maximum length of the sequence. Defaults to 100.

    Returns:
        torch.Tensor: Padded tensor of token IDs
    """
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


def collate_batch_from_tokenizer(batch, tokenizer):
    texts = batch
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    # sanity_check = decode_sequence_from_tokenizer(tokenizer, inputs["input_ids"][0])
    return inputs["input_ids"], inputs["attention_mask"]
