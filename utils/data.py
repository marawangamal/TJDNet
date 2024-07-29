import os

from typing import Union
from datasets import load_dataset, Dataset, DatasetDict
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers import DataCollatorForSeq2Seq, AutoTokenizer
from torch.utils.data import DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Tokenize the dataset and set labels
def tokenize_and_set_labels(examples, tokenizer, max_length=512):
    tokenized_inputs = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs


def get_wikitext2_dataloaders(
    cache_dir: str,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    max_length: int = 512,
    batch_size: int = 2,
):
    """Prepares data loaders for the Wikitext-2 dataset using a specified tokenizer.

    WARNING: This function sets 'labels' directly as 'input_ids' without any shifting for
    next-token prediction. Any necessary shifting for specific model requirements must be
    implemented separately.

    Args:
        cache_dir (str): Directory for storing downloaded datasets.
        tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): Tokenizer for text processing.
        max_length (int, optional): Maximum sequence length for tokenization. Defaults to 512.
        batch_size (int, optional): Number of samples per batch. Defaults to 2.

    Returns:
        tuple: Contains three DataLoaders for the train, validation, and test sets.
    """
    # Dataset config
    dataset_name = "Salesforce/wikitext"
    subset = "wikitext-2-raw-v1"
    train_split = "train"
    valid_split = "validation"
    test_split = "test"
    columns_to_keep = ["input_ids", "attention_mask", "labels"]

    # Load the dataset
    dataset = load_dataset(dataset_name, subset, cache_dir=cache_dir)
    assert isinstance(
        dataset, DatasetDict
    ), "Expected a DatasetDict, got something else."
    train_dataset, valid_dataset, test_dataset = [
        dataset[split] for split in [train_split, valid_split, test_split]
    ]

    # Tokenize the dataset
    train_tokenized, valid_tokenized, test_tokenized = [
        dataset.map(
            lambda examples: tokenize_and_set_labels(
                examples, tokenizer=tokenizer, max_length=max_length
            ),
            batched=True,
        )
        for dataset in [train_dataset, valid_dataset, test_dataset]
    ]

    # Chore: Remove columns that are not needed
    train_tokenized, valid_tokenized, test_tokenized = [
        dataset.remove_columns(
            [column for column in dataset.column_names if column not in columns_to_keep]
        )
        for dataset in [train_tokenized, valid_tokenized, test_tokenized]
    ]

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, return_tensors="pt", padding=True
    )

    train_dataloader = DataLoader(
        train_tokenized,  # type: ignore
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
        pin_memory=True,
        num_workers=2,
    )

    valid_dataloader, test_dataloader = [
        DataLoader(
            dataset,  # type: ignore
            batch_size=2,
            collate_fn=data_collator,
        )
        for dataset in [valid_tokenized, test_tokenized]
    ]

    # Debug: Print a sample sentence
    sample = next(iter(test_dataloader))
    decoded_sample_inp = tokenizer.decode(sample["input_ids"][1])[:50]
    decoded_sample_lbl = tokenizer.decode(sample["labels"][1])[:50]
    print(f"Decoded sample (input): ``{decoded_sample_inp}``")
    print(f"Decoded sample (labels): ``{decoded_sample_lbl}``")

    return train_dataloader, valid_dataloader, test_dataloader


if __name__ == "__main__":
    cache_dir = "./data"
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    get_wikitext2_dataloaders(cache_dir, tokenizer)
