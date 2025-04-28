# sharegpt.py

import os
from datasets import load_dataset

from dataloaders.common import group_texts
from dataloaders._base import HF_CACHE_DIR, setup


class ChatTemplateGithub:
    @classmethod
    def format_prompt(cls, prompt: str) -> str:
        return prompt


def prepare_sample(tokenizer, example):
    tokenized = tokenizer(
        example["code"] + tokenizer.eos_token, add_special_tokens=True
    )
    return tokenized


def load_github_data(
    tokenizer, input_seq_len, test_size=0.01, max_num_samples=68000, **kwargs
):
    dataset = load_dataset(
        "codeparrot/github-code",
        split="train",
        languages=["Python"],
        cache_dir=os.path.join(HF_CACHE_DIR, "datasets"),
    )
    dataset = dataset.select(range(max_num_samples))
    dataset = dataset.map(
        lambda x: prepare_sample(x, tokenizer),
        remove_columns=["code"],
    )
    dataset = dataset.map(lambda x: group_texts(x, input_seq_len), batched=True)
    dataset = dataset.train_test_split(test_size=test_size)
    return dataset


# def load_github_data(
#     tokenizer, input_seq_len, test_size=0.01, max_num_samples=68000, **kwargs
# ):
#     # Calculate sizes for train/test
#     train_size = int(max_num_samples * (1 - test_size))
#     test_size = max_num_samples - train_size

#     # Load separate streaming splits
#     train_dataset = load_dataset(
#         "codeparrot/github-code",
#         streaming=True,
#         split=f"train[:{train_size}]",
#         languages=["Python"],
#     )

#     test_dataset = load_dataset(
#         "codeparrot/github-code",
#         streaming=True,
#         split=f"train[{train_size}:{train_size + test_size}]",
#         languages=["Python"],
#     )

#     # Process each split
#     for split in [train_dataset, test_dataset]:
#         split = split.map(
#             lambda x: prepare_sample(x, tokenizer),
#             remove_columns=["code"],
#         )
#         split = split.map(lambda x: group_texts(x, input_seq_len), batched=True)

#     return {"train": train_dataset, "test": test_dataset}


# Usage example:
if __name__ == "__main__":
    setup()
    from transformers import AutoTokenizer

    # Example usage
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_github_data(
        tokenizer=tokenizer,
        input_seq_len=512,
        max_num_samples=100,
    )

    print(f"\nDataset sizes:")
    print(f"Train: {len(dataset['train'])} sequences")
    print(f"Test: {len(dataset['test'])} sequences")
    print(f"EOS token: {tokenizer.eos_token}")

    # Find an example with EOS token
    for batch_idx, example in enumerate(dataset["train"]):
        if tokenizer.eos_token in example["input_ids"]:
            break
    print(f"\n{batch_idx}nd batch:")
    decoded_text = tokenizer.decode(example["input_ids"])
    print(decoded_text)
