# gsm8k.py

from datasets import load_dataset
from git import Optional

from data.common import BaseChatTemplate, group_texts


class ChatTemplateGSM8k(BaseChatTemplate):
    TEMPLATE = """[QUESTION]\n{question}\n[ANSWER]{answer}"""

    @classmethod
    def format_qa(cls, question: str, answer: Optional[str] = None) -> str:
        return cls.TEMPLATE.format(question=question, answer=answer or "")

    @classmethod
    def format_prompt(cls, prompt: str):
        return cls.TEMPLATE.format(question=prompt, answer="")

    @classmethod
    def get_sample_prompt(cls):
        return cls.TEMPLATE.format(
            question="Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
            answer="",
        )

    @classmethod
    def safe_parse(cls, generation: str, eos_token: str):
        try:
            return (
                float(generation.split("####")[1].split(eos_token)[0].strip())
                if "####" in generation
                else None
            )
        except Exception as e:
            return None


def parse_qa(example, eos_token="<|endoftext|>"):
    return {
        "text": ChatTemplateGSM8k.format_qa(example["question"], example["answer"])
        + eos_token,
        "prompt": ChatTemplateGSM8k.format_prompt(example["question"]),
    }


def process_gsm8k_dataset(dataset, tokenizer, input_seq_len=512):
    # Process the selected samples

    dataset = dataset.map(
        lambda x: parse_qa(x, tokenizer.eos_token),
        remove_columns=dataset.column_names,
    )

    dataset = dataset.map(
        lambda x: tokenizer(x["text"], add_special_tokens=False),
        remove_columns=["text", "prompt"],
    )
    # TODO: maybe it can have a [SEP] token
    # BUG: removed grouping here, otherwise the model gets trained to start a new q after an answer
    dataset = dataset.map(
        lambda x: group_texts(x, input_seq_len),
        batched=True,
    )
    return dataset


def process_gsm8k_test_dataset(dataset, tokenizer):
    # Process the selected samples

    dataset = dataset.map(
        lambda x: parse_qa(x, tokenizer.eos_token),
        remove_columns=dataset.column_names,
    )

    dataset = dataset.map(
        lambda x: {
            **tokenizer(x["text"], add_special_tokens=False),
            "prompt_ids": tokenizer(x["prompt"])["input_ids"],
        },
    )
    return dataset


def is_valid_seq(x, max_seq_len):
    """Check if sequence length is within the model's limit.

    Args:
        x: Dataset example containing 'input_ids'
        max_seq_len: Maximum allowed sequence length

    Returns:
        bool: True if sequence length is valid, False otherwise
    """
    seq_len = len(x["input_ids"])
    is_valid = seq_len <= max_seq_len

    if not is_valid:
        print(f"Filtered sequence of length {seq_len} > {max_seq_len}")

    return is_valid


def load_gsm8k_data(
    tokenizer,
    input_seq_len,
    test_size=0.01,
    max_num_samples=68000,
    print_stats=True,
    **kwargs,
):
    train_dataset = load_dataset("openai/gsm8k", "main", split="train")
    val_dataset = load_dataset("openai/gsm8k", "main", split="test")
    test_dataset = load_dataset("openai/gsm8k", "main", split="test")

    train_dataset = process_gsm8k_dataset(train_dataset, tokenizer, input_seq_len)
    val_dataset = process_gsm8k_dataset(val_dataset, tokenizer, input_seq_len)
    test_dataset = process_gsm8k_test_dataset(test_dataset, tokenizer)

    # Get original sizes
    orig_train_size = len(train_dataset)
    orig_test_size = len(val_dataset)

    # Limit the length of the sequences
    train_dataset_filtered = train_dataset.filter(
        lambda x: is_valid_seq(x, input_seq_len)
    )
    val_dataset_filtered = val_dataset.filter(lambda x: is_valid_seq(x, input_seq_len))

    # Print statistics
    if print_stats:
        print("\nSequence Length Filtering Statistics:")
        print("-" * 40)
        print(f"Training set:")
        print(f"  Original samples: {orig_train_size}")
        print(f"  Filtered samples: {len(train_dataset_filtered)}")
        print(f"  Removed samples: {orig_train_size - len(train_dataset_filtered)}")
        print(
            f"  Percentage kept: {(len(train_dataset_filtered)/orig_train_size)*100:.2f}%"
        )

        print(f"\nTest set:")
        print(f"  Original samples: {orig_test_size}")
        print(f"  Filtered samples: {len(val_dataset_filtered)}")
        print(f"  Removed samples: {orig_test_size - len(val_dataset_filtered)}")
        print(
            f"  Percentage kept: {(len(val_dataset_filtered)/orig_test_size)*100:.2f}%"
        )
        print("-" * 40)

    return {
        "train": train_dataset_filtered,  # Return filtered datasets instead of original
        "eval": val_dataset_filtered,
        "test": test_dataset,
    }


# Usage example:
if __name__ == "__main__":
    from transformers import AutoTokenizer

    # Example usage
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    dataset = load_gsm8k_data(
        tokenizer=tokenizer, input_seq_len=512, max_num_samples=100
    )

    print(f"\nDataset sizes:")
    print(f"Train: {len(dataset['train'])} sequences")
    print(f"Eval: {len(dataset['eval'])} sequences")
    print(f"Test: {len(dataset['test'])} sequences")
    print(f"EOS token: {tokenizer.eos_token}")

    # Find an example with EOS token
    for batch_idx, example in enumerate(dataset["train"]):
        if tokenizer.eos_token in example["input_ids"]:
            break
    print(f"\n{batch_idx}nd batch:")
    decoded_text = tokenizer.decode(example["input_ids"])
    print(decoded_text)
