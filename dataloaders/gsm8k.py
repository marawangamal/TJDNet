# gsm8k.py

from typing import Union
from datasets import load_dataset
from git import Optional
import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from dataloaders._base import BaseChatTemplate, group_texts


class ChatTemplateGSM8k(BaseChatTemplate):
    TEMPLATE = """[QUESTION]\n{question}\n[ANSWER]{answer}"""

    TEMPLATE_FEW_SHOT = """
        You are a mathematical reasoning assistant that solves problems step by step.

        FORMAT INSTRUCTIONS:
        1. Show all your work with clear explanations
        2. For each calculation, use the format: <<calculation=result>>result
        3. End every answer with: #### [numerical_answer_only]{eos_token}

        EXAMPLE:
        [QUESTION] Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
        [ANSWER]  Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. #### 72 {eos_token}

        Now solve the following problem using the exact format shown above:
        [QUESTION] {question}
        [ANSWER] {answer}
    """

    @classmethod
    def get_sample_prompt(
        cls, is_few_shot: bool = False, eos_token: str = "<|eot_id|>"
    ):
        tmp = cls.TEMPLATE if not is_few_shot else cls.TEMPLATE_FEW_SHOT
        return tmp.format(
            question="Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
            answer="",
            eos_token=eos_token,
        )

    @classmethod
    def safe_parse(cls, generation: str, eos_token: str):
        try:
            return (
                float(
                    generation.split("####")[-1]
                    .split(eos_token)[0]
                    .strip()
                    .split(" ")[0]
                    .split("\n")[0]
                )
                if "####" in generation
                else None
            )
        except Exception as e:
            return None

    @classmethod
    def parse_answer(cls, generation: str, eos_token: str):
        try:
            return (
                float(
                    generation.split("####")[-1]
                    .split(eos_token)[0]
                    .strip()
                    .split(" ")[0]
                    .split("\n")[0]
                )
                if "####" in generation
                else float("nan")
            )
        except Exception as e:
            return float("nan")


def prepare_example(example, eos_token="<|endoftext|>", use_few_shot=False):
    """Process a single example into training format."""
    prompt_template = (
        ChatTemplateGSM8k.TEMPLATE
        if not use_few_shot
        else ChatTemplateGSM8k.TEMPLATE_FEW_SHOT
    )
    return {
        # Used for train/eval
        "text": ChatTemplateGSM8k.TEMPLATE.format(
            question=example["question"], answer=example["answer"]
        )
        + eos_token,
        # Used for test
        "prompt": prompt_template.format(
            question=(example["question"]), answer="", eos_token=eos_token
        ),
    }


def process_train_dataset(dataset, tokenizer, input_seq_len=512):
    # Process dataset for training
    # example => dict(text, prompt) => dict(input_ids, attention_mask) => grouped

    dataset = dataset.map(
        lambda x: prepare_example(x, tokenizer.eos_token),
        remove_columns=dataset.column_names,
    )

    dataset = dataset.map(
        lambda x: tokenizer(x["text"], add_special_tokens=False),
        remove_columns=["text", "prompt"],
    )
    # TODO: maybe it can have a [SEP] token
    dataset = dataset.map(lambda x: group_texts(x, input_seq_len), batched=True)
    dataset = dataset.map(
        lambda x: {
            **x,
            "labels": x["input_ids"].copy(),  # Copy input_ids to labels
        }
    )
    return dataset


def process_test_dataset(dataset, tokenizer, use_few_shot=False):
    # Process dataset for testing (prompt_ids -- no grouping)
    # example => dict(text, prompt) => dict(input_ids, prompt_ids, attention_mask)

    dataset = dataset.map(
        lambda x: prepare_example(x, tokenizer.eos_token, use_few_shot=use_few_shot),
        remove_columns=dataset.column_names,
    )

    dataset = dataset.map(
        lambda x: {
            **tokenizer(x["prompt"], add_special_tokens=False),
            "labels": ChatTemplateGSM8k.parse_answer(x["text"], tokenizer.eos_token),
        },
        remove_columns=["text", "prompt"],
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
    input_seq_len=128,
    test_size=0.01,
    max_num_samples=68000,
    print_stats=False,
    use_few_shot=False,
    **kwargs,
):
    train_dataset = load_dataset("openai/gsm8k", "main", split="train")
    val_dataset = load_dataset("openai/gsm8k", "main", split="test")
    test_dataset = load_dataset("openai/gsm8k", "main", split="test")

    train_dataset = process_train_dataset(train_dataset, tokenizer, input_seq_len)
    val_dataset = process_train_dataset(val_dataset, tokenizer, input_seq_len)
    test_dataset = process_test_dataset(
        test_dataset, tokenizer, use_few_shot=use_few_shot
    )

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
