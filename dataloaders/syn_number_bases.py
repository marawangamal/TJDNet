from typing import Optional
import random
from datasets import Dataset
from dataloaders.common import BaseChatTemplate, group_texts
import re


class ChatTemplateSynNumBase(BaseChatTemplate):
    TEMPLATE = """[QUESTION]\n{question}\n[RESPONSE]{response}"""

    @classmethod
    def format_qa(cls, question: str, response: Optional[str] = None) -> str:
        return cls.TEMPLATE.format(question=question, response=response or "")

    @classmethod
    def format_prompt(cls, prompt: str):
        return cls.TEMPLATE.format(question=prompt, response="")

    @classmethod
    def get_sample_prompt(cls):
        return cls.TEMPLATE.format(
            question="Randomly select a base m in [2, 8, 16] and output the representation of 42 in base m.",
            response="",
        )

    @classmethod
    def safe_parse(cls, generation: str, eos_token: str):
        try:
            return (
                generation.split("####")[1].split(eos_token)[0].strip()
                if "####" in generation
                else None
            )
        except Exception:
            return None

    @classmethod
    def check_answer(cls, answer_pred_unp: str, answer_true_unp: str, eos_token: str):
        """
        Check if predicted answer matches true answer by converting both to decimal.
        Answers are in format "VALUE [BASE]", e.g. "1010 [2]" or "FF [16]"
        """
        try:

            answer_pred = cls.safe_parse(answer_pred_unp, eos_token)
            answer_true = cls.safe_parse(answer_true_unp, eos_token)
            if not (answer_pred and answer_true):
                return False
            # Pattern matches: any characters followed by space + [number] at the end
            pattern = r"(\S+)\s*\[(\d+)\]"

            # Extract value and base for prediction
            pred_match = re.search(pattern, answer_pred.strip())
            if not pred_match:
                return False
            pred_value, pred_base = pred_match.groups()
            pred_base = int(pred_base)

            # Extract value and base for true answer
            true_match = re.search(pattern, answer_true.strip())
            if not true_match:
                return False
            true_value, true_base = true_match.groups()
            true_base = int(true_base)

            # Convert both to decimal
            pred_decimal = int(pred_value, pred_base)
            true_decimal = int(true_value, true_base)

            # Compare decimal values
            return pred_decimal == true_decimal

        except (ValueError, AttributeError):
            return False


def generate_sample():
    """Generate a number conversion sample for binary, octal, or hexadecimal."""
    # Random number between 0 and 255
    num = random.randint(0, 255)
    base = random.choice([2, 8, 16])

    question = f"Randomly select a base m in [2, 8, 16] and output the representation of {num} in base m."

    # Get representation based on base
    if base == 2:
        rep = bin(num)[2:]  # Remove '0b' prefix
        steps = f"1) Divide {num} by 2 repeatedly and track remainders\n2) Reading remainders bottom-up gives: {rep}"
    elif base == 8:
        rep = oct(num)[2:]  # Remove '0o' prefix
        steps = f"1) Divide {num} by 8 repeatedly and track remainders\n2) Reading remainders bottom-up gives: {rep}"
    else:  # base 16
        rep = hex(num)[2:].upper()  # Remove '0x' prefix and capitalize
        steps = f"1) Divide {num} by 16 repeatedly and track remainders\n2) For remainders 10-15, use letters A-F\n3) Reading remainders bottom-up gives: {rep}"

    response = f"\nLet's solve this:\n{steps}\n\n #### {rep} [{base}]"

    return {
        "question": question,
        "response": response,
    }


def parse_qa(example, eos_token="<|endoftext|>"):
    return {
        "text": ChatTemplateSynNumBase.format_qa(
            example["question"], example["response"]
        )
        + eos_token,
        "prompt": ChatTemplateSynNumBase.format_prompt(example["question"]),
    }


def process_synthetic_dataset(dataset, tokenizer, input_seq_len=512):
    dataset = dataset.map(
        lambda x: parse_qa(x, tokenizer.eos_token),
        remove_columns=dataset.column_names,
    )
    dataset = dataset.map(
        lambda x: tokenizer(x["text"], add_special_tokens=False),
        remove_columns=["text", "prompt"],
    )
    dataset = dataset.map(
        lambda x: group_texts(x, input_seq_len),
        batched=True,
    )
    return dataset


def process_synthetic_test_dataset(dataset, tokenizer):
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


class DataIterator:
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        for _ in range(self.num_samples):
            yield generate_sample()


def load_syn_num_base_data(
    tokenizer,
    input_seq_len,
    num_train_samples=50000,
    num_test_samples=100,
    **kwargs,
):
    train_dataset = Dataset.from_generator(lambda: DataIterator(num_train_samples))
    eval_dataset = Dataset.from_generator(lambda: DataIterator(num_test_samples))
    test_dataset = eval_dataset.select(range(len(eval_dataset)))  # Make copy

    train_dataset = process_synthetic_dataset(train_dataset, tokenizer, input_seq_len)
    eval_dataset = process_synthetic_dataset(eval_dataset, tokenizer, input_seq_len)
    test_dataset = process_synthetic_test_dataset(test_dataset, tokenizer)

    print(f"\nDataset sizes:")
    print(f"Train: {len(train_dataset)} sequences")
    print(f"Test: {len(eval_dataset)} sequences")
    print(f"Test: {len(test_dataset)} sequences")

    return {
        "train": train_dataset,
        "eval": eval_dataset,
        "test": test_dataset,
    }


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    dataset = load_syn_num_base_data(
        tokenizer=tokenizer,
        input_seq_len=512,
        num_train_samples=10000,
        num_test_samples=100,
    )

    example = next(iter(dataset["train"]))
    decoded_text = tokenizer.decode(example["input_ids"])
    print("\nSample output:")
    print(decoded_text)
