from typing import Optional
import random
from datasets import Dataset
from data.common import BaseChatTemplate, group_texts


class ChatTemplateSynNum(BaseChatTemplate):
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
            question="What is 14 in Binary?",
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


def generate_sample():
    """Generate a single decimal to binary conversion sample."""
    num = random.randint(0, 255)
    binary = bin(num)[2:]  # Remove '0b' prefix

    question = f"What is {num} in Binary?"
    response = f"\nLet's solve this:\n1) Divide {num} by 2 repeatedly and track remainders\n2) Reading remainders bottom-up gives: {binary}\n\n####\n{binary}"

    return {"question": question, "response": response}


def parse_qa(example, eos_token="<|endoftext|>"):
    return {
        "text": ChatTemplateSynNum.format_qa(example["question"], example["response"])
        + eos_token,
        "prompt": ChatTemplateSynNum.format_prompt(example["question"]),
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


def load_syn_num_data(
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

    dataset = load_syn_num_data(
        tokenizer=tokenizer,
        input_seq_len=512,
        num_train_samples=10000,
        num_test_samples=100,
    )

    example = next(iter(dataset["train"]))
    decoded_text = tokenizer.decode(example["input_ids"])
    print("\nSample output:")
    print(decoded_text)
