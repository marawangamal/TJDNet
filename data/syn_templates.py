from typing import Optional
import random
from datasets import Dataset
from data.common import BaseChatTemplate


class ChatTemplateSynthetic(BaseChatTemplate):
    TEMPLATE = """[CONTEXT]\n{context}\n[QUERY]\n{query}\n[RESPONSE]{response}"""

    @classmethod
    def format_qa(cls, context: str, query: str, response: Optional[str] = None) -> str:
        return cls.TEMPLATE.format(
            context=context, query=query, response=response or ""
        )

    @classmethod
    def format_prompt(cls, context: str, query: str):
        return cls.TEMPLATE.format(context=context, query=query, response="")

    @classmethod
    def get_sample_prompt(cls):
        return cls.TEMPLATE.format(
            context="Temperature in London is 20°C",
            query="What's the temperature in Fahrenheit?",
            response="",
        )

    @classmethod
    def safe_parse(cls, generation: str, eos_token: str):
        try:
            return generation.split("[RESPONSE]")[1].split(eos_token)[0].strip()
        except Exception:
            return None


def generate_sample():
    """Generate a single synthetic QA sample."""
    # Example: Temperature conversion questions
    temp_c = random.randint(-20, 40)
    temp_f = (temp_c * 9 / 5) + 32

    cities = ["London", "New York", "Tokyo", "Paris", "Sydney"]
    city = random.choice(cities)

    context = f"Temperature in {city} is {temp_c}°C"
    query = "What's the temperature in Fahrenheit?"
    response = f"\nLet's solve this step by step:\n1) To convert Celsius to Fahrenheit, use the formula: °F = (°C × 9/5) + 32\n2) Plugging in {temp_c}°C:\n   °F = ({temp_c} × 9/5) + 32\n   °F = {temp_f}\n\n####\n{temp_f}"

    return {"context": context, "query": query, "response": response}


def parse_qa(example, eos_token="<|endoftext|>"):
    return {
        "text": ChatTemplateSynthetic.format_qa(
            example["context"], example["query"], example["response"]
        )
        + eos_token
    }


def process_synthetic_dataset(dataset, tokenizer):
    dataset = dataset.map(
        lambda x: parse_qa(x, tokenizer.eos_token),
        remove_columns=dataset.column_names,
    )
    dataset = dataset.map(
        lambda x: tokenizer(x["text"], add_special_tokens=False),
        remove_columns=["text"],
    )
    return dataset


class DataIterator:
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        for _ in range(self.num_samples):
            yield generate_sample()


def load_synthetic_data(
    tokenizer,
    input_seq_len,
    num_train_samples=1000,
    num_test_samples=100,
    **kwargs,
):
    train_dataset = Dataset.from_generator(lambda: DataIterator(num_train_samples))
    test_dataset = Dataset.from_generator(lambda: DataIterator(num_test_samples))

    # Process datasets
    train_dataset = process_synthetic_dataset(train_dataset, tokenizer)
    test_dataset = process_synthetic_dataset(test_dataset, tokenizer)

    print(f"\nDataset sizes:")
    print(f"Train: {len(train_dataset)} sequences")
    print(f"Test: {len(test_dataset)} sequences")

    return {
        "train": train_dataset,
        "test": test_dataset,
    }


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    dataset = load_synthetic_data(
        tokenizer=tokenizer,
        input_seq_len=512,
        num_train_samples=100,
        num_test_samples=10,
    )

    # Print sample
    example = next(iter(dataset["train"]))
    decoded_text = tokenizer.decode(example["input_ids"])
    print("\nSample output:")
    print(decoded_text)
