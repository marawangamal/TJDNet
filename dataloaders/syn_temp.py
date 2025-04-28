from typing import Optional
import random
from datasets import Dataset
from dataloaders.common import BaseChatTemplate, group_texts


class ChatTemplateSynTemp(BaseChatTemplate):
    TEMPLATE = """[QUESTION]\n{question}\n[ANSWER]{answer}"""
    TEMPLATE_FEW_SHOT = """
        You are a temperature conversion assistant. Solve the temperature conversion problem step-by-step. End your answer with #### followed by the final numerical result.  Here is an example, follow the same output format. Just give the answer directly

        [QUESTION]
        What is 20°C in Fahrenheit?

        [ANSWER]
        Let's solve this step by step:\n1) To convert Celsius to Fahrenheit, use the formula: °F = (°C x 9/5) + 32\n2). Plugging in 20°C:\n   °F = (20 x 9/5) + 32\n   °F = 68\n\n#### 68
        
        [QUESTION]
        {question}

        [ANSWER]
        {answer}
    """

    @classmethod
    def get_sample_prompt(cls, is_few_shot: bool = False):
        tmp = cls.TEMPLATE if not is_few_shot else cls.TEMPLATE_FEW_SHOT
        return tmp.format(
            question="What is 20°C in Fahrenheit?",
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


def generate_sample():
    """Generate a single synthetic QA sample."""
    temp_c = random.randint(-20, 40)
    temp_f = (temp_c * 9 / 5) + 32

    question = f"What is {temp_c}°C in Fahrenheit?"
    response = f"\nLet's solve this step by step:\n1) To convert Celsius to Fahrenheit, use the formula: °F = (°C x 9/5) + 32\n2) Plugging in {temp_c}°C:\n   °F = ({temp_c} x 9/5) + 32\n   °F = {temp_f}\n\n####\n{temp_f}"

    return {"question": question, "response": response}


def parse_qa(example, eos_token="<|endoftext|>"):
    return {
        "text": ChatTemplateSynTemp.format_qa(example["question"], example["response"])
        + eos_token,
        "prompt": ChatTemplateSynTemp.format_prompt(example["question"]),
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


def load_syn_temp_data(
    tokenizer,
    input_seq_len,
    max_num_samples=50000,
    num_test_samples=100,
    print_stats=False,
    **kwargs,
):
    num_train_samples = min(max_num_samples, 10000)
    train_dataset = Dataset.from_generator(lambda: DataIterator(num_train_samples))
    eval_dataset = Dataset.from_generator(lambda: DataIterator(num_test_samples))
    test_dataset = eval_dataset.select(range(len(eval_dataset)))  # Make copy

    train_dataset = process_synthetic_dataset(train_dataset, tokenizer, input_seq_len)
    eval_dataset = process_synthetic_dataset(eval_dataset, tokenizer, input_seq_len)
    test_dataset = process_synthetic_test_dataset(test_dataset, tokenizer)

    if print_stats:
        print(f"\nDataset sizes:")
        print(f"Train: {len(train_dataset)} sequences")
        print(f"Eval: {len(eval_dataset)} sequences")
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

    dataset = load_syn_temp_data(
        tokenizer=tokenizer,
        input_seq_len=512,
        max_num_samples=10000,
        num_test_samples=100,
    )

    example = next(iter(dataset["train"]))
    decoded_text = tokenizer.decode(example["input_ids"])
    print("\nSample output:")
    print(decoded_text)
