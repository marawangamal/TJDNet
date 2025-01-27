# sharegpt.py

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


def parse_qa(example, eos_token="<|endoftext|>"):
    return {
        "text": ChatTemplateGSM8k.format_qa(example["question"], example["answer"])
        + eos_token
    }


def process_gsm8k_dataset(dataset, tokenizer, input_seq_len):
    # Process the selected samples

    dataset = dataset.map(
        lambda x: parse_qa(x, tokenizer.eos_token),
        remove_columns=dataset.column_names,
    )
    dataset = dataset.map(
        lambda x: tokenizer(x["text"], add_special_tokens=False),
        remove_columns=["text"],
    )
    # BUG: removed grouping here, otherwise the model gets trained to start a new q after an answer
    # dataset = dataset.map(lambda x: group_texts(x, input_seq_len), batched=True)
    return dataset


def load_gsm8k_data(
    tokenizer, input_seq_len, test_size=0.01, max_num_samples=68000, **kwargs
):
    train_dataset = load_dataset("openai/gsm8k", "main", split="train")
    test_dataset = load_dataset("openai/gsm8k", "main", split="test")
    train_dataset = process_gsm8k_dataset(train_dataset, tokenizer, input_seq_len)
    test_dataset = process_gsm8k_dataset(test_dataset, tokenizer, input_seq_len)
    return {
        "train": train_dataset,
        "test": test_dataset,
    }


# Usage example:
if __name__ == "__main__":
    from transformers import AutoTokenizer

    # Example usage
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    dataset = load_gsm8k_data(
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
