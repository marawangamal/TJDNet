# sharegpt.py

from datasets import load_dataset

from dataloaders.common import BaseChatTemplate, group_texts


class ChatTemplateShareGPT(BaseChatTemplate):
    HUMAN_PREFIX = "Human: "
    ASSISTANT_PREFIX = "Assistant: "
    MESSAGE_END = "\n\n"

    @classmethod
    def format_turn(cls, is_assistant: bool, message: str) -> str:
        """Format a single conversation turn."""
        prefix = cls.ASSISTANT_PREFIX if is_assistant else cls.HUMAN_PREFIX
        return prefix + message + cls.MESSAGE_END

    @classmethod
    def format_prompt(cls, prompt: str) -> str:
        """Format a human prompt for model input."""
        return cls.HUMAN_PREFIX + prompt + cls.MESSAGE_END

    @classmethod
    def get_sample_prompt(cls):
        return (
            cls.format_prompt(
                'complete the following code from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """'
            ),
        )


# TODO: use just 2 turns
def parse_conversation(example, eos_token="<|endoftext|>"):
    text = ""
    for msg in example["conversations"]:
        assert msg["from"] in ["gpt", "human"], "Invalid message sender"
        is_assistant = msg["from"] == "gpt"
        text += ChatTemplateShareGPT.format_turn(is_assistant, msg["value"])
        if is_assistant:
            text += eos_token
    return {"text": text}


def load_sharegpt_data(
    tokenizer, input_seq_len, test_size=0.01, max_num_samples=68000, **kwargs
):
    # Load only the needed number of samples by using streaming and taking first max_num_samples

    dataset = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train")
    dataset = dataset.select(range(max_num_samples))

    # dataset = load_dataset(
    #     "Aeala/ShareGPT_Vicuna_unfiltered", split="train", streaming=True
    # )
    # # Take only the required number of samples
    # dataset = dataset.take(max_num_samples)
    # # Convert streaming dataset to regular dataset
    # dataset = Dataset.from_generator(lambda: dataset)

    # Process the selected samples
    dataset = dataset.map(
        lambda x: parse_conversation(x, tokenizer.eos_token),
        remove_columns=dataset.column_names,
    )
    dataset = dataset.map(
        lambda x: tokenizer(x["text"], add_special_tokens=False),
        remove_columns=["text"],
    )
    dataset = dataset.map(lambda x: group_texts(x, input_seq_len), batched=True)
    dataset = dataset.train_test_split(test_size=test_size)
    return dataset


# Usage example:
if __name__ == "__main__":
    from transformers import AutoTokenizer

    # Example usage
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_sharegpt_data(
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
