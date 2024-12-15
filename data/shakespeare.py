# shakespeare.py
from datasets import load_dataset

from data.common import group_texts


def load_shakespeare_data(tokenizer, input_seq_len, test_size=0.2):
    dataset = load_dataset("tiny_shakespeare", split="train")
    dataset = dataset.map(
        lambda x: tokenizer(x["text"], add_special_tokens=False),
        remove_columns=["text"],
    )
    dataset = dataset.map(lambda x: group_texts(x, input_seq_len), batched=True)
    dataset = dataset.train_test_split(test_size=test_size)  # type: ignore
    # DEBUG: print first example decoded
    # print(f"First example: \n{tokenizer.decode(dataset['train']['input_ids'][0])}")  # type: ignore
    return dataset


# Usage example:
if __name__ == "__main__":
    from transformers import AutoTokenizer

    # Example usage
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_shakespeare_data(
        tokenizer=tokenizer,
        input_seq_len=512,
    )

    print(f"\nDataset sizes:")
    print(f"Train: {len(dataset['train'])} sequences")
    print(f"Test: {len(dataset['test'])} sequences")

    print(f"\nFirst batch:")
    batch = next(iter(dataset["train"]))
    print(tokenizer.decode(batch["input_ids"][:50]))
