"""
Resources: 
https://huggingface.co/docs/transformers/tasks/language_modeling
https://github.com/dariush-bahrami/character-tokenizer/tree/master


Two options for data loading:

Given a dataset of sequences of different length {s1, s2, ..., s2}, we have two options for dataloading

1. Simple (preprocess_simple)
    - Convert each sequence to be of length `max_len` via padding or trunction 

2. Advanced (preprocess_function & group texts)
    - Combine to sinlge length string s = [s_1, s_2, ..., s_b], then split into chunks of size `max_len`. This is less 
    - Less wastefulness from truncation


"""

from datasets import load_dataset

from data.common import group_texts


def load_shakespeare_data(tokenizer, input_seq_len, test_size=0.2, **kwargs):
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
