# sharegpt.py
# TODO: Don't use group texts, instead use padding and attention mask

from datasets import load_dataset

from data.common import group_texts


def parse_conversation(example):
    text = ""
    for msg in example["conversations"]:
        assert msg["from"] in ["gpt", "human"], "Invalid message sender"
        speaker = "Assistant" if msg["from"] == "gpt" else "Human"
        text += f"{speaker}: {msg['value']}\n\n"
    return {"text": text}


def load_sharegpt_data(tokenizer, input_seq_len, test_size=0.2):
    dataset = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train")
    # Limit to first 100 examples for testing
    # dataset = dataset.select(np.arange(max_num_samples))
    dataset = dataset.map(parse_conversation, remove_columns=dataset.column_names)
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
    )

    print(f"\nDataset sizes:")
    print(f"Train: {len(dataset['train'])} sequences")
    print(f"Test: {len(dataset['test'])} sequences")

    # Sanity check get first batch and decode
    print(f"\nFirst batch:")
    batch = next(iter(dataset["train"]))
    print(tokenizer.decode(batch["input_ids"]))
