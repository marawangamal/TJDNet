# sharegpt.py
# TODO: Don't use group texts, instead use padding and attention mask

from datasets import load_dataset, Dataset

from data.common import group_texts


def parse_conversation(example, eos_token="<|endoftext|>"):
    text = ""
    for msg in example["conversations"]:
        assert msg["from"] in ["gpt", "human"], "Invalid message sender"
        speaker = "Assistant" if msg["from"] == "gpt" else "Human"
        text += f"{speaker}: {msg['value']}\n\n"
    text += eos_token  # Add EOS token at the end
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
        max_num_samples=10,
    )

    print(f"\nDataset sizes:")
    print(f"Train: {len(dataset['train'])} sequences")
    print(f"Test: {len(dataset['test'])} sequences")
    print(f"EOS token: {tokenizer.eos_token}")

    # Sanity check get specific batch and decode
    batch_idx = 2
    print(f"\n{batch_idx}nd batch:")

    # Get the specific batch
    train_batches = list(iter(dataset["train"]))
    batch = train_batches[batch_idx]

    # Print raw decoded text
    decoded_text = tokenizer.decode(batch["input_ids"])
    print(
        "EOS token found in decoded text"
        if tokenizer.eos_token in decoded_text
        else "EOS token not found in decoded text"
    )

    print("\nRaw decoded text:")
    print(tokenizer.decode(batch["input_ids"]))
