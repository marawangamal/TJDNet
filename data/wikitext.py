# wikitext.py
from datasets import load_dataset


def group_texts(examples, input_seq_len):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])  # type: ignore
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= input_seq_len:
        total_length = (total_length // input_seq_len) * input_seq_len
    # Split by chunks of input_seq_len.
    result = {
        k: [t[i : i + input_seq_len] for i in range(0, total_length, input_seq_len)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def load_wikitext_data(tokenizer, input_seq_len, version="wikitext-2-v1"):
    """Load and preprocess WikiText dataset.

    Args:
        tokenizer: Tokenizer to use for processing text
        input_seq_len: Maximum sequence length
        version: Which WikiText version to use. Options are:
                - wikitext-2-raw-v1
                - wikitext-103-raw-v1
                - wikitext-2-v1
                - wikitext-103-v1
    """
    # # Load dataset - WikiText already comes with train/test/validation splits
    # dataset = load_dataset("wikitext", version)

    # # Tokenize
    # tokenized_dataset = dataset.map(
    #     lambda x: tokenize(x, tokenizer),
    #     remove_columns=["text"],
    #     desc="Tokenizing dataset",
    # )

    # # Group into sequences
    # processed_dataset = tokenized_dataset.map(
    #     lambda x: group_texts(x, input_seq_len),
    #     batched=True,
    #     desc="Grouping texts",
    # )

    # # Create a train/test dict to match Shakespeare format
    # return {"train": processed_dataset["train"], "test": processed_dataset["test"]}
    dataset = load_dataset("wikitext", version)
    dataset = dataset.map(
        lambda x: tokenizer(x["text"], add_special_tokens=False),
        remove_columns=["text"],
    )
    dataset = dataset.map(lambda x: group_texts(x, input_seq_len), batched=True)
    dataset = dataset.train_test_split(test_size=test_size)  # type: ignore
    # DEBUG: print first example decoded
    # print(f"First example: \n{tokenizer.decode(dataset['train']['input_ids'][0])}")  # type: ignore
    return dataset
