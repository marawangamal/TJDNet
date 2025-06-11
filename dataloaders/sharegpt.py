from dataloaders.base import AbstractDataset
from datasets import load_dataset, DatasetDict

# TODO: is attention mask correct? Or does it include prev question
# Question: ....? Answer Choices: (A) ... Answer: C<|endoftext|>Question: Joe thought that the reflected sunshine made something look beautiful.  What might Joe be looking at?
# E.g., Second question should not attend to the first question


class ShareGPT(AbstractDataset):
    # Zero-shot template
    template = """[QUESTION]\n{question}\n[ANSWER]{answer}"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def parse_answer(self, generation: str):
        # Take first char after "Answer: "
        try:
            return generation.split("Answer: ")[-1].strip()[0]
        except Exception as e:
            return ""

    @classmethod
    def get_sample_prompt(cls) -> str:
        return cls.template.format(
            question="Write a short story about a cat who learns to play the piano.",
            answer="",
        )

    def format_train_example(self, example):
        output_str = ""
        for conv in example["conversations"]:
            if conv["from"] == "human":
                substr = f"[QUESTION]\n{conv['value']}"
            elif conv["from"] == "gpt":
                substr = f"\n[ANSWER]{conv['value']}"
            output_str += substr
        return output_str + self.eos

    def format_test_example(self, example):
        raise NotImplementedError("ShareGPT dataset does not have a separate test set.")

    def format_test_label(self, example):
        raise NotImplementedError("ShareGPT dataset does not have a separate test set.")

    def load_data(self):
        ds = load_dataset(
            "Aeala/ShareGPT_Vicuna_unfiltered",
            split=(
                f"train[:{self.max_num_samples}]" if self.max_num_samples else "train"
            ),
            cache_dir=self.cache_dir,
        )
        # Split the dataset into train and eval
        ds_dict = ds.train_test_split(test_size=0.1, seed=42, shuffle=True)
        ds_dict["eval"] = ds_dict["test"]  # Use eval split as test split

        for split in ds_dict:
            ds_dict[split] = self._process_train_dataset(ds_dict[split], self.tokenizer)

        ds_dict = DatasetDict(ds_dict)
        return ds_dict


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    sgpt = ShareGPT(tokenizer, max_num_samples=100).load_data()
    print(sgpt)

    # Print a sample from the dataset
    print("Train example:")
    print(tokenizer.decode(sgpt["train"][0]["input_ids"]))

    print("Eval example:")
    print(tokenizer.decode(sgpt["eval"][0]["input_ids"]))
