from dataloaders.base import AbstractDataset
from datasets import load_dataset, DatasetDict

# TODO: is attention mask correct? Or does it include prev question
# Question: ....? Answer Choices: (A) ... Answer: C<|endoftext|>Question: Joe thought that the reflected sunshine made something look beautiful.  What might Joe be looking at?
# E.g., Second question should not attend to the first question


class CSQA(AbstractDataset):
    # Zero-shot template
    template = (
        """Question: {question}\nAnswer Choices: {answer_choices}\nAnswer: {answer}"""
    )

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
            question="Where would you find magazines along side many other printed works?",
            answer_choices="A: doctor, B: bookstore, C: market, D: train station, E: mortuary",
            answer="",
        )

    def format_train_example(self, example):
        choices = example["choices"]
        answer_choices, answer_labels = choices["text"], choices["label"]
        answer_choices_str = ", ".join(
            [
                f"({answer_labels[i]}) {answer_choices[i]}"
                for i in range(len(answer_choices))
            ]
        )
        return self.template.format(
            question=example["question"],
            answer_choices=answer_choices_str,
            answer=example["answerKey"],
        )

    def format_test_example(self, example):
        choices = example["choices"]
        answer_choices, answer_labels = choices["text"], choices["label"]
        answer_choices_str = ", ".join(
            [
                f"({answer_labels[i]}) {answer_choices[i]}"
                for i in range(len(answer_choices))
            ]
        )
        return self.template.format(
            question=example["question"],
            answer_choices=answer_choices_str,
            answer="",
        )

    def format_test_label(self, example):
        return example["answerKey"]

    def load_raw_data(self):
        base_datasets = {
            "train": load_dataset("tau/commonsense_qa", split="train"),
            "eval": load_dataset("tau/commonsense_qa", split="validation"),
        }
        test_datasets = {
            "test": load_dataset("tau/commonsense_qa", split="test"),
        }
        # AdHoc limit test set to 50% of total
        test_datasets["test"] = (
            test_datasets["test"]
            .shuffle(seed=42)
            .select(range(int(len(test_datasets["test"]) * 0.5)))  # type: ignore
        )

        for split in base_datasets:
            base_datasets[split] = self._process_train_dataset(
                base_datasets[split], self.tokenizer
            )
        for split in test_datasets:
            test_datasets[split] = self._process_test_dataset(
                test_datasets[split], self.tokenizer
            )

        return DatasetDict({**base_datasets, **test_datasets})


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    csqa = CSQA(tokenizer, max_num_samples=100).load_data()
    print(csqa)

    # Print a sample from the dataset
    print("Train example:")
    print(tokenizer.decode(csqa["train"][0]["input_ids"]))

    print("Test example:")
    print(tokenizer.decode(csqa["test"][0]["input_ids"]))
