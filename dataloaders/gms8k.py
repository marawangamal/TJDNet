from networkx import d_separated
from dataloaders.base import AbstractDataset
from datasets import load_dataset, DatasetDict


class GSM8k(AbstractDataset):
    templates = {
        "0_shot": """[QUESTION]\n{question}\n[ANSWER]{answer}""",
        "few_shot": (
            "You are a helpful assistant that answers math questions. "
            "The final answer should be preceeded by ####.\n"
            "Here is an example:\n"
            "[QUESTION]\n"
            "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. "
            "Each can has 3 tennis balls. How many tennis balls does he have now?\n"
            "[ANSWER]\n"
            "Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. "
            "5 + 6 = 11. The answer is 11. #### 11\n"
            "Now, answer the following question:\n"
            "[QUESTION]\n"
            "{question}\n"
            "[ANSWER]\n"
            "{answer}"
        ),
        "few_shot:standard": (
            "You are a helpful assistant that answers math questions. "
            "The final answer should be preceeded by ####.\n"
            "Here is an example:\n"
            "[QUESTION]\n"
            "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. "
            "Each can has 3 tennis balls. How many tennis balls does he have now?\n"
            "[ANSWER]\n"
            "The answer is 11. #### 11\n"
            "Now, answer the following question:\n"
            "[QUESTION]\n"
            "{question}\n"
            "[ANSWER]\n"
            "{answer}"
        ),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def parse_answer(self, generation: str) -> float:
        try:
            return (
                float(
                    generation.split("####")[-1]
                    .split(self.eos)[0]
                    .strip()
                    .split(" ")[0]
                    .split("\n")[0]
                )
                if "####" in generation
                else float("nan")
            )
        except Exception as e:
            return float("nan")

    def get_sample_prompt(self) -> str:
        return self.templates[self.template_mode].format(
            question="Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
            answer="",
        )

    def format_train_example(self, example):
        return (
            self.templates[self.template_mode].format(
                question=example["question"],
                answer=example["answer"],
            )
            + self.eos
        )

    def format_test_example(self, example):
        return self.templates[self.template_mode].format(
            question=example["question"],
            answer="",
        )

    def format_test_label(self, example):
        return self.parse_answer(example["answer"])

    def load_data(self):

        base_datasets = {
            "train": load_dataset("openai/gsm8k", "main", split="train"),
            "eval": load_dataset("openai/gsm8k", "main", split="test"),
        }
        test_datasets = {
            "test": load_dataset("openai/gsm8k", "main", split="test"),
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

        ds = DatasetDict({**base_datasets, **test_datasets})

        if self.max_num_samples is not None:
            for split in ds:
                if len(ds[split]) > self.max_num_samples:
                    ds[split] = (
                        ds[split].shuffle(seed=42).select(range(self.max_num_samples))
                    )

        return ds


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    gsm8k = GSM8k(tokenizer, max_num_samples=100).load_data()
    print(gsm8k)
