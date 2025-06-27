from dataloaders.base import AbstractDataset
from datasets import load_dataset, DatasetDict


class AQUA(AbstractDataset):
    templates = {
        "0_shot": """[QUESTION]\n{question}\n[CHOICES]{choices}\n[ANSWER]{answer}""",
        "few_shot": (
            "You are a helpful assistant that answers math questions. "
            "The final answer should be preceeded by ####.\n"
            "Here is an example:\n"
            "[QUESTION]\n"
            "Two friends plan to walk along a 43-km trail, starting at opposite ends of the trail at the same time."
            r"If Friend P's rate is 15% faster than Friend Q's, how many kilometers will Friend P have walked when they pass each other?"
            "\n[CHOICES]\n"
            "A) 21, B) 21.5, C) 22, D) 22.5, E) 23\n"
            "[ANSWER]\n"
            "If Q complete x kilometers, then P completes 1.15x kilometers. \n"
            "x + 1.15x = 43 \n"
            "2.15x=43 \n"
            "x = 43/2.15 = 20\n"
            "Then P will have have walked 1.15*20=23 km.\n"
            "The answer is E. #### E\n"
            "Now, answer the following question:\n"
            "[QUESTION]\n"
            "{question}\n"
            "[CHOICES]\n"
            "{choices}\n"
            "[ANSWER]\n"
            "{answer}"
        ),
        "few_shot:standard": (
            "You are a helpful assistant that answers math questions. "
            "The final answer should be preceeded by ####.\n"
            "Here is an example:\n"
            "[QUESTION]\n"
            "Two friends plan to walk along a 43-km trail, starting at opposite ends of the trail at the same time."
            r"If Friend P's rate is 15% faster than Friend Q's, how many kilometers will Friend P have walked when they pass each other?"
            "\n[CHOICES]\n"
            "A) 21, B) 21.5, C) 22, D) 22.5, E) 23\n"
            "[ANSWER]\n"
            "The answer is E. #### E\n"
            "Now, answer the following question:\n"
            "[QUESTION]\n"
            "{question}\n"
            "[CHOICES]\n"
            "{choices}\n"
            "[ANSWER]\n"
            "{answer}"
        ),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.letter2num = {
            "A": 1,
            "B": 2,
            "C": 3,
            "D": 4,
            "E": 5,
            "F": 6,
            "G": 7,
        }

    def parse_answer(self, generation: str):
        try:
            answer = ""
            if "####" in generation:
                answer = (
                    generation.split("####")[-1]
                    .split(self.eos)[0]
                    .strip()
                    .split(" ")[0]
                    .split("\n")[0]
                )
            return (
                self.letter2num[answer] if answer in self.letter2num else float("nan")
            )

        except Exception as e:
            return float("nan")

    def get_sample_prompt(self) -> str:
        return self.templates[self.template_mode].format(
            question="Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
            answer="",
        )

    def format_train_example(self, example):
        answer_choices_str = ", ".join(example["options"])
        return self.templates[self.template_mode].format(
            question=example["question"],
            choices=answer_choices_str,
            answer=f"{example['rationale']} #### {example['correct']} {self.eos}",
        )

    def format_test_example(self, example):
        answer_choices_str = ", ".join(example["options"])
        return self.templates[self.template_mode].format(
            question=example["question"],
            choices=answer_choices_str,
            answer="",
        )

    def format_test_label(self, example):
        return self.letter2num[example["correct"]]

    def load_raw_data(self):
        base_datasets = {
            "train": load_dataset("deepmind/aqua_rat", "raw", split="train"),
            "eval": load_dataset("deepmind/aqua_rat", "raw", split="validation"),
        }
        test_datasets = {
            "test": load_dataset("deepmind/aqua_rat", "raw", split="test"),
        }
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
    aqua = AQUA(tokenizer, max_num_samples=100).load_data()
    print(aqua)

    # Print a sample from the dataset
    print("=" * 50)
    print("Train example:")
    print(tokenizer.decode(aqua["train"][0]["input_ids"]))
    print("-" * 50)
    print("Test example (input_ids):")
    print(tokenizer.decode(aqua["test"][0]["input_ids"]))
    print("Test example (labels):")
    print(aqua["test"][0]["labels"])
    print("-" * 50)

    # # Put in a dataloader
    # from torch.utils.data import DataLoader
    # from transformers import DataCollatorForLanguageModeling

    # dl = DataLoader(
    #     aqua["train"],
    #     batch_size=2,
    #     shuffle=True,
    #     collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    # )

    # tokenizer.pad_token = tokenizer.eos_token  # type: ignore
    # next(iter(dl))
