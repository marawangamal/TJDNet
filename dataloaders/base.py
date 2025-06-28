from abc import ABC, abstractmethod
from typing import Literal, Optional, Union
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from datasets import DatasetDict


class AbstractDataset(ABC):
    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        seq_len: int = 512,
        max_num_samples: Optional[int] = None,
        max_test_samples: Optional[int] = None,
        max_tokens: Optional[int] = None,
        template_mode: Literal["0_shot", "few_shot", "few_shot:standard"] = "0_shot",
        cache_dir: Optional[str] = None,
        domain_shift: Literal["in", "mild", "hard"] = "in",
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.eos: str = tokenizer.eos_token  # type: ignore
        self.max_num_samples = max_num_samples
        self.max_test_samples = max_test_samples
        self.max_tokens = max_tokens
        self.template_mode = template_mode
        self.cache_dir = cache_dir
        self.domain_shift = domain_shift

    @classmethod
    @abstractmethod
    def get_sample_prompt(cls) -> str:
        """Get a sample prompt for the dataset"""
        pass

    @abstractmethod
    def parse_answer(self, generation: str) -> float:
        """Parse the answer from the generated text"""
        pass

    @abstractmethod
    def load_raw_data(self) -> DatasetDict:
        """Load and process the dataset, but do NOT apply any limiting."""
        pass

    def load_data(self) -> DatasetDict:
        ds = self.load_raw_data()
        return self.apply_limits(ds)

    def apply_limits(self, ds: DatasetDict) -> DatasetDict:
        for split in ds:
            if self.max_tokens is not None:
                ds[split] = self._limit_by_tokens(ds[split], split)
            elif self.max_num_samples is not None:
                if len(ds[split]) > self.max_num_samples:
                    ds[split] = (
                        ds[split].shuffle(seed=42).select(range(self.max_num_samples))
                    )
        return ds

    @abstractmethod
    def format_train_example(self, example) -> str:
        """Format the training example"""
        pass

    @abstractmethod
    def format_test_example(self, example) -> str:
        """Format the test example"""
        pass

    # TODO: change return type to be only str
    @abstractmethod
    def format_test_label(self, example) -> float:
        """Format the test label"""
        pass

    @staticmethod
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

    def _process_train_dataset(self, dataset, tokenizer):
        # Format text using template
        dataset = dataset.map(
            lambda x: {"text": self.format_train_example(x)},
            remove_columns=dataset.column_names,
        )

        # Tokenize
        dataset = dataset.map(
            lambda x: tokenizer(x["text"], add_special_tokens=False),
            remove_columns=["text"],
        )

        # Group instead of padding/truncation
        dataset = dataset.map(lambda x: self.group_texts(x, self.seq_len), batched=True)
        return dataset

    def _process_test_dataset(self, dataset, tokenizer):
        # Format text using template
        dataset = dataset.map(
            lambda x: {
                "text": self.format_test_example(x),
                "labels": self.format_test_label(x),
            },
            remove_columns=dataset.column_names,
        )

        # Tokenize + add float labels
        dataset = dataset.map(
            lambda x: {
                **tokenizer(x["text"], add_special_tokens=False),
                "labels": x["labels"],  # preserve float labels
            },
            remove_columns=["text"],
        )
        return dataset

    # # TODO: change format_test_label to return output string which will then be tokenized
    # def _process_test_dataset_v2(self, dataset, tokenizer):
    #     # Format text using template
    #     dataset = dataset.map(
    #         lambda x: {
    #             "text": self.format_test_example(x),
    #             "labels": self.format_test_label(x),
    #         },
    #         remove_columns=dataset.column_names,
    #     )

    #     # Tokenize + add float labels
    #     dataset = dataset.map(
    #         lambda x: {
    #             **tokenizer(x["text"], add_special_tokens=False),
    #             "labels": tokenizer(
    #                 x["labels"], add_special_tokens=False
    #             ).input_ids,  # tokenize labels
    #         },
    #         remove_columns=["text"],  # remove original text and labels
    #     )
    #     return dataset
    def _limit_by_tokens(self, dataset, split_name):
        """Limit dataset by total token count - memory efficient version"""
        if self.max_tokens is None:
            return dataset

        total_tokens = 0
        example_count = 0

        # Count examples without storing them in memory
        for example in dataset:
            example_tokens = len(example["input_ids"])
            if total_tokens + example_tokens <= self.max_tokens:
                total_tokens += example_tokens
                example_count += 1
            else:
                break

        print(
            f"{split_name}: Limited to {total_tokens} tokens ({example_count} examples)"
        )
        return dataset.select(range(example_count))
