#!/usr/bin/env python3
"""Minimal token counter for datasets."""

from transformers import AutoTokenizer
from dataloaders import DATASETS
from tabulate import tabulate


def main():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    summary_data = []

    for name, dataset_class in DATASETS.items():
        try:
            dataset = dataset_class(tokenizer=tokenizer, max_num_samples=10000)
            data = dataset.load_data()

            print(f"{name}:")
            for split, split_data in data.items():
                tokens = sum(len(ex["input_ids"]) for ex in split_data)
                examples = len(split_data)
                print(f"  {split}: {examples:,} examples, {tokens:,} tokens")
                summary_data.append([name, split, examples, tokens])
            print()

        except Exception as e:
            print(f"{name}: error - {e}")

    # Print summary table
    print("=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(
        tabulate(
            summary_data,
            headers=["Dataset", "Split", "Examples", "Tokens"],
            tablefmt="grid",
        )
    )


if __name__ == "__main__":
    main()


# ================================================================================
# SUMMARY TABLE
# ================================================================================
# +-------------+---------+------------+----------+
# | Dataset     | Split   |   Examples |   Tokens |
# +=============+=========+============+==========+
# | gsm8k       | train   |       2352 |  1204224 |
# +-------------+---------+------------+----------+
# | gsm8k       | eval    |        426 |   218112 |
# +-------------+---------+------------+----------+
# | gsm8k       | test    |       1319 |    88142 |
# +-------------+---------+------------+----------+
# | stemp       | train   |       1870 |   957440 |
# +-------------+---------+------------+----------+
# | stemp       | eval    |         18 |     9216 |
# +-------------+---------+------------+----------+
# | stemp       | test    |        100 |     1800 |
# +-------------+---------+------------+----------+
# | csqa        | train   |       1022 |   523264 |
# +-------------+---------+------------+----------+
# | csqa        | eval    |        127 |    65024 |
# +-------------+---------+------------+----------+
# | csqa        | test    |        570 |    30984 |
# +-------------+---------+------------+----------+
# | sharegpt    | train   |      30675 | 15705600 |
# +-------------+---------+------------+----------+
# | sharegpt    | test    |        310 |   158720 |
# +-------------+---------+------------+----------+
# | sharegpt    | eval    |        310 |   158720 |
# +-------------+---------+------------+----------+
# | aqua        | train   |      10000 |  5120000 |
# +-------------+---------+------------+----------+
# | aqua        | eval    |         78 |    39936 |
# +-------------+---------+------------+----------+
# | aqua        | test    |        254 |    21613 |
# +-------------+---------+------------+----------+
# | shakespeare | train   |        507 |   259584 |
# +-------------+---------+------------+----------+
# | shakespeare | test    |         53 |    27136 |
# +-------------+---------+------------+----------+
# | shakespeare | eval    |         53 |    27136 |
# +-------------+---------+------------+----------+
