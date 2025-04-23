---
language:
- en
pretty_name: TJDNet

configs:
- config_name: gpt2_gsm8k
  data_files:
  - split: train
    path: "gpt2_gsm8k/train.jsonl"
  - split: test
    path: "gpt2_gsm8k/test.jsonl"
- config_name: gpt2_poem
  data_files:
  - split: train
    path: "gpt2_poem/train.jsonl"
  - split: test
    path: "gpt2_poem/test.jsonl"
- config_name: gpt2_newline
  data_files:
  - split: train
    path: "gpt2_newline/train.jsonl"
  - split: test
    path: "gpt2_newline/test.jsonl"
- config_name: gpt2_space
  data_files:
  - split: train
    path: "gpt2_space/train.jsonl"
  - split: test
    path: "gpt2_space/test.jsonl"
---