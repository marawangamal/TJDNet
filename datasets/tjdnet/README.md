---
language:
- en
pretty_name: TJDNet

configs:
# GPT2
- config_name: gpt2_gsm8k
  data_files:
  - split: train
    path: "gpt2/gsm8k/train.jsonl"
  - split: test
    path: "gpt2/gsm8k/test.jsonl"
- config_name: gpt2_poem
  data_files:
  - split: train
    path: "gpt2/poem/train.jsonl"
  - split: test
    path: "gpt2/poem/test.jsonl"
- config_name: gpt2_newline
  data_files:
  - split: train
    path: "gpt2/newline/train.jsonl"
  - split: test
    path: "gpt2/newline/test.jsonl"
- config_name: gpt2_space
  data_files:
  - split: train
    path: "gpt2/space/train.jsonl"
  - split: test
    path: "gpt2/space/test.jsonl"
# Llama
- config_name: meta_llama_llama_2_7b_chat_hf_gsm8k
  data_files:
  - split: train
    path: "meta_llama_llama_2_7b_chat_hf/gsm8k/train.jsonl"
  - split: test
    path: "meta_llama_llama_2_7b_chat_hf/gsm8k/test.jsonl"
- config_name: meta_llama_llama_2_7b_chat_hf_poem
  data_files:
  - split: train
    path: "meta_llama_llama_2_7b_chat_hf/poem/train.jsonl"
  - split: test
    path: "meta_llama_llama_2_7b_chat_hf/poem/test.jsonl"
- config_name: meta_llama_llama_2_7b_chat_hf_newline
  data_files:
  - split: train
    path: "meta_llama_llama_2_7b_chat_hf/newline/train.jsonl"
  - split: test
    path: "meta_llama_llama_2_7b_chat_hf/newline/test.jsonl"
- config_name: meta_llama_llama_2_7b_chat_hf_space
  data_files:
  - split: train
    path: "meta_llama_llama_2_7b_chat_hf/space/train.jsonl"
  - split: test
    path: "meta_llama_llama_2_7b_chat_hf/space/test.jsonl"
---