---
language:
- en
pretty_name: TJDNet

configs:
# GPT2
- config_name: gpt2_gsm8k_h2
  data_files:
  - split: train
    path: "gpt2/h2/gsm8k/train.jsonl"
  - split: test
    path: "gpt2/h2/gsm8k/test.jsonl"
- config_name: gpt2_poem_h2
  data_files:
  - split: train
    path: "gpt2/h2/poem/train.jsonl"
  - split: test
    path: "gpt2/h2/poem/test.jsonl"
- config_name: gpt2_newline_h2
  data_files:
  - split: train
    path: "gpt2/h2/newline/train.jsonl"
  - split: test
    path: "gpt2/h2/newline/test.jsonl"
- config_name: gpt2_space_h2
  data_files:
  - split: train
    path: "gpt2/h2/space/train.jsonl"
  - split: test
    path: "gpt2/h2/space/test.jsonl"
# Llama
- config_name: meta_llama_llama_2_7b_chat_hf_gsm8k_h2
  data_files:
  - split: train
    path: "meta_llama_llama_2_7b_chat_hf/h2/gsm8k/train.jsonl"
  - split: test
    path: "meta_llama_llama_2_7b_chat_hf/h2/gsm8k/test.jsonl"
- config_name: meta_llama_llama_2_7b_chat_hf_poem_h2
  data_files:
  - split: train
    path: "meta_llama_llama_2_7b_chat_hf/h2/poem/train.jsonl"
  - split: test
    path: "meta_llama_llama_2_7b_chat_hf/h2/poem/test.jsonl"
- config_name: meta_llama_llama_2_7b_chat_hf_newline_h2
  data_files:
  - split: train
    path: "meta_llama_llama_2_7b_chat_hf/h2/newline/train.jsonl"
  - split: test
    path: "meta_llama_llama_2_7b_chat_hf/h2/newline/test.jsonl"
- config_name: meta_llama_llama_2_7b_chat_hf_space_h2
  data_files:
  - split: train
    path: "meta_llama_llama_2_7b_chat_hf/h2/space/train.jsonl"
  - split: test
    path: "meta_llama_llama_2_7b_chat_hf/h2/space/test.jsonl"
---