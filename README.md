# TJDNet: Speeding up Language Model Inference via Tensorized Joint Distribution Networks

Speeding up language model inference via tensorized joint distributions. This project aims to leverage tensor operations to optimize the computational efficiency of large language model inference, providing a faster and more scalable solution for real-time applications.

## Quick Start

```python
model = AutoModelForCausalLM.from_pretrained("gpt2")
fast_model = TJDNet(model)  # <--- Just wrap your model with TJDNet

# Train the model
for epoch in range(epochs):
    for batch in data_loader:
        loss = fast_model(**batch)
        loss.backward()
        ...
```

## Installation

1. Clone the project from GitHub:

   ```bash
   git clone git@github.com:marawangamal/TJDNet.git
   cd TJDNet
   ```

2. Set Up a Virtual Environment (Optional)

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install the package in editable mode
   ```

## Training

```bash
python train.py  --model_name gpt2 --dataset_name wikitext --max_seq_len 128 --num_epochs 3
```
