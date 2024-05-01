# TJDNet: Speeding up Language Model Inference via Tensorized Joint Distribution Networks

Speeding up language model inference via tensorized joint distributions. This project aims to leverage tensor operations to optimize the computational efficiency of large language model inference, providing a faster and more scalable solution for real-time applications.

## Requirements

- Python 3.7 or higher
- Additional requirements are listed in `requirements.txt`.

## Installation

### Clone the Project

Start by cloning the repository to your local machine:

```bash
git clone git@github.com:marawangamal/TJDNet.git
cd TJDNet
```

### Set Up a Virtual Environment (Optional)

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install the required dependencies

```bash
pip install -r requirements.txt
pip install -e .  # Install the package in editable mode
```

## Basic Usage

```python
model = AutoModelForCausalLM.from_pretrained("gpt2")
fast_model = TJDNet(model)

# Train the model
for epoch in range(epochs):
    for batch in data_loader:
        output = fast_model(**batch)
        output.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        ...

# Inference
sample_prompt = "The meaning of life is"
fast_model.generate(sample_prompt, max_length=50)
```

## Training

```bash
python train_clm.py
```
