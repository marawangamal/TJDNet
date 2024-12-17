# TJDNet: Speeding up Language Model Inference via Tensorized Joint Distribution Networks

Speeding up language model inference via tensorized joint distributions. This codebase implements TJDNet for GPT and LLAMA models but can be easily extended to other models.


## Installation

```bash
pip install -r requirements.txt
pip install -e . # to install the tjd package in editable mode
```

## Overview

TJD works by:
1. Taking a base language model
2. Adding a tractable joint distribution head
3. Training the head and (optionally) last layer while keeping other parameters frozen

## Quick Start

Here's a minimal example using TJDNet with GPT2:

```python
from models.tjdgpt2 import TJDGPT2

model = TJDGPT2(
    model_head="mps",    # Type of TJDNet head
    rank=2,              # Rank of the joint distribution
    horizon=8,           # Horizon for joint prediction
)
```

## Adding Custom Model

To add TJD support for a custom model, create a new class that inherits from the base `TJD` class. Here's how:

1. Create a new file `models/tjd_your_model.py`:

```python
from models._tjd import TJD

class TJDYourModel(TJD):
    def __init__(
        self,
        # Base Model Parameters
        vocab_size: int = <YOUR_VOCAB_SIZE>,
        n_embd: int = <YOUR_HIDDEN_SIZE>,
        model_config: Dict = <YOUR_MODEL_CONFIG>,
        # TJD Specific Parameters
        model_head: str = "base",
        rank: int = 2,
        horizon: int = 8,
        positivity_func: str = "exp",
    ):
        super().__init__(
            n_embd=n_embd,
            vocab_size=vocab_size,
            rank=rank,
            horizon=horizon,
            model_head=model_head,
            positivity_func=positivity_func,
            model_kwargs={model_config}
        )

    def get_base_model(self, **model_kwargs):
        """Initialize your base model."""
        pass

    def get_last_hidden_state(self, input_ids, attention_mask=None):
        """Get last hidden state from your model."""
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )