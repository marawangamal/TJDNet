# TJDNet: Speeding up Language Model Inference via Tensorized Joint Distribution Networks

Speeding up language model inference via tensorized joint distributions. This codebase implements TJDNet for GPT and LLAMA models but can be easily extended to other models.

## Requirements
- Python 3.9
- PyTorch
- transformers
- human-eval

## Installation

To install all requirements, run the following commands:
```bash
pip install -r requirements.txt
pip install -e .  # Install TJD package
pip install -e eval/human-eval # Install HumanEval
```

## Training
To train the MPS model in the paper, run this command (best checkpoint will be saved under `checkpoints`)
```bash 
python train.py --model_head mps --rank 2 --horizon 2
```

## Evaluation
To evaluate on HumanEval, run the following commands

1. Generate completetions (will be saved to samples.jsonl)
    ```
    python eval/generate_completions.py --ckpt checkpoints/<checkpoint directory name>
    ```
2. Evaluate completetions
    ```
    python eval/human-eval/human_eval/evaluate_functional_correctness.py samples.jsonl
    ```


## View datasets
To view a sample from the shakespeare dataset run
```bash
python data/shakespeare.py
```

## Visualization
1. Generate completetions (will be saved to samples.jsonl)
    ```
    python eval/generate_completions.py --dev --ckpt checkpoints/<checkpoint directory name>
    ```

2. Visualize a code completion sample
    ```
    python eval/visualize.py samples.jsonl
    ```


## Using custom models

To add a custom model, create a new class that inherits from the base `TJD` class. Here's how:


```python
from models._tjd import TJD

class SimpleModel(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.linear = nn.Linear(n_embd, n_embd)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        x = self.embedding(input_ids)
        return self.linear(x)

class TJDSimpleModel(TJD):
    def __init__(
        self,
        # Base Model Parameters
        vocab_size: int,
        n_embd: int,
        model_config: Dict,
        # TJD Specific Parameters
        model_head: str = "mps",
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
        return MySimpleModel(**model_kwargs)

    def get_last_hidden_state(self, input_ids, attention_mask=None):
        """Get last hidden state from your model."""
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

def main():
    model = TJDCustomModel(
        vocab_size=128, 
        n_embd=32, 
        model_config={
            vocab_size: 128, 
            n_embd: 32, 
        }
    )
    batch_size, seq_length = 4, 16
    input_ids = torch.randint(0, 10000, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length))
    outputs = model(input_ids, attention_mask)
    print(f"Output shape: {outputs.shape}")

if __name__ == "__main__":
    main()
```