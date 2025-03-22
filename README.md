<!-- # TJDNet: Speeding up Language Model Inference via Tensorized Joint Distribution Networks

Speeding up language model inference via tensorized joint distributions. This codebase implements TJDNet for GPT and LLAMA models but can be easily extended to other models. -->

<div align="center">

<h1>TJDNet: Speeding up Language Model Inference via Tensorized Joint Distribution Networks</h1>

<i> Speeding up language model inference via tensorized joint distributions </i>


<img src="assets/image.png" style="width: 800px;" />
<!-- <i>Speeding up language model inference via tensorized joint distributions.</i> -->

<!-- <i> (Left) N forward passes to decode. (Right) TJDNet decoding uses Single forward
pass through transformer</i> -->

</div>


## Requirements

Python 3.9 (ensure this exact version or a compatible environment)

### Installation 
To install all dependencies, run the following commands:
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .  # Install TJD package
```

Optionally, you can also install human-eval
```bash
pip install -e eval/human-eval
```

## Training
To fine-tune Lllama7b using the Matrix Product State (MPS) head, run this command (best checkpoint will be saved under `checkpoints`)
```bash 
accelerate launch --use_fsdp --config_file configs/fsdp/fsdp_4gpus.yaml train.py \
  --dataset gsm8k \
  --model_type llama7b \
  --lr 1e-5 \
  --model_head mps \
  --num_layers 2 \
  --hidden_dim 768 \
  --horizon 2 \
  --horizon_eval 2 \
  --rank 2
```

## Evaluation
To compute accuracy for all checkpoints of a given experiment run:
```bash 
python sripts/eval_acc.py --checkpoint/<experiment_folder>
```

## Results

Results obtained after training LLama7b on GSM8k for 10 epochs.

| Model                            | Latency [s]   | Accuracy      |
|:---------------------------------|:--------------|:--------------|
| llama::base::bs::1               | 1.441 ± 0.007 | 0.1290 |
| llama::cp::rank4::horizon2  | 0.745 ± 0.004 | 0.0492 |
| llama::cp::rank8::horizon2  | 0.752 ± 0.002 | 0.0540 |
| llama::cp::rank16::horizon2 | 0.767 ± 0.003 | 0.0549 |
| llama::cp::rank32::horizon2 | 0.833 ± 0.028 | 0.0584 |
| llama::ucp::rank4::horizon2  | - | - |
| llama::ucp::rank8::horizon2  | - | - |
| llama::ucp::rank16::horizon2 | - | - |
| llama::ucp::rank32::horizon2 | - | - |


<!-- | Model                            | Latency [s]   | Accuracy      |
|:---------------------------------|:--------------|:--------------|
| llama::base::bs::1               | 1.441 ± 0.007 | 0.1290 |
| llama::cp::nl2::rank4::horizon2  | 0.745 ± 0.004 | 0.0492 |
| llama::cp::nl2::rank8::horizon2  | 0.752 ± 0.002 | 0.0540 |
| llama::cp::nl2::rank16::horizon2 | 0.767 ± 0.003 | 0.0549 |
| llama::cp::nl2::rank32::horizon2 | 0.833 ± 0.028 | - |
| llama::ucp::nl2::rank4::horizon2  | - | - |
| llama::ucp::nl2::rank8::horizon2  | - | - |
| llama::ucp::nl2::rank16::horizon2 | - | - |
| llama::ucp::nl2::rank32::horizon2 | - | - | -->



### Reproducing our GSM8k results
1. Run jobs specified in [here](/scripts/jobs_jr/train.yaml) (resumable by default)
2. Run jobs specified in [here](/scripts/jobs_jr/eval.yaml) to compute accuracies, when finished the results will be under `checkpoints/ckpt_dir/eval_results_b32_sNone_t128.json` (resumable by default)
3. Run [here](/scripts/eval_latency.py) to get table of latencies and manually add the corresponding accuracies

> [!NOTE]
> - You can run the jobs specified in the yaml files using our helper `scripts/jobrunner.py`. 
> - You must specify the checkpoints to compute acc for in `/scripts/jobs_jr/eval.yaml`


## Creating a custom TJDModel

To add a custom model, see examples under [here](/tjdnet/models/tjdgpt2.py). A custom TJD mdoel must inherit from the TJD class and defin `get_base_model` and `get_last_hidden_state` methods


```python
from tjdnet.models._tjd import TJD

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



<!-- 
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

## Visualization
1. Generate completetions (will be saved to samples.jsonl)
    ```
    python eval/generate_completions.py --dev --ckpt checkpoints/<checkpoint directory name>
    ```

2. Visualize a code completion sample
    ```
    python eval/visualize.py samples.jsonl
    ``` -->