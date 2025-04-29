<!-- # TJDNet: Speeding up Language Model Inference via Tensorized Joint Distribution Networks

Speeding up language model inference via tensorized joint distributions. This codebase implements TJDNet for GPT and LLAMA models but can be easily extended to other models. -->

<div align="center">

<h1>TJDNet: Speeding up Language Model Inference via Tensorized Joint Distribution Networks</h1>


<i> Speeding up language model inference via tensorized joint distributions </i>


<img src="assets/image.png" style="width: 500;" />
<!-- <i>Speeding up language model inference via tensorized joint distributions.</i> -->

<!-- <i> (Left) N forward passes to decode. (Right) TJDNet decoding uses Single forward
pass through transformer</i> -->

</div>

## Overivew

This repository provides the implementation for TJDNet, allowing for faster inference with Large Language Models (LLMs) like GPT and LLaMA variants. The core idea is to replace the standard autoregressive sampling head with a tensorized head (e.g., MPS or CP) that predicts the joint distribution of multiple future tokens simultaneously.

While examples focus on GPT and LLaMA, the framework is designed to be extensible to other transformer architectures. Experimental results are presented [here](#Results).


## Installation 
Requires **Python 3.9+**. Using a virtual environment (like venv or conda) is highly recommended.

```bash
# 1. Clone the repository (if you haven't already)
git clone git@github.com:marawangamal/TJDNet.git
cd tjdnet

# 2. Create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Install the TJDNet package itself in editable mode
pip install -e .

# 5. (optional) login to wandb
wandb login

# 6. (optional) login to huggingface -- needed to run llama models
huggingface-cli login
```

### Sanity Check ✅

To verify that your installation and setup are correct train a small model on a toy dataset and confirm it reaches nearly 100% accuracy:

```bash
python train.py --compute_acc
```

After training for 4 epochs (~5 minutes on a single 40GB GPU), you should observe **100% accuracy** on the stemp dataset and an output like this

```txt
What is -8°C in Fahrenheit?

Let's solve this step by step:
1) To convert Celsius to Fahrenheit, use the formula: °F = (°C x 9/5) + 32
2) Plugging in -8°C:
   °F = (-8 x 9/5) + 32
   °F = 17.6

####
17.6<|endoftext|>
Eval accuracy: 1.0
```



## Training

To fine-tune Llama using the Canonical Polyadic (CP) head, run this command (best checkpoint will be saved under `checkpoints`)
```bash 
accelerate launch --use_fsdp --config_file configs/fsdp/fsdp_4gpus.yaml train.py \
    --dataset gsm8k \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --epochs 50 \
    --batch_size 8 \ 
    --seq_len 128 \ 
    --lr 1e-5 \ 
    --model_head cp \ 
    --hidden_dim 5120 \ 
    --horizon 2 \ 
    --horizon_eval 2 \ 
    --rank 16
```

## Evaluation
To run evaluation (compute accuracy) run the following command
```bash 
python scripts/eval_acc.py -c <checkpoint_path>
```

## Scripts

TJDNet provides several scripts for analysis and benchmarking:

- `scripts/eval_acc.py`: Evaluate model accuracy
- `scripts/eval_latency.py`: 
- `scripts/plots/plot_output_dist_spectrum.py`: Visualize specturm of output token distribution
- `scripts/plots/plot_lat_mem_rank_horizon.py`: Benchmark latency and memory vs. rank and horizon
- `scripts/datasets/create_hf_tjdnet_ds.py`: Generate huggingface tjdnet model generation likelihoods dataset
- `scripts/jobrunner.py`: SLURM job submission utility (Described in more detail below)

Run any script with `--help` for usage information.


### SLURM batch Job runner

Use `scripts/jobrunner.py` to submit and track multiple experiments, particularly on clusters using the SLURM workload manager.

* **Submit a single job:**
    Wrap your full training or evaluation command string in quotes.
    ```bash
    python scripts/jobrunner.py --job "<your_full_command_here>"
    ```

* **Submit batch jobs from a config file:**
    Define parameters for multiple jobs in a YAML file (see `config/train.yaml` for format).
    ```bash
    python scripts/jobrunner.py -f config/train.yaml
    ```

* **Check status of submitted jobs:**
    ```bash
    python scripts/jobrunner.py -s
    ```

*(Note: This job runner currently assumes a SLURM environment (`sbatch`, `squeue` commands).)*


### Generate Huggingface Dataset

Use `scripts/create_hf_tjdnet_ds.py` to create the hf dataset then run 
```bash
huggingface-cli upload mremila/tjdnet datasets/tjdnet --repo-type dataset
```

## Results
Results obtained after training LLama7b on GSM8k for 50 epochs are given


<!-- HASH: 2e1b9e465bb1128a076d5500112fe9da441de5b3 -->
| Model                                               | Latency [s]   | Accuracy |  
|:----------------------------------------------------|:--------------|:---------|
| llama::base::bs::1                                  | 2.884 ± 0.003 | 0.1290   |
| llama::cp::rank1::hd5120::horizon2::bs::1           |               | train*   | (to compare with fb paper)
| llama::cp::rank1::hd5120::horizon3::bs::1           |               | train*   | (to compare with fb paper)
| llama::cp::rank8::hd2048::horizon2::bs::1           |               | 0.0902   |
| llama::cp::rank8::hd4096::horizon2::bs::1           |               | 0.0842   |
| llama::cp::rank8::hd5120::horizon2::bs::1           | 1.520 ± 0.001 | 0.0925   |
| llama::cp::rank16::hd5120::horizon2::bs::1          | 1.565 ± 0.008 | 0.0857   |
| llama::cp::rank32::hd5120::horizon2::bs::1          | 1.565 ± 0.008 | 0.0750   |  
| llama::cp::rank8::hd8192::horizon2::bs::1           |               | 0.0773   |
| llama::cp::rank8::hd5192::horizon3::bs::1           |               | 0.050    | 
| llama::cp::rank8::hd5192::horizon4::bs::1           |               | 0.028    |
| llama::cp::rank8::hd5192::horizon2::bs::1::umel     |               | 0.055    |  
| llama::mps::rank2::hd5192::horizon2::bs::1          |               | eval*    | wifd66589a
| llama::mps::rank4::hd5192::horizon2::bs::1          |               | train*   |  
| llama::cpo::rank8::hd2048::horizon2::bs::1          |               | train*   |  (to compare with oslo paper)
| llama::cpo::rank8::hd2048::horizon3::bs::1          |               | eval*    |  wi51b58f77 (to compare with oslo paper)



<!-- HASH: d679fdd46af3f9becd14178810fead94ee245412 -->
| Model                                               | Latency [s]   | Accuracy |  
|:----------------------------------------------------|:--------------|:---------|
| llama-3.2-3b-instruct::base-fs::bs::1               |               | 0.6527   |  # few shot
| llama-3.2-3b-instruct::base-ft::bs::1               |               | -        |  # retrained last layer

<!-- 
accelerate launch --use_fsdp --config_file configs/fsdp/fsdp_4gpus.yaml train.py --dataset gsm8k --model_type llama7b --epochs 50 --batch_size 8 --seq_len 128 --lr 1e-5 --model_head cpo --hidden_dim 2048 --horizon 3 --horizon_eval 3 --rank 8  2025-04-27 21:56:47 -->


