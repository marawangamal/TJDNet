# TJDNet: Speeding up Language Model Inference via Tensorized Joint Distribution Networks

Speeding up language model inference via tensorized joint distributions. This project aims to leverage tensor operations to optimize the computational efficiency of large language model inference, providing a faster and more scalable solution for real-time applications.

## Installation
```bash
pip install -r requirements.txt
pip install -e .  # Install the package in editable mode
```

## Training

```bash
python train.py  --model base # baseline GPT2 model
python train.py  --model mps # GPT2 with MPS tensorized joint distribution
python train.py  --model cp # GPT2 with CP tensorized joint distribution
```


<!-- ## Sanity Checks

These should have roughly the same loss

```bash
python train.py --model base --rank 1 --horizon 1  // baseline model (no tensorization)
python train.py --model mps --rank 1 --horizon 1 // mps model
python train.py --model cp --rank 1 --horizon 1 // cp model
``` -->

<!-- ## Running Tests

To ensure everything is set up correctly, you can run the unit tests:

```bash
python -m unittest discover -s __tests__
``` -->
