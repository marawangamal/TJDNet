"""
Minimal script to load a PyTorch model from a state dict, compute the transition matrix, 
and plot the singular values of the matrix.

Requirements:
- PyTorch (torch)
- Matplotlib (for plotting)

Functionality:
1. Loads a PyTorch model from a given state dict file path.
2. Computes a transition matrix using `get_transition_mat(model)`.
3. Plots the singular value spectrum of the computed matrix.
"""

import string
import torch
import matplotlib.pyplot as plt
import argparse

from TJDNet import TGPT2
from TJDNet import CharacterTokenizer


# Function to load the model's state dict
def load_model(state_dict_path):
    state_dict = torch.load(state_dict_path)
    model_state_dict = state_dict["state_dict"]
    model_config = state_dict["model_config"]
    model = TGPT2(**model_config)
    model.load_state_dict(model_state_dict)
    return model


# Function to compute the transition matrix (replace with actual logic)
def get_transition_mat(model, tokenizer):
    # Return a dummy transition matrix (10x10) for demonstration purposes
    # Replace this with actual transition matrix computation based on the model
    x = torch.tensor(tokenizer.encode([" "])).reshape(1, 1)  # (B, T)
    dummy_label = torch.tensor(tokenizer.encode(["a"])).reshape(1, 1)  # (B, T)
    dummy_labels = torch.tensor(tokenizer.encode(["a", "b"])).reshape(1, 2)  # (B, T)
    p_mat = torch.zeros(model.vocab_size, model.vocab_size)
    with torch.no_grad():  # Disable gradients since we are not training
        py1_x = model(x, labels=dummy_label).logits.softmax(dim=-1)
        for i1, py1_x_i in enumerate(py1_x[0, 0]):
            py2_x = model(
                torch.stack([x, torch.tensor([[i1]])], dim=1)[:, :, 0],
                labels=dummy_labels,
            ).logits.softmax(dim=-1)
            p_mat[i1] = py2_x[0, 1] * py1_x_i
        return p_mat


# Main script
if __name__ == "__main__":
    # Argument parser to get the state dict path from command-line arguments
    parser = argparse.ArgumentParser(
        description="Load a model and plot its transition matrix singular values."
    )
    parser.add_argument(
        "--model_path", type=str, help="Path to the model's state dict (.pth file)"
    )

    characters = list(string.ascii_letters + string.digits + string.punctuation) + [
        "\n",
        " ",
        "\t",
    ]
    tokenizer = CharacterTokenizer(characters, 256)

    args = parser.parse_args()

    # Load the model from the provided state dict path
    model = load_model(args.model_path)

    # Compute the transition matrix
    transition_matrix = get_transition_mat(model, tokenizer)

    # Compute singular values
    U, S, Vh = torch.linalg.svd(torch.tensor(transition_matrix))

    # Plot the singular values
    plt.figure()
    plt.plot(S.numpy(), marker="o", linestyle="-", color="b")
    plt.title("Singular Value Spectrum")
    plt.xlabel("Index")
    plt.ylabel("Singular Value")
    plt.grid(True)
    # plt.show()
    plt.savefig("singular_values.png")
