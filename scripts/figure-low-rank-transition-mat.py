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

import torch
import matplotlib.pyplot as plt
import argparse

from TJDNet import TGPT2


# Function to load the model's state dict
def load_model(state_dict_path):
    model = TGPT2()  # Replace with your actual model class
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)
    return model


# Function to compute the transition matrix (replace with actual logic)
def get_transition_mat(model):
    # Return a dummy transition matrix (10x10) for demonstration purposes
    # Replace this with actual transition matrix computation based on the model
    return torch.randn(10, 10).detach().numpy()


# Main script
if __name__ == "__main__":
    # Argument parser to get the state dict path from command-line arguments
    parser = argparse.ArgumentParser(
        description="Load a model and plot its transition matrix singular values."
    )
    parser.add_argument(
        "model_path", type=str, help="Path to the model's state dict (.pth file)"
    )

    args = parser.parse_args()

    # Load the model from the provided state dict path
    model = load_model(args.model_path)

    # Compute the transition matrix
    transition_matrix = get_transition_mat(model)

    # Compute singular values
    U, S, Vh = torch.linalg.svd(torch.tensor(transition_matrix))

    # Plot the singular values
    plt.figure()
    plt.plot(S.numpy(), marker="o", linestyle="-", color="b")
    plt.title("Singular Value Spectrum")
    plt.xlabel("Index")
    plt.ylabel("Singular Value")
    plt.grid(True)
    plt.show()
