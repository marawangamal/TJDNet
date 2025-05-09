import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


def plot_memory_by_params(csv_file, output_path="results/plots/mhcp_profile.png"):
    """Generate three plots showing memory usage vs. rank, horizon, and hidden dimension."""
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Clean column names (they have spaces and |)
    df.columns = [col.strip() for col in df.columns]

    # Drop empty first and last columns if they exist
    if "" in df.columns:
        df = df.drop("", axis=1)

    # Extract parameters from model name
    def extract_param(model_str, pattern):
        match = re.search(pattern, model_str)
        return int(match.group(1)) if match else None

    # Extract rank, horizon, and hidden dimension
    df["Rank"] = df["Model"].apply(lambda x: extract_param(x, r"rank(\d+)"))
    df["Horizon"] = df["Model"].apply(lambda x: extract_param(x, r"horizon(\d+)"))
    df["HiddenDim"] = df["Model"].apply(lambda x: extract_param(x, r"hd(\d+)"))

    # Clean memory column - extract value before ±
    memory_col = (
        "GPU Memory (allocated)[MB]"
        if "GPU Memory (allocated)[MB]" in df.columns
        else "GPU Memory (allocated)[MB]  "
    )
    df["Memory"] = df[memory_col].apply(lambda x: float(x.split("±")[0].strip()))

    # Create the plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Calculate min and max memory values for consistent y-axis
    min_memory = df["Memory"].min() * 0.98  # Add a small margin
    max_memory = df["Memory"].max() * 1.02  # Add a small margin

    # Plot 1: Memory vs Rank
    rank_data = df.groupby("Rank")["Memory"].mean().reset_index()
    axes[0].plot(
        rank_data["Rank"], rank_data["Memory"], marker="o", linewidth=2, color="#8884d8"
    )
    axes[0].set_title("Memory vs. Rank")
    axes[0].set_xlabel("Rank")
    axes[0].set_ylabel("GPU Memory (MB)")
    axes[0].set_ylim(min_memory, max_memory)
    axes[0].grid(True, linestyle="--", alpha=0.7)

    # Plot 2: Memory vs Horizon
    horizon_data = df.groupby("Horizon")["Memory"].mean().reset_index()
    axes[1].plot(
        horizon_data["Horizon"],
        horizon_data["Memory"],
        marker="o",
        linewidth=2,
        color="#82ca9d",
    )
    axes[1].set_title("Memory vs. Horizon")
    axes[1].set_xlabel("Horizon")
    axes[1].set_ylabel("GPU Memory (MB)")
    axes[1].set_ylim(min_memory, max_memory)
    axes[1].grid(True, linestyle="--", alpha=0.7)

    # Plot 3: Memory vs Hidden Dimension
    hd_data = df.groupby("HiddenDim")["Memory"].mean().reset_index()
    axes[2].plot(
        hd_data["HiddenDim"],
        hd_data["Memory"],
        marker="o",
        linewidth=2,
        color="#ffc658",
    )
    axes[2].set_title("Memory vs. Hidden Dimension")
    axes[2].set_xlabel("Hidden Dimension")
    axes[2].set_ylabel("GPU Memory (MB)")
    axes[2].set_ylim(min_memory, max_memory)
    axes[2].grid(True, linestyle="--", alpha=0.7)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plots saved to {output_path}")

    # Show figure
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate memory usage plots from a CSV file"
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        default="results/data/llama31_grid.csv",
        help="Path to the CSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/plots/mhcp_profile.png",
        help="Directory to save the output plots",
    )
    args = parser.parse_args()

    plot_memory_by_params(args.csv_file, args.output)
