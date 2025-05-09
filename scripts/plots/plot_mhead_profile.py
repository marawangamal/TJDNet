import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np


def plot_memory_by_params(csv_file, output_path="results/plots/memory_analysis.png"):
    """Generate plots showing memory usage vs. parameters using SQL-like approach."""
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Clean column names
    df.columns = [col.strip() for col in df.columns]

    # Extract parameters from model name
    def extract_param(model_str, pattern):
        match = re.search(pattern, model_str)
        return int(match.group(1)) if match else None

    # Extract rank, horizon, and hidden dimension
    df["Rank"] = df["Model"].apply(lambda x: extract_param(x, r"rank(\d+)"))
    df["Horizon"] = df["Model"].apply(lambda x: extract_param(x, r"horizon(\d+)"))
    df["HiddenDim"] = df["Model"].apply(lambda x: extract_param(x, r"hd(\d+)"))

    # Clean memory column - extract value before ±
    memory_col = "GPU Memory (allocated)[MB]"
    if memory_col not in df.columns:
        # Try to find column with GPU Memory and allocated in the name
        for col in df.columns:
            if "GPU Memory" in col and "allocated" in col:
                memory_col = col
                break

    df["Memory"] = df[memory_col].apply(lambda x: float(x.split("±")[0].strip()))

    # Calculate min and max memory values for consistent y-axis
    min_memory = df["Memory"].min() * 0.98  # Add a small margin
    max_memory = df["Memory"].max() * 1.02  # Add a small margin

    # SQL-like approach: Let's create "pivoted" views for each parameter
    # These will show how memory changes with one parameter while controlling for others

    # Create the figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Memory vs. Rank: Control for HiddenDim and Horizon
    # Group by all parameters, find combinations with varying rank
    ctrl_for_rank = df.pivot_table(
        index=["HiddenDim", "Horizon"], columns="Rank", values="Memory", aggfunc="mean"
    )

    # Only keep combinations where we have multiple rank values (can draw a line)
    ctrl_for_rank = ctrl_for_rank.dropna(thresh=2)

    # For each combination of HiddenDim and Horizon, plot a line across ranks
    for idx, row in ctrl_for_rank.iterrows():
        hd, horizon = idx
        # Plot a line with rank as x-axis and memory as y-axis
        axes[0].plot(row.index, row.values, marker="o", label=f"hd={hd}, h={horizon}")

    axes[0].set_title("Memory vs. Rank\n(Controlling for Hidden Dim & Horizon)")
    axes[0].set_xlabel("Rank")
    axes[0].set_ylabel("GPU Memory (MB)")
    axes[0].set_ylim(min_memory, max_memory)
    axes[0].grid(True, linestyle="--", alpha=0.7)

    # 2. Memory vs. Horizon: Control for Rank and HiddenDim
    ctrl_for_horizon = df.pivot_table(
        index=["Rank", "HiddenDim"], columns="Horizon", values="Memory", aggfunc="mean"
    )

    # Only keep combinations where we have multiple horizon values
    ctrl_for_horizon = ctrl_for_horizon.dropna(thresh=2)

    # For each combination of Rank and HiddenDim, plot a line across horizons
    for idx, row in ctrl_for_horizon.iterrows():
        rank, hd = idx
        # Plot a line with horizon as x-axis and memory as y-axis
        axes[1].plot(row.index, row.values, marker="o", label=f"r={rank}, hd={hd}")

    axes[1].set_title("Memory vs. Horizon\n(Controlling for Rank & Hidden Dim)")
    axes[1].set_xlabel("Horizon")
    axes[1].set_ylabel("GPU Memory (MB)")
    axes[1].set_ylim(min_memory, max_memory)
    axes[1].grid(True, linestyle="--", alpha=0.7)

    # 3. Memory vs. Hidden Dimension: Control for Rank and Horizon
    ctrl_for_hd = df.pivot_table(
        index=["Rank", "Horizon"], columns="HiddenDim", values="Memory", aggfunc="mean"
    )

    # Only keep combinations where we have multiple hidden dimension values
    ctrl_for_hd = ctrl_for_hd.dropna(thresh=2)

    # For each combination of Rank and Horizon, plot a line across hidden dimensions
    for idx, row in ctrl_for_hd.iterrows():
        rank, horizon = idx
        # Plot a line with hidden dimension as x-axis and memory as y-axis
        axes[2].plot(row.index, row.values, marker="o", label=f"r={rank}, h={horizon}")

    axes[2].set_title("Memory vs. Hidden Dimension\n(Controlling for Rank & Horizon)")
    axes[2].set_xlabel("Hidden Dimension")
    axes[2].set_ylabel("GPU Memory (MB)")
    axes[2].set_ylim(min_memory, max_memory)
    axes[2].grid(True, linestyle="--", alpha=0.7)

    # Add compact legends to each plot
    for ax in axes:
        if len(ax.get_lines()) > 0:  # Only add legend if we have lines
            ax.legend(fontsize="x-small", loc="best", framealpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plots saved to {output_path}")

    # Create a simple swarmplot to show the overall distribution
    plt.figure(figsize=(15, 5))

    # Plot 1: Memory vs Rank with swarmplot
    plt.subplot(1, 3, 1)
    sns.boxplot(x="Rank", y="Memory", data=df, width=0.5, color="lightgray")
    sns.swarmplot(x="Rank", y="Memory", data=df, size=7, palette="viridis")
    plt.title("Memory vs. Rank\n(All Data Points)")
    plt.xlabel("Rank")
    plt.ylabel("GPU Memory (MB)")
    plt.ylim(min_memory, max_memory)
    plt.grid(True, linestyle="--", alpha=0.7)

    # Plot 2: Memory vs Horizon with swarmplot
    plt.subplot(1, 3, 2)
    sns.boxplot(x="Horizon", y="Memory", data=df, width=0.5, color="lightgray")
    sns.swarmplot(x="Horizon", y="Memory", data=df, size=7, palette="plasma")
    plt.title("Memory vs. Horizon\n(All Data Points)")
    plt.xlabel("Horizon")
    plt.ylabel("GPU Memory (MB)")
    plt.ylim(min_memory, max_memory)
    plt.grid(True, linestyle="--", alpha=0.7)

    # Plot 3: Memory vs Hidden Dimension with swarmplot
    plt.subplot(1, 3, 3)
    sns.boxplot(x="HiddenDim", y="Memory", data=df, width=0.5, color="lightgray")
    sns.swarmplot(x="HiddenDim", y="Memory", data=df, size=7, palette="cividis")
    plt.title("Memory vs. Hidden Dimension\n(All Data Points)")
    plt.xlabel("Hidden Dimension")
    plt.ylabel("GPU Memory (MB)")
    plt.ylim(min_memory, max_memory)
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()

    # Save swarmplot figure
    swarmplot_path = output_path.replace(".png", "_swarmplot.png")
    plt.savefig(swarmplot_path, dpi=300, bbox_inches="tight")
    print(f"Swarmplots saved to {swarmplot_path}")


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
        default="results/plots/memory_analysis.png",
        help="Path to save the output plots",
    )
    args = parser.parse_args()

    plot_memory_by_params(args.csv_file, args.output)
