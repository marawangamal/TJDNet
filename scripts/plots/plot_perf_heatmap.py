import re
import os.path as osp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main(savedir="results/plots"):

    # Read the CSV file
    df = pd.read_csv("results/data/gpt2_grid.csv")

    # Example row
    # Model   Latency [s] GPU Memory (allocated)[MB] GPU Memory,(reserved) [MB] CPU Memory,(rss) [MB]      Accuracy,Params [M],
    # gpt2::cp::rank1::horizon2::hd5120::bs::1 3.782 ± 0.004,26585.470 ± 0.000,26668.000 ± 0.000,1284.477 ± 0.012 0.000 ± 0.000,6944.241

    # 1. Parse model rank and horizon from the model column and add to the DataFrame

    def extract_rank(model_str):
        match = re.search(r"rank(\d+)", model_str)
        if match:
            return int(match.group(1))
        return None

    def extract_horizon(model_str):
        match = re.search(r"horizon(\d+)", model_str)
        if match:
            return int(match.group(1))
        return None

    def extract_latency(latency_str):
        # Remove the " ± " part and convert to float
        return float(latency_str.split(" ± ")[0])

    # Add new columns with extracted values
    df["Rank"] = df["Model"].apply(extract_rank)
    df["Horizon"] = df["Model"].apply(extract_horizon)
    df["Latency [s]"] = df["Latency [s]"].apply(extract_latency)

    # Convert the data to a pivot table format suitable for a heatmap
    # Horizon on x-axis, Rank on y-axis, and Accuracy as values
    pivot_table = df.pivot_table(index="Rank", columns="Horizon", values="Latency [s]")
    pivot_table = pivot_table.sort_index(ascending=False)  # Sort by Rank

    # Create the figure and axes
    plt.figure(figsize=(10, 8))

    # Create the heatmap
    sns.heatmap(
        pivot_table,
        # cmap="viridis",  # You can change the colormap as needed (e.g., 'YlGnBu', 'plasma', 'coolwarm')
        # annot=True,  # Show the values in each cell
        fmt=".3f",  # Format the values to 3 decimal places
        linewidths=0.5,  # Add lines between cells
        cbar_kws={"label": "Latency [s]"},  # Label for the color bar
    )  # Label for the color bar

    # Set the title and labels
    plt.title("Latency Heatmap by Horizon and Rank", fontsize=14)
    plt.xlabel("Horizon", fontsize=12)
    plt.ylabel("Rank", fontsize=12)

    # Adjust layout to make sure everything fits
    plt.tight_layout()

    # Save the figure
    path = osp.join(savedir, "hm_latency.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")

    # # Show the plot
    # plt.show()
    print(f"Saved heatmap to {path}")


if __name__ == "__main__":
    main()
