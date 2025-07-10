import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the summary data
csv_path = os.path.join("results", "data", "meta_llama_spectrum_summary.csv")
df = pd.read_csv(csv_path)

# Datasets in desired order
categories = ["gsm8k", "aqua_rat", "reddit", "sst2", "wikitext2"]

# Set seaborn style
sns.set_theme(style="whitegrid")

# Melt the DataFrame for seaborn
melted = pd.melt(
    df,
    id_vars=["Category", "ModelType"],
    value_vars=["MatrixRankMean", "SpectralEnergyMean"],
    var_name="Metric",
    value_name="MeanValue",
)

# Add Std values for error bars
std_map = {
    "MatrixRankMean": "MatrixRankStd",
    "SpectralEnergyMean": "SpectralEnergyStd",
}


def get_std(row):
    std_val = df[
        (df["Category"] == row["Category"]) & (df["ModelType"] == row["ModelType"])
    ][std_map[row["Metric"]]]
    if isinstance(std_val, pd.Series):
        return std_val.iloc[0]
    return std_val


melted["Std"] = melted.apply(get_std, axis=1)

# Prepare subplots
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot Matrix Rank
matrix_rank_data = melted[melted["Metric"] == "MatrixRankMean"].copy()
if isinstance(matrix_rank_data, pd.Series):
    matrix_rank_data = matrix_rank_data.to_frame().T
sns.barplot(
    data=matrix_rank_data,
    x="Category",
    y="MeanValue",
    hue="ModelType",
    ax=axes[0],
    palette="Set2",
    capsize=0.15,
    errcolor="gray",
    errwidth=1.5,
    ci=None,
)
# Add error bars manually
for i, cat in enumerate(categories):
    for j, model in enumerate(["pretrained", "random"]):
        row = matrix_rank_data[
            (matrix_rank_data["Category"] == cat)
            & (matrix_rank_data["ModelType"] == model)
        ]
        if isinstance(row, pd.Series):
            row = row.to_frame().T
        if isinstance(row, pd.DataFrame) and not row.empty:
            mean_val = row.iloc[0]["MeanValue"]
            std_val = row.iloc[0]["Std"]
            axes[0].errorbar(
                x=i + (j - 0.5) * 0.2,
                y=mean_val,
                yerr=std_val,
                fmt="none",
                c="gray",
                capsize=5,
                lw=1.5,
            )
axes[0].set_ylabel("Matrix Rank")
axes[0].set_title("Meta-LLaMA-2-7B-Chat: Matrix Rank by Dataset")
axes[0].legend_.remove()  # Remove legend from the top plot
axes[0].grid(True, axis="y", alpha=0.3)

# Plot Spectral Energy
spectral_energy_data = melted[melted["Metric"] == "SpectralEnergyMean"].copy()
if isinstance(spectral_energy_data, pd.Series):
    spectral_energy_data = spectral_energy_data.to_frame().T
sns.barplot(
    data=spectral_energy_data,
    x="Category",
    y="MeanValue",
    hue="ModelType",
    ax=axes[1],
    palette="Set2",
    capsize=0.15,
    errcolor="gray",
    errwidth=1.5,
    ci=None,
)
# Add error bars manually
for i, cat in enumerate(categories):
    for j, model in enumerate(["pretrained", "random"]):
        row = spectral_energy_data[
            (spectral_energy_data["Category"] == cat)
            & (spectral_energy_data["ModelType"] == model)
        ]
        if isinstance(row, pd.Series):
            row = row.to_frame().T
        if isinstance(row, pd.DataFrame) and not row.empty:
            mean_val = row.iloc[0]["MeanValue"]
            std_val = row.iloc[0]["Std"]
            axes[1].errorbar(
                x=i + (j - 0.5) * 0.2,
                y=mean_val,
                yerr=std_val,
                fmt="none",
                c="gray",
                capsize=5,
                lw=1.5,
            )
axes[1].set_ylabel("Number of Components for 99% Explained Variance")
axes[1].set_title(
    "Meta-LLaMA-2-7B-Chat: Number of Components for 99% Explained Variance by Dataset"
)
axes[1].legend(title="Model Type", loc="upper right")  # Place legend only here
axes[1].grid(True, axis="y", alpha=0.3)
axes[1].set_xticklabels(categories)

plt.tight_layout()

# Save plot
save_path = os.path.join("results", "meta_llama_rand_vs_pretrained_comparison.png")
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()
