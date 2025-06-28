#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# File paths
DATA_PATH = "results/data/dcomplexity.csv"
OUTPUT_PATH = "results/dcomplexity_delta_vs_rank.png"

# Set seaborn style defaults
sns.set_theme(style="whitegrid")

# Read the data
df = pd.read_csv(DATA_PATH)
df["Emprical Rank"] = pd.to_numeric(df["Emprical Rank"], errors="coerce")
df["Delta_PPL_percent"] = 100 * (df["PPL (STP)"] - df["PPL (MTP)"]) / df["PPL (STP)"]

# Sort by empirical rank
plot_df = df.sort_values("Emprical Rank")

# Create the plot using seaborn
plt.figure(figsize=(8, 6))

# Use seaborn's scatterplot
sns.scatterplot(
    data=plot_df,
    x="Emprical Rank",
    y="Delta_PPL_percent",
    s=100,
    alpha=0.7,
    color="steelblue",
)

# Add dataset labels to points
for _, row in plot_df.iterrows():
    plt.annotate(
        str(row["Dataset"]),
        (float(row["Emprical Rank"]), float(row["Delta_PPL_percent"])),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=10,
        ha="left",
    )

plt.xlabel("Empirical Rank")
plt.ylabel("Δ PPL (%)")
plt.title("Δ PPL (%) vs Empirical Rank")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
plt.show()
