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

# Define markers and colors for each dataset
markers = ["o", "s", "^", "D"]
colors = ["steelblue", "coral", "green", "purple"]

# Plot each dataset separately with different markers
for i, (_, row) in enumerate(plot_df.iterrows()):
    plt.scatter(
        row["Emprical Rank"],
        row["Delta_PPL_percent"],
        s=100,
        alpha=0.8,
        color=colors[i],
        marker=markers[i],
        label=row["Dataset"],
        edgecolor="black",
        linewidth=1,
    )

plt.xlabel("Empirical Rank")
plt.ylabel("Δ PPL (%)")
plt.title("Δ PPL (%) vs Empirical Rank")
plt.legend()

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
plt.show()
