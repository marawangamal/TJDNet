import pandas as pd
import matplotlib.pyplot as plt

# ---- Replace these rows with your real experimental numbers ----
data = [
    {"task": "simple-math", "empirical_rank": 5, "cp_rank": 4, "drop_pct": 0.8},
    {"task": "simple-math", "empirical_rank": 5, "cp_rank": 8, "drop_pct": 0.5},
    {"task": "gsm8k", "empirical_rank": 42, "cp_rank": 4, "drop_pct": 4.1},
    {"task": "gsm8k", "empirical_rank": 42, "cp_rank": 8, "drop_pct": 2.7},
    {"task": "CommonSenseQA", "empirical_rank": 29, "cp_rank": 4, "drop_pct": 2.3},
    {"task": "CommonSenseQA", "empirical_rank": 29, "cp_rank": 8, "drop_pct": 1.4},
]
# ----------------------------------------------------------------

df = pd.DataFrame(data)

import matplotlib.pyplot as plt

# df must have: task, empirical_rank, cp_rank, drop_pct
palette = {4: "tab:blue", 8: "tab:orange"}  # add more if needed
markers = {"simple-math": "o", "gsm8k": "^", "CommonSenseQA": "s"}

fig, ax = plt.subplots(figsize=(6, 4))
for _, row in df.iterrows():
    ax.scatter(
        row["empirical_rank"],
        row["drop_pct"],
        color=palette[row["cp_rank"]],
        marker=markers[row["task"]],
        s=90,
        edgecolor="black",
        linewidth=0.5,
    )

ax.set_xlabel("Empirical tensor rank (H-token)")
ax.set_ylabel("Performance drop (%)")
ax.set_title("Drop vs. Empirical Rank across Tasks")
# Build custom legend
from matplotlib.lines import Line2D

legend_elems = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="simple-math",
        markerfacecolor="grey",
        markeredgecolor="black",
    ),
    Line2D(
        [0],
        [0],
        marker="^",
        color="w",
        label="gsm8k",
        markerfacecolor="grey",
        markeredgecolor="black",
    ),
    Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        label="CommonSenseQA",
        markerfacecolor="grey",
        markeredgecolor="black",
    ),
    Line2D([0], [0], marker="o", color="tab:blue", label="CP-rank 4", markersize=10),
    Line2D([0], [0], marker="o", color="tab:orange", label="CP-rank 8", markersize=10),
]
ax.legend(handles=legend_elems, frameon=False, ncol=2)
ax.grid(True, linestyle=":", linewidth=0.6)
plt.tight_layout()
plt.show()
