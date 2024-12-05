import wandb
import matplotlib.pyplot as plt
import numpy as np

api = wandb.Api()
runs = api.runs("marawan-gamal/tjdnet-shakepeare-v3")

data = {"mps": {}, "cp": {}}
baseline_loss = None

for run in runs:
    if run.config.get("model") == "base":
        baseline_loss = run.summary["eval_loss"]
    elif run.config.get("model") in ["mps", "cp"]:
        model = run.config["model"]
        horizon = run.config.get("horizon", 0)
        if horizon not in data[model]:
            data[model][horizon] = {"ranks": [], "losses": []}
        data[model][horizon]["ranks"].append(run.config.get("rank", 0))
        data[model][horizon]["losses"].append(run.summary["eval_loss"])

plt.figure(figsize=(10, 6))
colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]

if baseline_loss:
    plt.axhline(y=baseline_loss, color="black", linestyle="-", label="Baseline")

for i, (horizon, horizon_data) in enumerate(data["mps"].items()):
    ranks, losses = zip(*sorted(zip(horizon_data["ranks"], horizon_data["losses"])))
    plt.plot(ranks, losses, "--", marker="o", label=f"MPS h={horizon}", color=colors[i])

    if horizon in data["cp"]:
        cp_data = data["cp"][horizon]
        ranks, losses = zip(*sorted(zip(cp_data["ranks"], cp_data["losses"])))
        plt.plot(
            ranks, losses, "-", marker="o", label=f"CP h={horizon}", color=colors[i]
        )

plt.xlabel("Rank")
plt.ylabel("Eval Loss")
plt.legend()
plt.grid(True)
plt.savefig("loss_comparison.png")
