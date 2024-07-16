import wandb
import numpy as np
import re
import json
import matplotlib.pyplot as plt


def get_norm_const_points(run):
    # Get the history of specific metrics
    history = run.history()
    config = json.loads(run.json_config)
    output_size = config["output_size"]["value"]

    # Ensure the '_step' column is present
    if "_step" not in history.columns:
        print(f"Run {run.id} does not have '_step' column.")
        return None

    # Extract data for plotting
    epochs = history["_step"].values
    y = history["norm_constant"].values

    # Filter out nans
    mask = ~np.isnan(y)
    epochs = epochs[mask]
    y = y[mask]

    return [output_size], y[:1].tolist(), None, None, run.name


def get_points(run, metric_name, color_fn=None, label_fn=None):
    # Get the history of specific metrics
    history = run.history()
    config = json.loads(run.json_config)
    output_size = config["output_size"]["value"]

    # Ensure the '_step' column is present
    if "_step" not in history.columns:
        print(f"Run {run.id} does not have '_step' column.")
        return None

    # Extract data for plotting
    epochs = history["_step"].values
    y = history[metric_name].values

    # Filter out nans
    mask = ~np.isnan(y)
    epochs = epochs[mask]
    y = y[mask]

    color = color_fn(run) if color_fn else None
    label = label_fn(run) if label_fn else run.name
    return epochs.tolist(), y.tolist(), None, color, label


def se_color_fn(run):
    config = json.loads(run.json_config)
    output_size = config["output_size"]["value"]
    # Different RGB grades of RED for different output sizes in range [1, 12]
    return (0.2 + (0.5 * output_size / 12), 0, 0)


experiments = {
    "NORM_CONST": {
        "title": None,
        "filename": "norm_const_vs_output_size",
        "ylabel": "Norm Constant",
        "xlabel": "Sequence Length",
        "metric_fn": get_norm_const_points,
        "plot_type": "scatter",  # or "scatter"
        "legend": False,
        "aggregate": True,
        "valid_exps_list": [
            "r2_o11_v4_nabs_b5_n_20000_l100_lr0.0001_e1e-06_ep0.0_irandn_positive",
            "r2_o10_v4_nabs_b5_n_20000_l100_lr0.0001_e1e-06_ep0.0_irandn_positive",
            "r2_o9_v4_nabs_b5_n_20000_l100_lr0.0001_e1e-06_ep0.0_irandn_positive",
            "r2_o8_v4_nabs_b5_n_20000_l100_lr0.0001_e1e-06_ep0.0_irandn_positive",
            "r2_o7_v4_nabs_b5_n_20000_l100_lr0.0001_e1e-06_ep0.0_irandn_positive",
            "r2_o6_v4_nabs_b5_n_20000_l100_lr0.0001_e1e-06_ep0.0_irandn_positive",
            "r2_o5_v4_nabs_b5_n_20000_l100_lr0.0001_e1e-06_ep0.0_irandn_positive",
            "r2_o4_v4_nabs_b5_n_20000_l100_lr0.0001_e1e-06_ep0.0_irandn_positive",
            "r2_o3_v4_nabs_b5_n_20000_l100_lr0.0001_e1e-06_ep0.0_irandn_positive",
        ],
    },
    "SE_MAX": {
        "title": None,
        "filename": "se_max_vs_epoch",
        "ylabel": "Max Squared Error",
        "xlabel": "Epoch",
        "metric_fn": lambda run: get_points(
            run,
            "SSE_MAX",
            color_fn=se_color_fn,
            label_fn=lambda run: f"S={json.loads(run.json_config)['output_size']['value']}",
        ),
        "plot_type": "line",  # or "scatter"
        "legend": True,
        "aggregate": False,
        "valid_exps_list": [
            "r2_o11_v4_nabs_b5_n_20000_l100_lr0.0001_e1e-06_ep0.0_irandn_positive",
            "r2_o10_v4_nabs_b5_n_20000_l100_lr0.0001_e1e-06_ep0.0_irandn_positive",
            "r2_o9_v4_nabs_b5_n_20000_l100_lr0.0001_e1e-06_ep0.0_irandn_positive",
            "r2_o8_v4_nabs_b5_n_20000_l100_lr0.0001_e1e-06_ep0.0_irandn_positive",
            "r2_o7_v4_nabs_b5_n_20000_l100_lr0.0001_e1e-06_ep0.0_irandn_positive",
            "r2_o6_v4_nabs_b5_n_20000_l100_lr0.0001_e1e-06_ep0.0_irandn_positive",
            "r2_o5_v4_nabs_b5_n_20000_l100_lr0.0001_e1e-06_ep0.0_irandn_positive",
            "r2_o4_v4_nabs_b5_n_20000_l100_lr0.0001_e1e-06_ep0.0_irandn_positive",
            "r2_o3_v4_nabs_b5_n_20000_l100_lr0.0001_e1e-06_ep0.0_irandn_positive",
        ],
    },
}


def main(exp_name):

    exp_config = experiments[exp_name]

    # Initialize W&B API
    api = wandb.Api()

    # Specify the project name
    project_name = "marawan-gamal/tjdnet"

    # Fetch all runs in the project
    runs = api.runs(f"{project_name}")

    plt.figure(figsize=(10, 5))

    # Loop through all runs
    x_tot, y_tot = [], []
    for run in runs:

        # Check if the run satisfies the regex
        if not run.name in exp_config["valid_exps_list"]:
            continue

        x, y, marker, color, label = exp_config["metric_fn"](run)
        x_tot.extend(x)
        y_tot.extend(y)
        if exp_config["plot_type"] == "scatter" and not exp_config["aggregate"]:
            plt.scatter(x, y, marker=marker, color=color, label=label)
        elif not exp_config["aggregate"]:
            plt.plot(x, y, marker=marker, color=color, label=label)

    if exp_config["aggregate"]:
        plt.plot(x_tot, y_tot, linestyle="--", marker="x", color="blue")

    # Set the title and labels
    if exp_config["title"]:
        plt.title(exp_config["title"])
    plt.xlabel(exp_config["xlabel"])
    plt.ylabel(exp_config["ylabel"])
    if exp_config["legend"]:
        plt.legend()
    plt.savefig(f"{exp_config['filename']}.pdf", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main("NORM_CONST")
