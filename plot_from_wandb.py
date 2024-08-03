from typing import List
import wandb
import numpy as np
import re
import json
import matplotlib.pyplot as plt

LEGEND_FONT_SIZE = 20
AXIS_LABEL_FONT_SIZE = 18
TICK_LABEL_FONT_SIZE = 16

plt.rcParams["axes.labelsize"] = AXIS_LABEL_FONT_SIZE
plt.rcParams["xtick.labelsize"] = TICK_LABEL_FONT_SIZE
plt.rcParams["ytick.labelsize"] = TICK_LABEL_FONT_SIZE


def get_norm_const_points(runs_list: List):
    # Get the history of specific metrics

    ys = []
    xs = []
    for run in runs_list:
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

        xs.append(output_size)
        ys.append(y[0])

    color = "red" if "loentropy" in runs_list[0].name else "blue"
    label = "entropy" if "loentropy" in runs_list[0].name else "preference"
    marker = "^" if "preference" in runs_list[0].name else "x"
    linestyle = "--" if "preference" in runs_list[0].name else "-"

    return xs, ys, marker, color, linestyle, label


def get_points(
    run, metric_name, color_fn=None, label_fn=None, marker_fn=None, linestyle_fn=None
):
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
    marker = marker_fn(run) if marker_fn else None
    linestyle = linestyle_fn(run) if linestyle_fn else "-"
    return epochs.tolist(), y.tolist(), marker, color, linestyle, label


def se_color_fn(run):
    config = json.loads(run.json_config)
    output_size = config["output_size"]["value"]

    min_val = 0.4
    max_val = 1.0 - min_val

    color = (
        (min_val + (max_val * output_size / 12), 0, 0)
        if "loentropy" in run.name
        else (0, 0, min_val + (max_val * output_size / 12))
    )
    return color


experiments = {
    "NORM_CONST": {
        "title": None,
        "filename": "figures/norm_const_vs_output_size.pdf",
        "ylabel": "Max Floating Point Value",
        "xlabel": "Sequence Length",
        "aggregate": True,
        "metric_fn_aggr": get_norm_const_points,
        "plot_type": "scatter",  # or "scatter"
        "legend": True,
        "aggregations": [
            [
                "tdirac_r2_o10_v4_nabs_b16_n_2000_l100_lr0.001_e1e-06_ep0.0_aFalse_irandn_positive_loentropy",
                # tdirac_r2_o10_v4_nabs_b16_n_2000_l100_lr0.001_e1e-06_ep0.0_aFalse_iuniform_positive_lopreference SSE_MAX	0.9999920129776001
                "tdirac_r2_o8_v4_nabs_b16_n_2000_l100_lr0.001_e1e-06_ep0.0_aFalse_irandn_positive_loentropy",
                "tdirac_r2_o6_v4_nabs_b16_n_2000_l100_lr0.001_e1e-06_ep0.0_aFalse_irandn_positive_loentropy",
                "tdirac_r2_o4_v4_nabs_b16_n_2000_l100_lr0.001_e1e-06_ep0.0_aFalse_irandn_positive_loentropy",
            ],
            [
                "tdirac_r2_o10_v4_nabs_b16_n_2000_l100_lr0.001_e1e-06_ep0.0_aFalse_iuniform_positive_lopreference",
                "tdirac_r2_o8_v4_nabs_b16_n_2000_l100_lr0.001_e1e-06_ep0.0_aFalse_iuniform_positive_lopreference",
                "tdirac_r2_o6_v4_nabs_b16_n_2000_l100_lr0.001_e1e-06_ep0.0_aFalse_iuniform_positive_lopreference",
                "tdirac_r2_o4_v4_nabs_b16_n_2000_l100_lr0.001_e1e-06_ep0.0_aFalse_iuniform_positive_lopreference",
            ],
        ],
    },
    "SE_MAX": {
        "title": None,
        "filename": "figures/se_max_vs_epoch.pdf",
        "ylabel": "Max Squared Error",
        "xlabel": "Epoch",
        "metric_fn": lambda run: get_points(
            run,
            "SSE_MAX",
            color_fn=se_color_fn,
            label_fn=lambda run: (
                # f"S={json.loads(run.json_config)['output_size']['value']} {'(preference loss)' if 'lopreference' in run.name else ''}"
                f"S={json.loads(run.json_config)['output_size']['value']} {'(preference)' if 'lopreference' in run.name else '(entropy)'}"
            ),
            linestyle_fn=lambda run: "--" if "lopreference" in run.name else "-",
        ),
        "plot_type": "line",  # or "scatter"
        "legend": True,
        "aggregate": False,
        "valid_exps_list": [
            "tdirac_r2_o10_v4_nabs_b16_n_2000_l100_lr0.001_e1e-06_ep0.0_aFalse_irandn_positive_loentropy",
            "tdirac_r2_o8_v4_nabs_b16_n_2000_l100_lr0.001_e1e-06_ep0.0_aFalse_irandn_positive_loentropy",
            # "tdirac_r2_o6_v4_nabs_b16_n_2000_l100_lr0.001_e1e-06_ep0.0_aFalse_irandn_positive_loentropy",
            "tdirac_r2_o4_v4_nabs_b16_n_2000_l100_lr0.001_e1e-06_ep0.0_aFalse_irandn_positive_loentropy",
            "tdirac_r2_o10_v4_nabs_b16_n_2000_l100_lr0.001_e1e-06_ep0.0_aFalse_iuniform_positive_lopreference",
            "tdirac_r2_o8_v4_nabs_b16_n_2000_l100_lr0.001_e1e-06_ep0.0_aFalse_iuniform_positive_lopreference",
            # "r2tdirac__o6_v4_nabs_b16_n_2000_l100_lr0.001_e1e-06_ep0.0_aFalse_iuniform_positive_lopreference",
            "tdirac_r2_o4_v4_nabs_b16_n_2000_l100_lr0.001_e1e-06_ep0.0_aFalse_iuniform_positive_lopreference",
        ],
    },
}


def main_aggr(exp_name):

    exp_config = experiments[exp_name]

    # Initialize W&B API
    api = wandb.Api()

    # Specify the project name
    project_name = "marawan-gamal/TJDNet (Synthetic)"

    # Fetch all runs in the project
    runs = api.runs(f"{project_name}")

    aggregation_name_lists = exp_config["aggregations"]
    aggregations = []
    for aggregation_names in aggregation_name_lists:
        aggregations_inner = []
        for run in runs:
            if run.name in aggregation_names:
                aggregations_inner.append(run)
        aggregations.append(aggregations_inner)

    plt.figure(figsize=(10, 5))

    for aggregation in aggregations:
        x, y, marker, color, linestyle, label = exp_config["metric_fn_aggr"](
            aggregation
        )
        plt.plot(x, y, marker=marker, color=color, label=label, linestyle=linestyle)

    # Set the title and labels
    if exp_config["title"]:
        plt.title(exp_config["title"])
    plt.xlabel(exp_config["xlabel"])
    plt.ylabel(exp_config["ylabel"])
    if exp_config["legend"]:
        plt.legend(prop={"size": LEGEND_FONT_SIZE})
    plt.savefig(exp_config["filename"], bbox_inches="tight", dpi=300)


def main_basic(exp_name):

    exp_config = experiments[exp_name]

    # Initialize W&B API
    api = wandb.Api()

    # Specify the project name
    project_name = "marawan-gamal/TJDNet (Synthetic)"

    # Fetch all runs in the project
    runs = api.runs(f"{project_name}")

    plt.figure(figsize=(10, 5))

    # Loop through all runs
    x_tot, y_tot = [], []
    for run in runs:

        # Check if the run satisfies the regex
        if not run.name in exp_config["valid_exps_list"]:
            continue

        x, y, marker, color, linestyle, label = exp_config["metric_fn"](run)
        x_tot.extend(x)
        y_tot.extend(y)
        if exp_config["plot_type"] == "scatter" and not exp_config["aggregate"]:
            plt.scatter(x, y, marker=marker, color=color, label=label)
        elif not exp_config["aggregate"]:
            plt.plot(x, y, marker=marker, color=color, label=label, linestyle=linestyle)

    if exp_config["aggregate"]:
        plt.plot(x_tot, y_tot, linestyle="--", marker="x", color="blue")

    # Set the title and labels
    if exp_config["title"]:
        plt.title(exp_config["title"])
    plt.xlabel(exp_config["xlabel"])
    plt.ylabel(exp_config["ylabel"])
    if exp_config["legend"]:
        plt.legend(prop={"size": LEGEND_FONT_SIZE})
    plt.savefig(exp_config["filename"], bbox_inches="tight", dpi=300)


def main(exp_name):

    exp_config = experiments[exp_name]
    if exp_config["aggregate"]:
        main_aggr(exp_name)
    else:
        main_basic(exp_name)


if __name__ == "__main__":
    main("SE_MAX")
    main("NORM_CONST")
