from utils.utils import group_arr, plot_groups


def main():

    exps = [
        {"name": "cp", "rank": r, "horizon": h, "latency": r + h**2}
        for (r, h) in zip([1, 2, 3], [2, 2, 2])
    ] + [
        {"name": "ucp", "rank": r, "horizon": h, "latency": r + h}
        for (r, h) in zip([1, 2, 3], [2, 2, 2])
    ]

    # === Group by model first, then by horizon ===
    grouped = group_arr(
        exps,
        lambda d: d["name"],
        lambda d: d["horizon"],
    )

    # Plot latency vs sequence length for every model
    plot_groups(
        grouped,
        x_key="rank",
        y_key="latency",
        path="demo_latency_plot.png",
        title="Latency vs. sequence length",
    )
    print("✔ demo_latency_plot.png written")


if __name__ == "__main__":
    main()
