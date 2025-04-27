import itertools
from utils.utils import group_arr, plot_groups


def main():

    exps = (
        [
            {"name": "cp", "rank": r, "horizon": h, "latency": r + h}
            for (r, h) in itertools.product([1, 2, 4], [1, 2, 3])
        ]
        + [
            {"name": "ucp", "rank": r, "horizon": h, "latency": 1 / 2 * (r + h)}
            for (r, h) in itertools.product([1, 2, 4], [1, 2, 3])
        ]
        + [
            {"name": "mps", "rank": r, "horizon": h, "latency": r**2 + h}
            for (r, h) in itertools.product([1, 2, 4], [1, 2, 3])
        ]
        + [
            {"name": "umps", "rank": r, "horizon": h, "latency": 1 / 8 * (r + h)}
            for (r, h) in itertools.product([1, 2, 4], [1, 2, 3])
        ]
    )

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
        # title="Latency vs. sequence length",
        style_dims=[
            "color",
            "marker",
        ],  # First level controls color, second controls marker
        style_cycles={
            "color": [
                "#0173B2",
                "#DE8F05",
                "#029E73",
                "#D55E00",
                "#CC78BC",
                "#CA9161",
                "#FBAFE4",
                "#949494",
            ]
        },
    )

    print("✔ demo_latency_plot.png written")


if __name__ == "__main__":
    main()
