import argparse
import json
import os
import os.path as osp
import pandas as pd


import re

patterns = {
    "e": r"e(\d{1,2})_",  # e followed by 1-2 digits then underscore
    "m": r"(?:^|_)m([a-zA-Z0-9]+)",
    "mh": r"(?:^|_)mh([a-zA-Z0-9]+)",
    "umel": r"(?:^|_)umel(True|False)",
    "r": r"(?:^|_)r(\d+)",
    "h": r"(?:^|_)h(\d+)(?:_|$)",
    "hd": r"(?:^|_)hd(\d+)",
    "lr": r"(?:^|_)l(\d+(?:e_\d+|\.\d+))",
}

pd.set_option("display.max_colwidth", 150)


def get_friendly_name(exp):
    # Parse important config vals: *e, *m, *mh, *r, *h, *hd, _umel

    parts = []
    for key, pattern in patterns.items():
        match = re.search(pattern, exp)
        if match:
            parts.append(f"{key}{match.group(1)}")

    # Join with double colons
    return "::".join(parts)


def main(args):
    results = []

    for exp in os.listdir(args.dir):
        exp_path = os.path.join(args.dir, exp)
        if not os.path.isdir(exp_path):
            continue

        # Apply filter if provided
        if args.filter and args.filter not in exp:
            continue

        # Get accuracy and latency
        accuracy = None
        accuracy_progress = None
        latency = None
        params = None
        epoch = None
        epochs = None
        progress = None

        # Find accuracy file
        acc_file = os.path.join(exp_path, "eval_results_accuracy_b1_t128.json")
        if osp.exists(acc_file):
            with open(acc_file) as f:
                ckpt_results = json.load(f)
                # find bes accuracy
                if len(ckpt_results.keys()) > 0:
                    # Get the best accuracy
                    ckpt_accs = [
                        (v.get("avg"), v.get("count"), v.get("total_samples"))
                        for v in ckpt_results.values()
                        if v.get("count") == v.get("total_samples")
                        or v.get("count") >= 50
                    ]
                    if ckpt_accs:
                        best_result = max(
                            ckpt_accs, key=lambda x: x[0] if x[0] is not None else -1
                        )
                        accuracy = best_result[0]
                        accuracy_num_samples = best_result[1]
                        accuracy_num_samples_total = best_result[2]
                        accuracy_progress = f"{accuracy_num_samples / accuracy_num_samples_total * 100:.0f}% {accuracy_num_samples}/{accuracy_num_samples_total}"

        # Find latency file
        lat_file = os.path.join(exp_path, "eval_results_latency.json")
        if osp.exists(lat_file):
            with open(lat_file) as f:
                lat_results = json.load(f)
                latency = lat_results["modes"]["eval"]["Latency [s]"]["mean"]
                params = lat_results["modes"]["eval"]["Params [M]"]

        # Find latest checkpoint file
        ckpts = [f for f in os.listdir(exp_path) if f.startswith("checkpoint-")]
        if len(ckpts) > 0:
            latest_ckpt_file = os.path.join(
                exp_path, sorted(ckpts)[-1], "trainer_state.json"
            )
            if os.path.exists(latest_ckpt_file):
                with open(latest_ckpt_file) as f:
                    ckpt_args = json.load(f)
                    epoch = ckpt_args.get("epoch", None)

                args_file = os.path.join(exp_path, "args.json")
                if os.path.exists(args_file):
                    with open(args_file) as f:
                        exp_args = json.load(f)
                        epochs = exp_args.get("epochs", None)
                        prog_percent = epoch / epochs * 100
                        progress = f"{prog_percent:.0f}% ({epoch}/{epochs})"

        name = get_friendly_name(exp) if args.friendly_name else exp
        results.append([name, progress, accuracy, accuracy_progress, latency, params])

    # Create and print dataframe
    df = pd.DataFrame(
        results,
        columns=[
            "experiment",
            "train_progress",
            "accuracy",
            "accuracy_progress",
            "latency",
            "params [M]",
        ],
    )

    # Sort by specified column if provided, otherwise sort by accuracy
    sort_col = args.sort if args.sort in df.columns else "experiment"
    ascending = not args.desc
    df = df.sort_values(sort_col, ascending=ascending)

    # print markdown table
    print(df.to_markdown(index=False))
    print(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the model")
    parser.add_argument(
        "-d", "--dir", type=str, required=True, help="Directory containing experiments"
    )
    parser.add_argument(
        "-f", "--friendly_name", action="store_true", help="Use friendly names"
    )
    # Add new filter and sort arguments
    parser.add_argument(
        "--filter", type=str, help="Filter experiments containing this string"
    )
    parser.add_argument(
        "--sort",
        type=str,
        default="experiment",
        help="Column to sort by (experiment, train_progress, accuracy, accuracy_progress, latency)",
    )
    parser.add_argument("--desc", action="store_true", help="Sort in descending order")
    main(parser.parse_args())
