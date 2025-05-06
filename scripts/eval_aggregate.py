import argparse
import json
import os
import os.path as osp
import pandas as pd


import re


def get_friendly_name(exp):
    # Parse important config vals: *e, *m, *mh, *r, *h, *hd, _umel

    patterns = {
        "e": r"e(\d{1,2})_",  # e followed by 1-2 digits then underscore
        "m": r"(?:^|_)m([a-zA-Z0-9]+)",
        "mh": r"(?:^|_)mh([a-zA-Z0-9]+)",
        "r": r"(?:^|_)r(\d+)",
        "h": r"(?:^|_)h(\d+)(?:_|$)",
        "hd": r"(?:^|_)hd(\d+)",
    }

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

        # Get accuracy and latency
        accuracy = None
        latency = None

        # Find accuracy file
        acc_file = os.path.join(exp_path, "eval_results_accuracy_b1_t128.json")
        if osp.exists(acc_file):
            with open(acc_file) as f:
                ckpt_results = json.load(f)
                # find bes accuracy
                if len(ckpt_results.keys()) > 0:
                    # Get the best accuracy
                    accuracy = max(
                        [
                            v.get("avg")
                            for v in ckpt_results.values()
                            if v.get("count") == v.get("total_samples")
                        ]
                        + [-1]
                    )

        # Find latency file
        lat_file = os.path.join(exp_path, "eval_results_latency.json")
        if osp.exists(lat_file):
            with open(lat_file) as f:
                lat_results = json.load(f)
                latency = lat_results["modes"]["eval"]["Latency [s]"]["mean"]

        results.append([get_friendly_name(exp), accuracy, latency])

    # Create and print dataframe
    df = pd.DataFrame(results, columns=["experiment", "accuracy", "latency"])
    df = df.sort_values("accuracy", ascending=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the model")
    parser.add_argument(
        "-d", "--dir", type=str, required=True, help="Directory containing experiments"
    )
    main(parser.parse_args())
