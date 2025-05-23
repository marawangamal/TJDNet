#!/usr/bin/env python3
import argparse
import os
import subprocess
from collections import defaultdict
from pathlib import Path
import torch


def load_experiment(exp_path):
    """Load experiment eval_loss and hyperparams."""
    ckpt_path = exp_path / "best.ckpt"
    if not ckpt_path.exists():
        return None

    meta_path = ckpt_path / "meta.pt" if ckpt_path.is_dir() else ckpt_path
    try:
        ckpt = torch.load(meta_path, map_location="cpu")
        hparams = ckpt["hyper_parameters"]

        # Find ModelCheckpoint callback with eval_loss
        for key, cb in ckpt.get("callbacks", {}).items():
            if "ModelCheckpoint" in key and "eval_loss" in key:
                return cb["best_model_score"], hparams
        return None
    except:
        return None


def main(args):
    exp_dir = Path(args.experiments)

    # Load and filter experiments
    experiments = {}
    target_group = args.group_id.split("-")[args.group_level]

    for exp_path in exp_dir.iterdir():
        if not exp_path.is_dir():
            continue

        data = load_experiment(exp_path)
        if data is None:
            continue

        eval_loss, hparams = data
        exp_group = hparams.get("group_id", "")
        if not exp_group:
            continue
        exp_group = exp_group.split("-")

        if (
            len(exp_group) > args.group_level
            and exp_group[args.group_level] == target_group
        ):
            experiments[exp_path.name] = (eval_loss, hparams)

    if not experiments:
        print("No matching experiments found")
        return

    # Group experiments
    if args.group_by:
        groups = defaultdict(dict)
        for name, (loss, hparams) in experiments.items():
            # Create composite key from multiple group_by parameters
            key_parts = []
            for param in args.group_by:
                value = hparams.get(param, "unknown")
                key_parts.append(f"{param}={value}")
            key = ", ".join(key_parts)
            groups[key][name] = (loss, hparams)
    else:
        groups = {"all": experiments}

    # Find and tag best in each group
    best_models = set()
    for group_name, group_exps in groups.items():
        best_name = min(group_exps.keys(), key=lambda k: group_exps[k][0])
        best_loss = group_exps[best_name][0]

        # Tag best
        with open(exp_dir / best_name / ".best", "w") as f:
            f.write(f"Best model: {best_name}\nEval loss: {best_loss:.6f}\n")

        # Consolidate if directory
        ckpt_path = exp_dir / best_name / "best.ckpt"
        if ckpt_path.is_dir() and args.consolidate:
            subprocess.run(
                [
                    "python",
                    "-m",
                    "lightning.pytorch.utilities.consolidate_checkpoint",
                    str(ckpt_path),
                ],
                capture_output=True,
            )

        best_models.add(best_name)
        print(f"Best in {group_name}: {best_name} ({best_loss:.6f})")

    # Remove .best from others
    for name in experiments:
        if name not in best_models:
            best_file = exp_dir / name / ".best"
            best_file.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tag the best model")
    parser.add_argument(
        "--experiments", default="experiments", help="Experiments directory"
    )
    parser.add_argument(
        "--group_level", type=int, default=0, help="Group level to filter"
    )
    parser.add_argument("--group_id", required=True, help="Group ID to filter")
    parser.add_argument(
        "--group_by",
        nargs="+",
        help="Tag best within each group_by attr (can specify multiple attributes)",
    )
    parser.add_argument(
        "--consolidate",
        action="store_true",
        help="Consolidate the best checkpoint if it is a directory",
        default=False,
    )

    main(parser.parse_args())
