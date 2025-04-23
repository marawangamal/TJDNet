from typing import Optional, Any, Dict, List, Tuple, Union
from collections.abc import Mapping

import torch
import matplotlib.pyplot as plt
from pathlib import Path


class AverageMeter:
    """Computes and stores the average and current value

    Attributes:
        val (float): Last recorded value
        sum (float): Sum of all recorded values
        count (int): Count of recorded values
        avg (float): Running average of recorded values
    """

    def __init__(self, sum: float = 0, count: int = 0, **kwargs):
        """Initialize the AverageMeter"""
        self.reset()
        self.sum = sum
        self.count = count
        self.avg = sum / count if count != 0 else 0

    def reset(self):
        """Reset all statistics"""
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        """Update statistics given new value and optional count

        Args:
            val (float): Value to record
            n (int, optional): Number of values represented by val. Defaults to 1.
        """
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

    def dump(self):
        """Return the current statistics"""
        return {"sum": self.sum, "count": self.count, "avg": self.avg}


def get_experiment_name(
    configs: Dict,
    abbrevs: Optional[Dict] = None,
) -> str:
    """Create an experiment name from the configuration dictionary.
    Args:
        configs (Dict): Experiment configuration dictionary.
        abbrevs (Optional[Dict], optional): Working dictionary of abbreviations used in recursive calls. Defaults to None.
    Raises:
        ValueError: Abbreviation not found for key.
    Returns:
        str: Experiment name.
    """
    if abbrevs is None:
        abbrevs = {}

    def get_abbreviation(key: str) -> str:
        if "_" in key:
            parts = key.split("_")
            return "".join(p[0] for p in parts)
        return key[0]

    for key, value in configs.items():
        if isinstance(value, dict):
            get_experiment_name(value, abbrevs=abbrevs)
        else:
            abbrev = get_abbreviation(key)
            i = 1
            while abbrev in abbrevs:
                if i == len(key):
                    raise ValueError(
                        f"Could not find a suitable abbreviation for key: {key}"
                    )
                abbrev = key[: i + 1]
                i += 1

            abbrevs[abbrev] = (
                str(value)
                .replace(" ", "")
                .replace(",", "_")
                .replace("[", "")
                .replace("]", "")
            )

    return "_".join(f"{k}{v}" for k, v in abbrevs.items())


def truncate_tens(tens: torch.Tensor, val: int):
    """Truncate tensor after encountering a value"""
    idx = torch.argmax((tens == val).int())
    return tens[: idx + 1]


def group_arr(results: list, *group_fns):
    """Groups items in an array. (Eg. array[t] => {k: {k1: array[t'], k2: array[t'']}}).

    Args:
        results (list): List of dictionaries with keys 'name', 'x_axis', 'y_axis', 'group'
        group_fns (list): List of functions to group by. Each function should take a dictionary
            and return a key to group by.

    Returns:
        dict: Nested dictionary with groups defined by group_fns
    """
    grouped = {}

    for result in results:
        # Start with the top-level grouped dictionary
        current = grouped

        # Apply each grouping function in sequence
        for g_idx, group_fn in enumerate(group_fns):
            group_key = group_fn(result)

            # Create the group if it doesn't exist
            if group_key not in current:
                # If this is the last grouping function, initialize with an empty list
                if g_idx == len(group_fns) - 1:
                    current[group_key] = []
                else:
                    current[group_key] = {}

            # Move to the next level for the next grouping function
            if g_idx == len(group_fns) - 1:
                # At the deepest level, append the result
                current[group_key].append(result)
            else:
                # Otherwise, continue nesting
                current = current[group_key]

    return grouped


def _walk_groups(
    node: Union[Mapping, List[Dict[str, Any]]],
    path: Tuple[str, ...] = (),  # just a tuple now
) -> List[Tuple[Tuple[str, ...], List[Dict[str, Any]]]]:
    """Return [(group_path, leaf_items), …] for every leaf in a nested dict."""
    if isinstance(node, list):  # leaf reached
        return [(path, node)]

    leaves: List[Tuple[Tuple[str, ...], List[Dict[str, Any]]]] = []
    for key, child in node.items():
        leaves.extend(_walk_groups(child, path + (str(key),)))
    return leaves


def plot_groups(
    grouped: Dict[str, Any],
    x_key: str,
    y_key: str,
    path: str,
    title: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    figsize: tuple[int, int] = (10, 6),
    sort_by_x: bool = True,
) -> None:
    """
    Plot data held in a nested group structure produced by `group_arr`.

    Args:
        grouped: The nested dictionary returned by `group_arr`.
        x_key: Key in the result dicts to use as the x-axis.
        y_key: Key in the result dicts to use as the y-axis.
        path: File path where the figure will be saved (extension decides format).
        title: Plot title (optional).
        x_label: X-axis label (optional, defaults to `x_key`).
        y_label: Y-axis label (optional, defaults to `y_key`).
        figsize: Figure size in inches.
        sort_by_x: If True, data for each series are sorted by x before plotting.

    Returns:
        None (figure is saved to *path*).
    """
    # Gather every leaf (one series per leaf)
    series = _walk_groups(grouped)

    if not series:
        raise ValueError("Nothing to plot – `grouped` appears to be empty")

    plt.figure(figsize=figsize)

    for group_path, items in series:
        # Extract x and y values
        xs = [item[x_key] for item in items]
        ys = [item[y_key] for item in items]

        if sort_by_x:
            xs, ys = zip(*sorted(zip(xs, ys), key=lambda t: t[0]))

        # Build a readable legend label like "dataset / split / run-1"
        label = " / ".join(map(str, group_path)) if group_path else "all"

        plt.plot(xs, ys, marker="o", label=label)

    plt.xlabel(x_label or x_key)
    plt.ylabel(y_label or y_key)
    if title:
        plt.title(title)

    # Only show legend if there is more than one line
    if len(series) > 1:
        plt.legend()

    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()
