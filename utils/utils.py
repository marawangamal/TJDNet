from __future__ import annotations

import re
from typing import Optional, Any, Dict, List, Sequence, Tuple, Union
from collections.abc import Mapping

import torch
import matplotlib.pyplot as plt
from pathlib import Path

from collections import defaultdict
from itertools import cycle


PlotKw = Mapping[str, Any]


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


def replace_spec_chars(value: str, replacement: str = "_") -> str:
    # Remove special characters and replace with underscores
    return re.sub(r"[^a-zA-Z0-9]", replacement, str(value))


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

            abbrevs[abbrev] = replace_spec_chars(value)

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
    path: Tuple[str, ...] = (),
) -> List[Tuple[Tuple[str, ...], List[Dict[str, Any]]]]:
    if isinstance(node, list):
        return [(path, node)]
    leaves: List[Tuple[Tuple[str, ...], List[Dict[str, Any]]]] = []
    for key, child in node.items():
        leaves.extend(_walk_groups(child, path + (str(key),)))
    return leaves


def _build_default_cycles(
    style_dims: Sequence[str],
    style_cycles: Dict[str, Sequence[Any]],
) -> Dict[str, cycle]:
    """
    Create an endless cycle of style values for every property in *style_dims*,
    falling back to sensible defaults when the user hasn not supplied one.

    - "marker"   → simple marker list
    - "linestyle"→ common dash patterns
    - anything else (typically "color") → Matplotlib's default colour cycle
    """
    default_prop_cycles: Dict[str, cycle] = {}
    for prop in style_dims:
        if prop in style_cycles and style_cycles[prop]:
            default_prop_cycles[prop] = cycle(style_cycles[prop])
        else:
            if prop == "marker":
                default_prop_cycles[prop] = cycle("o ^ s D P v < >".split())
            elif prop == "linestyle":
                default_prop_cycles[prop] = cycle(("-", "--", ":", "-."))
            else:  # colour or any other prop
                default_prop_cycles[prop] = cycle(
                    plt.rcParams["axes.prop_cycle"].by_key()["color"]
                )
    return default_prop_cycles


def plot_groups(
    grouped: Dict[str, Any],
    x_key: str,
    y_key: str,
    path: Union[str, Path],
    *,
    style_dims: Sequence[str] = ("marker", "color"),
    style_cycles: Optional[Dict[str, Sequence[Any]]] = None,
    custom_maps: Optional[Dict[str, Dict[str, Any]]] = None,
    fig_kwargs: PlotKw | None = None,  # NEW ➊
    axes_kwargs: PlotKw | None = None,  # NEW ➋
    line_kwargs: (
        PlotKw | None
    ) = None,  # NEW ➌  (applied to every line *in addition* to auto styles)
) -> None:
    """
    Plot a line for every leaf of ``grouped``.  All Matplotlib customisation
    lives in three dicts:

    fig_kwargs  → forwarded to ``plt.figure``
    axes_kwargs → forwarded to ``ax.set(**axes_kwargs)``
    line_kwargs → merged into per-line style before ``ax.plot``

    This keeps the signature stable while still letting users tweak anything
    Matplotlib supports.
    """
    if not grouped:
        raise ValueError("Nothing to plot – `grouped` is empty")

    style_cycles = style_cycles or {}
    custom_maps = custom_maps or {}
    fig_kwargs = fig_kwargs or {}
    axes_kwargs = axes_kwargs or {}
    line_kwargs = line_kwargs or {}

    # ------------------------------------------------------------------ setup
    leaves = _walk_groups(grouped)
    default_prop_cycles = _build_default_cycles(style_dims, style_cycles)
    assigned: Dict[str, Dict[str, Any]] = defaultdict(dict, custom_maps)

    def get_style(prop: str, key: str) -> Any:
        if key not in assigned[prop]:
            assigned[prop][key] = next(default_prop_cycles[prop])
        return assigned[prop][key]

    plt.figure(**({"figsize": (10, 6)} | fig_kwargs))  # ➊
    ax = plt.gca()

    # ----------------------------------------------------------- draw lines
    for path_tuple, items in leaves:
        xs = [d[x_key] for d in items]
        ys = [d[y_key] for d in items]
        xs, ys = zip(*sorted(zip(xs, ys)))  # always sort on x

        # auto-assigned styles from group keys
        auto = {prop: get_style(prop, key) for prop, key in zip(style_dims, path_tuple)}
        ax.plot(xs, ys, label=" / ".join(path_tuple), **auto, **line_kwargs)  # ➌

    # ---------------------------------------------------------- axes stuff
    if len(leaves) > 1:
        ax.legend()
    ax.set(**axes_kwargs)  # ➋
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.figure.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(path)
    plt.close(ax.figure)
