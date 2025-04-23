from __future__ import annotations

from typing import Optional, Any, Dict, List, Sequence, Tuple, Union
from collections.abc import Mapping

import torch
import matplotlib.pyplot as plt
from pathlib import Path

from collections import defaultdict
from itertools import cycle


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
    path: Tuple[str, ...] = (),
) -> List[Tuple[Tuple[str, ...], List[Dict[str, Any]]]]:
    if isinstance(node, list):
        return [(path, node)]
    leaves: List[Tuple[Tuple[str, ...], List[Dict[str, Any]]]] = []
    for key, child in node.items():
        leaves.extend(_walk_groups(child, path + (str(key),)))
    return leaves


def plot_groups(
    grouped: Dict[str, Any],
    x_key: str,
    y_key: str,
    path: Union[str, Path],
    *,
    style_dims: Sequence[str] = ("marker", "color"),
    style_cycles: Union[Dict[str, Sequence[Any]], None] = None,
    custom_maps: Union[Dict[str, Dict[str, Any]], None] = None,
    title: Union[str, None] = None,
    x_label: Union[str, None] = None,
    y_label: Union[str, None] = None,
    figsize: Tuple[int, int] = (10, 6),
    sort_by_x: bool = True,
) -> None:
    """
    Plot a line for every leaf in a ``group_arr`` result, mapping successive
    group levels to Matplotlib style properties.

    Parameters
    ----------
    grouped : dict
        Nested dictionary produced by :func:`group_arr`.
    x_key, y_key : str
        Keys in each result dictionary to read the *x* and *y* values from.
    path : str or pathlib.Path
        Output file (extension decides image format).  Missing parent
        directories are created automatically.
    style_dims
        Ordered sequence of matplotlib line properties that will be
        controlled by the first, second, … group levels.
        Example: ("marker", "color", "linestyle").
    style_cycles
        Optional dict overriding the default cycle for a given property:
        {"marker": ["o", "s"], "linestyle": ["-", "--"]}.
    custom_maps
        Pre-assign specific style values to some keys:
        {"color": {"dev": "tab:red", "prod": "tab:green"}}.
    title, x_label, y_label : str, optional
        Plot title and axis labels.  The axis labels default to *x_key* /
        *y_key* when omitted.
    figsize : tuple[int, int], default ``(10, 6)``
        Figure size in inches.
    sort_by_x : bool, default ``True``
        If *True*, each (x, y) series is sorted by *x* before plotting.

    Raises
    ------
    ValueError
        If *grouped* is empty.

    Notes
    -----
    * Group levels beyond ``len(style_dims)`` are **not** visually encoded
      but are concatenated into the legend label.
    * Matplotlib default colour cycle is used for ``"color"`` whenever no
      custom cycle or map is supplied.

    Returns
    -------
    None
        The plot is written to *path* and the figure is closed.
    """

    leaves = _walk_groups(grouped)
    if not leaves:
        raise ValueError("Nothing to plot – `grouped` is empty")

    style_cycles = style_cycles or {}
    custom_maps = custom_maps or {}

    # Build a default cycle for any property not supplied
    default_prop_cycles: Dict[str, cycle] = {}
    for prop in style_dims:
        if prop in style_cycles:
            default_prop_cycles[prop] = cycle(style_cycles[prop])
        else:
            # Good defaults: let marker cycle through a few symbols,
            # colour cycle uses rcParams, others fallback to sensible lists
            if prop == "marker":
                default_prop_cycles[prop] = cycle("o^sDPv<>")
            elif prop == "linestyle":
                default_prop_cycles[prop] = cycle(("-", "--", ":", "-."))
            else:
                default_prop_cycles[prop] = cycle(
                    plt.rcParams["axes.prop_cycle"].by_key()["color"]
                )

    # Remember which key already received which value
    assigned: Dict[str, Dict[str, Any]] = defaultdict(dict)
    assigned.update(custom_maps)  # pre-seed with user maps

    def get_style(prop: str, key: str) -> Any:
        if key not in assigned[prop]:
            assigned[prop][key] = next(default_prop_cycles[prop])
        return assigned[prop][key]

    plt.figure(figsize=figsize)

    for path_tuple, items in leaves:
        xs = [d[x_key] for d in items]
        ys = [d[y_key] for d in items]
        if sort_by_x:
            xs, ys = zip(*sorted(zip(xs, ys), key=lambda t: t[0]))

        # Build kwargs incrementally
        style_kwargs: Dict[str, Any] = {}
        for lvl, key in enumerate(path_tuple[: len(style_dims)]):
            prop = style_dims[lvl]
            style_kwargs[prop] = get_style(prop, key)

        # Anything beyond the styled levels just contributes to the label
        label = " / ".join(path_tuple)

        plt.plot(xs, ys, label=label, **style_kwargs)

    plt.xlabel(x_label or x_key)
    plt.ylabel(y_label or y_key)
    if title:
        plt.title(title)
    if len(leaves) > 1:
        plt.legend()
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()
