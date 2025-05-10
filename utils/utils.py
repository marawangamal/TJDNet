from __future__ import annotations

import re
from typing import Optional, Any, Dict, List, Sequence, Tuple, Union
from collections.abc import Mapping
import uuid

import numpy as np
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


def plot_conf_bands(
    x_values, y_values_by_group, save_path, title="Spectrum with Confidence Bands"
):
    """
    Plot confidence bands for grouped data with minimal input requirements.

    Parameters:
    -----------
    x_values : list or array
        Array of x values (shared across all groups)
    y_values_by_group : dict
        Dictionary where keys are group names and values are lists of lists.
        Each inner list contains y values for a specific item in the group.
        Example: {'poem': [[0.5, 0.4, 0.3], [0.6, 0.5, 0.4]], 'gsm8k': [[0.2, 0.1, 0.05]]}
    save_path : str
        Path to save the plot
    """
    # Setup figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define colors for groups
    colors = [
        "#0173B2",
        "#DE8F05",
        "#029E73",
        "#D55E00",
        "#CC78BC",
        "#CA9161",
        "#FBAFE4",
        "#949494",
    ]

    # Plot each group with confidence bands
    for i, (group_key, y_lists) in enumerate(y_values_by_group.items()):
        # Convert to numpy array for calculations
        y_array = np.array(y_lists)

        # Compute mean and std dev across samples
        y_mean = np.mean(y_array, axis=0)
        y_std = np.std(y_array, axis=0)

        # Get color for this group
        color = colors[i % len(colors)]

        # Plot mean line
        ax.plot(x_values, y_mean, label=group_key, color=color)

        # Plot confidence band
        ax.fill_between(
            x_values,
            y_mean - y_std,
            y_mean + y_std,
            color=color,
            alpha=0.2,
            edgecolor="none",
        )

    # Set plot parameters
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel("Singular Value")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.3)

    # Save the plot
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"Confidence band plot saved to {save_path}")

    return save_path


def run_checks(spec: list[dict]):
    failures = [
        (i, item["msg"]() if callable(item["msg"]) else item["msg"])
        for i, item in enumerate(spec, 1)  # start index at 1
        if not item["test"]()
    ]
    if failures:
        msgs = "\n".join(f"  {idx}. {m}" for idx, m in failures)
        raise ValueError(f"{len(failures)} check(s) failed:\n{msgs}")


def printr(msg):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    print(f"[RANK {local_rank}] {msg}")


def printo(msg):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        printr(msg)


def generate_wandb_id():
    """Generate a random wandb_id that's compatible with W&B requirements."""
    # Generate a random UUID and take the first 8 characters
    # This gives us plenty of uniqueness while keeping the ID short
    random_id = str(uuid.uuid4()).replace("-", "")[:8]
    return random_id


def setup_dist_class_fsdp_wrapping(model, training_args):
    """Modify FSDP wrapping to include distribution classes."""
    import importlib
    from functools import partial

    # Find all distribution classes in the model
    dist_classes = set()

    def find_dist_classes_in_model(module):
        for name, child in module.named_children():
            if "Dist" in child.__class__.__name__:
                dist_classes.add(child.__class__)
                print(f"Found distribution class: {child.__class__.__name__}")
            find_dist_classes_in_model(child)

    # Find classes in the model
    find_dist_classes_in_model(model)

    # Only proceed if we found distribution classes and have FSDP
    if (
        dist_classes
        and hasattr(training_args, "_fsdp_plugin")
        and training_args._fsdp_plugin is not None
    ):
        # Get existing auto wrap policy
        fsdp_plugin = training_args._fsdp_plugin
        existing_policy = getattr(fsdp_plugin, "auto_wrap_policy", None)

        # Create a combined policy
        def combined_wrap_policy(module, recurse=True, **kwargs):
            # Check if module is one of our distribution classes
            if module.__class__ in dist_classes:
                return True

            # Otherwise, use the existing policy
            if callable(existing_policy):
                return existing_policy(module, recurse=recurse, **kwargs)

            return False

        # Apply the combined policy
        fsdp_plugin.auto_wrap_policy = combined_wrap_policy
        print(
            f"Applied custom FSDP wrapping for {len(dist_classes)} distribution classes"
        )
    else:
        print("No distribution classes found or FSDP not active")
