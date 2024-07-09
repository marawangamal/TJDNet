import torch
from typing import Dict, Optional


def get_experiment_name(
    configs: Dict,
    abbrevs: Optional[Dict] = None,
) -> str:
    """Create an experiment name from the configuration dictionary.

    Args:
        configs (Dict): Experiment configuration dictionary.
        abbrevs (Optional[Dict], optional): Working dictionary of abbreviations used in recursive calls. Defaults to None.
        mode (str, optional): Return mode. Defaults to "dict".

    Raises:
        ValueError: Abbreviation not found for key.

    Returns:
        str: Experiment name.
    """

    if abbrevs is None:
        abbrevs = {}

    for key, value in configs.items():
        if isinstance(value, dict):
            get_experiment_name(value, abbrevs=abbrevs)
        else:
            i = 1
            while i <= len(key):
                if key[:i] not in abbrevs:
                    abbrevs[key[:i]] = (
                        str(value)
                        .replace(" ", "")
                        .replace(",", "_")
                        .replace("[", "")
                        .replace("]", "")
                    )
                    break
                i += 1

                if i == len(key) + 1:
                    raise ValueError(
                        "Could not find a suitable abbreviation for key: {}".format(key)
                    )

    return "_".join(["{}{}".format(k, v) for k, v in abbrevs.items()])
