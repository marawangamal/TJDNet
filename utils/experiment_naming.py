from utils.utils import replace_spec_chars


from typing import Dict, Optional


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
