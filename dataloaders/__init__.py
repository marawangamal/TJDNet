from typing import Dict, Type

from dataloaders.aqua import AQUA
from dataloaders.base import AbstractDataset
from dataloaders.csqa import CSQA
from dataloaders.gms8k import GSM8k
from dataloaders.shakespeare import Shakespeare
from dataloaders.sharegpt import ShareGPT
from dataloaders.stemp import STemp


DATASETS: Dict[str, Type[AbstractDataset]] = {
    "gsm8k": GSM8k,
    "stemp": STemp,
    "csqa": CSQA,
    "sharegpt": ShareGPT,
    "aqua": AQUA,  # Uncomment if AQUA dataset is implemented
    "shakespeare": Shakespeare,
    # Add more datasets here as needed
}
