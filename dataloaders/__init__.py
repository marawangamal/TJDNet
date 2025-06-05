from dataloaders.base import AbstractDataset
from dataloaders.csqa import CSQA
from dataloaders.gms8k import GSM8k
from dataloaders.sharegpt import ShareGPT
from dataloaders.stemp import STemp

DATASETS: dict[str, type[AbstractDataset]] = {
    "gsm8k": GSM8k,
    "stemp": STemp,
    "csqa": CSQA,
    "sharegpt": ShareGPT,
}
