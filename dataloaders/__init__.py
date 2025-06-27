from dataloaders.aqua import AQUA
from dataloaders.base import AbstractDataset
from dataloaders.csqa import CSQA
from dataloaders.gms8k import GSM8k
from dataloaders.reddit import Reddit
from dataloaders.shakespeare import Shakespeare
from dataloaders.sharegpt import ShareGPT
from dataloaders.sst2 import SST2
from dataloaders.stemp import STemp
from dataloaders.wikitext2 import WikiText2

DATASETS: dict[str, type[AbstractDataset]] = {
    "gsm8k": GSM8k,
    "stemp": STemp,
    "csqa": CSQA,
    "sharegpt": ShareGPT,
    "aqua": AQUA,  # Uncomment if AQUA dataset is implemented
    "shakespeare": Shakespeare,
    "wikitext2": WikiText2,
    "sst2": SST2,
    "reddit": Reddit,
    # Add more datasets here as needed
}
