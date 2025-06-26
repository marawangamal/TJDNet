# tjdnet/distributions/__init__.py
from typing import Dict, Type
from tjdnet.distributions._base import AbstractDist
from tjdnet.distributions.stp import STPDist
from tjdnet.distributions.cp import CPDist
from tjdnet.distributions.cpb import CPBDist
from tjdnet.distributions.cpc import CPCDist
from tjdnet.distributions.cpe import CPEDist
from tjdnet.distributions.multihead import MultiHeadDist


TJD_DISTS: Dict[str, Type[AbstractDist]] = {
    "stp": STPDist,
    "cp": CPDist,
    "cpb": CPBDist,
    "cpc": CPCDist,
    "cpe": CPEDist,
    "multihead": MultiHeadDist,
}
