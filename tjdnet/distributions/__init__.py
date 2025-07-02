# tjdnet/distributions/__init__.py
from typing import Dict, Type
from tjdnet.distributions._base import AbstractDist
from tjdnet.distributions.stp import STPDist
from tjdnet.distributions.cp import CPDist
from tjdnet.distributions.cpb import CPCond
from tjdnet.distributions.cpc import CPCondl
from tjdnet.distributions.cpe import CPME
from tjdnet.distributions.multihead import MultiHeadDist


TJD_DISTS: Dict[str, Type[AbstractDist]] = {
    "stp": STPDist,
    "cp": CPDist,
    "cpb": CPCond,
    "cpc": CPCondl,
    "cpe": CPME,
    "multihead": MultiHeadDist,
}
