# tjdnet/distributions/__init__.py
from typing import Dict, Type
from tjdnet.distributions._base import BaseDistribution
from tjdnet.distributions.base import BaseDist
from tjdnet.distributions.cp import CPDist
from tjdnet.distributions.cpb import CPBDist
from tjdnet.distributions.mps import MPSDist
from tjdnet.distributions.ucp import UCPDist
from tjdnet.distributions.umps import UMPSDist


TJD_DISTS: Dict[str, Type[BaseDistribution]] = {
    "base": BaseDist,
    "cp": CPDist,
    "cpo": CPDist,
    "cpb": CPBDist,
    "ucp": UCPDist,
    "mps": MPSDist,
    "umps": UMPSDist,
}
