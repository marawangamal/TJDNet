# tjdnet/distributions/__init__.py
from typing import Dict, Type
from tjdnet.distributions._tjdist import TJDist
from tjdnet.distributions.base import BaseDist
from tjdnet.distributions.cp import CPDist
from tjdnet.distributions.cpo import CPODist


TJD_DISTS: Dict[str, Type[TJDist]] = {
    "base": BaseDist,
    "cp": CPDist,
    "cpo": CPODist,
    # "cpb": CPBDist,
    # "ucp": UCPDist,
    # "mps": MPSDist,
    # "umps": UMPSDist,
}
