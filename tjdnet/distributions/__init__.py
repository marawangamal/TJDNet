# tjdnet/distributions/__init__.py
from typing import Dict, Type
from tjdnet.distributions._tjdist import TJDist
from tjdnet.distributions.stp import STPDist
from tjdnet.distributions.cp import CPDist
from tjdnet.distributions.cpb import CPBDist
from tjdnet.distributions.cpe import CPEffDist


TJD_DISTS: Dict[str, Type[TJDist]] = {
    "stp": STPDist,
    "cp": CPDist,
    "cpb": CPBDist,
    "cpe": CPEffDist,
}
