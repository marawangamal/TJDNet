# tjdnet/distributions/__init__.py
from typing import Dict, Type
from tjdnet.distributions._base import AbstractDist
from tjdnet.distributions.cp_rmoe import CPRMoEDist
from tjdnet.distributions.dummy import DummyDist
from tjdnet.distributions.stp import STPDist
from tjdnet.distributions.cp import CPDist
from tjdnet.distributions.cp_cond import CPCond
from tjdnet.distributions.cp_condl import CPCondl
from tjdnet.distributions.cpme import CPME
from tjdnet.distributions.multihead import MultiHeadDist


TJD_DISTS: Dict[str, Type[AbstractDist]] = {
    "stp": STPDist,
    "cp": CPDist,
    "cp_cond": CPCond,
    "cp_condl": CPCondl,
    "cp_rmoe": CPRMoEDist,
    "cpme": CPME,
    "multihead": MultiHeadDist,
    "dummy": DummyDist,
}
