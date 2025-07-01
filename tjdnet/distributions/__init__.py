# tjdnet/distributions/__init__.py
from typing import Dict, Type
from tjdnet.distributions._base import AbstractDist
from tjdnet.distributions.stp import STPDist
from tjdnet.distributions.cp import CPDist
from tjdnet.distributions.cpme import CPMEDist
from tjdnet.distributions.cp_cond import CPCondDist
from tjdnet.distributions.cp_condl import CPCondlDist
from tjdnet.distributions.multihead import MultiHeadDist


TJD_DISTS: Dict[str, Type[AbstractDist]] = {
    "stp": STPDist,  # Single token prediction
    "cp": CPDist,  # CP
    "cpme": CPMEDist,  # CP, Memory-efficient
    "cp_cond": CPCondDist,  # CP, conditional
    "cp_condl": CPCondlDist,  # CP, conditional log-space
    "multihead": MultiHeadDist,  # Multi-head (i.e., MoE model)
}
