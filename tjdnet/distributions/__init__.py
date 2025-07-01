# tjdnet/distributions/__init__.py
from typing import Dict, Type
from tjdnet.distributions._base import AbstractDist
from tjdnet.distributions.stp import STPDist
from tjdnet.distributions.cp import CPDist
from tjdnet.distributions.cpsd import CPSDDist
from tjdnet.distributions.cpsd_cond import CPSDCondDist
from tjdnet.distributions.cpsd_condl import CPSDCondlDist
from tjdnet.distributions.multihead import MultiHeadDist


TJD_DISTS: Dict[str, Type[AbstractDist]] = {
    "stp": STPDist,  # Single token prediction
    "cp": CPDist,  # CP
    "cpsd": CPSDDist,  # CP, Shared decoder
    "cpsd_cond": CPSDCondDist,  # CP, Shared decoder, conditional
    "cpsd_condl": CPSDCondlDist,  # CP, Shared decoder, conditional log-space
    "multihead": MultiHeadDist,  # Multi-head (i.e., MoE model)
}
