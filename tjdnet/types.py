from typing import Literal

PositivityFuncType = Literal[
    "relu", "leaky_relu", "sq", "abs", "exp", "safe_exp", "sigmoid", "none"
]

ModelHeadType = Literal["stp", "cp", "cpme", "cp_cond", "cp_condl", "multihead"]
