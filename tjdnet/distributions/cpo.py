import torch

from tjdnet.distributions._tjdist import BaseDistConfig
from tjdnet.distributions.cp import CPDist


class CPODist(CPDist):
    def __init__(self, config: BaseDistConfig, **kwargs):
        """CP-oslodets Distribution

        Note:
            This distribution is a variant of the CP distribution that alings with
            https://arxiv.org/pdf/2410.17765

        Args:
            n_embd (int): Embedding dimension
            vocab_size (int): Vocabulary size
            rank (int): Rank of the CP decomposition
            horizon (int): Horizon of the model (Number of tokens to predict)
        """
        config.param_net.out_dim_encoder = (
            config.rank * config.horizon * config.vocab_size
        )
        config.param_net.out_dim_decoder = 1
        config.param_net.hidden_dim = config.vocab_size
        config.param_net.use_decoder = False
        super().__init__(config, bypass_config=True, **kwargs)

    def get_params(self, x: torch.Tensor, **kwargs):
        B = x.size(0)
        params = self.param_func(x)  # (B, R * H, V)
        params_reshaped = params.reshape(B, self.rank, self.horizon, self.vocab_size)
        return params_reshaped  # (B, R, H, V)  // H* is model level horizon
