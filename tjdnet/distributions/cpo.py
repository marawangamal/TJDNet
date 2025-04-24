from git import Optional
import torch

from tjdnet.distributions._base import BaseDistConfig
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
        # config.param_net.out_dim = config.rank * config.vocab_size
        config.param_net.out_dim_encoder = (
            config.rank * config.horizon * config.vocab_size
        )
        config.param_net.out_dim_decoder = 1
        config.param_net.hidden_dim = 1
        super().__init__(config, bypass_config=True, **kwargs)

    def _get_params(
        self, last_hidden_state: torch.Tensor, horizon: Optional[int] = None, **kwargs
    ):
        batch_size, seq_len, _ = last_hidden_state.size()  # (B, T, D)
        params = self.param_func(last_hidden_state)  # (B, T, RHV, 1)
        params_reshaped = params.reshape(
            batch_size, seq_len, self.rank, self.horizon, self.vocab_size
        )
        if horizon is not None:
            return params_reshaped[:, :, :, :horizon, :]  # (B, T, R, H, V)
        return params_reshaped  # (B, T, R, H*, V)  // H* is model level horizon
