from typing import List, Tuple
from git import Optional
import torch
import torch.autograd.profiler as profiler

from tjdnet.distributions._tjdist import BaseDistConfig, TJDist
from tjdnet.distributions.cp import CPDist
from tjdnet.tensorops.cp import select_margin_cp_tensor_batched, sum_cp_tensor
from tjdnet.utils import get_positional_encodings, sample_topk


# MAT SIZES
# CP:
# - tp_net: (d_in x rank x vocab_size x horizon) (e.g  768 x 32 x 50257 x 8)

# UCP
# - tp_net: (d_in x rank x vocab_size) (e.g  768 x 32 x 50257)
# - pos_encodings: (horizon x vocab_size)  (e.g 8 x 50257)


class UCPDist(CPDist):
    def __init__(self, config: BaseDistConfig, **kwargs):
        """CP Distribution

        Args:
            n_embd (int): Embedding dimension
            vocab_size (int): Vocabulary size
            rank (int): Rank of the CP decomposition
            horizon (int): Horizon of the model (Number of tokens to predict)
        """
        # config.param_net.out_dim = config.rank * config.vocab_size
        config.param_net.out_dim_encoder = config.rank
        config.param_net.out_dim_decoder = config.vocab_size
        super().__init__(config, bypass_config=True, **kwargs)

    def forward(
        self, last_hidden_state: torch.Tensor, horizon: Optional[int] = None, **kwargs
    ):
        batch_size, seq_len, _ = last_hidden_state.size()  # (B, T, D)
        params = self.param_func(last_hidden_state)  # (B, T, R, V)
        params_reshaped = params.reshape(
            batch_size, seq_len, self.rank, 1, self.vocab_size
        )

        # Add positional encoding across the horizon dimension
        pos_encodings = get_positional_encodings(
            seq_len=self.horizon, d_model=self.vocab_size, device=params.device
        )  # (H, V)
        pos_encodings = pos_encodings.reshape(
            1, 1, 1, self.horizon, self.vocab_size
        ).expand(batch_size, seq_len, self.rank, self.horizon, self.vocab_size)

        params_reshaped = params_reshaped + pos_encodings  # (B, T, R, H, V)

        if horizon is not None:
            return params_reshaped[:, :, :, :horizon, :]  # (B, T, R, H, V)

        return params_reshaped  # (B, T, R, H*, V)  // H* is model level horizon
