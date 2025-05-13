import torch
from tjdnet.distributions._tjdist import BaseDistConfig
from tjdnet.distributions.mps import MPSDist


class UMPSDist(MPSDist):
    def __init__(self, config: BaseDistConfig, **kwargs):
        config.param_net.out_dim_encoder = config.rank * config.rank
        config.param_net.out_dim_decoder = config.vocab_size
        super().__init__(config, bypass_config=True, **kwargs)

    def forward(self, last_hidden_state: torch.Tensor, **kwargs):
        return self.param_func(last_hidden_state)  # (B, T, RR, V)

    def get_mps_params(
        self,
        last_hidden_state: torch.Tensor,
        use_cache: bool = False,
        save_cache: bool = False,
    ):
        """Get both trainable and fixed parameters from the last hidden state.

        Args:
            last_hidden_state (torch.Tensor): Last hidden state of the transformer of shape (B, T, D)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                A tuple containing:
                - torch.Tensor: Alpha of shape (B, T, R)
                - torch.Tensor: Core of shape (B, T, HRVR)
                - torch.Tensor: Beta of shape (B, T, R)
        """
        batch_size, seq_len, _ = last_hidden_state.size()
        core = self._get_params_from_cache(
            last_hidden_state, use_cache, save_cache
        )  # (B, T, RR, V)
        alpha = (self.alpha.reshape(1, 1, self.rank).repeat(batch_size, seq_len, 1)).to(
            last_hidden_state.device
        )  # (B, T, R)
        beta = (self.beta.reshape(1, 1, self.rank).repeat(batch_size, seq_len, 1)).to(
            last_hidden_state.device
        )  # (B, T, R)
        return (
            alpha,
            core.reshape(
                batch_size,
                seq_len,
                1,  # No horizon dim
                self.rank,
                self.rank,
                self.vocab_size,
            )
            .permute(0, 1, 2, 3, 5, 4)
            .expand(-1, -1, self.dist_config.horizon, -1, -1, -1),  # (B, T, H, R, V, R)
            beta,
        )
