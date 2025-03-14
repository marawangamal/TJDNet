from typing import List, Optional, Tuple
import torch

from tjdnet.distributions._base import BaseDistConfig, BaseDistribution
from tjdnet.tensorops.umps import (
    sample_from_umps_tensor,
    select_from_umps_tensor,
    select_margin_umps_tensor,
    sum_umps_tensorV2,
)
from tjdnet.utils import sample_topk


class UMPSDist(BaseDistribution):
    def __init__(self, config: BaseDistConfig, **kwargs):
        config.param_net.out_dim = (
            config.rank + (config.rank * config.vocab_size * config.rank) + config.rank
        )
        super().__init__(config)

    def _get_params(self, last_hidden_state: torch.Tensor, **kwargs):
        params_tilde = self.param_func(last_hidden_state)
        return self.positivity_func(params_tilde)  # (B, T, R + R + RVR)

    def get_umps_params(
        self,
        last_hidden_state: torch.Tensor,
        use_cache: bool = False,
        save_cache: bool = False,
    ):
        batch_size, seq_len, _ = last_hidden_state.shape
        umps_params = self._get_params_from_cache(
            last_hidden_state, use_cache, save_cache
        )
        alpha = umps_params[:, :, : self.rank]
        beta = umps_params[:, :, self.rank : self.rank * 2]
        core = umps_params[:, :, self.rank * 2 :].reshape(
            batch_size, seq_len, self.rank, self.vocab_size, self.rank
        )
        return alpha, core, beta

    def evaluate_at_points(
        self,
        last_hidden_state: torch.Tensor,
        points: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Evaluate the distribution at the given points.

        Args:
            last_hidden_state (torch.Tensor): Hidden states of the transformer of shape (B, T, D)
            points (torch.Tensor): Points to evaluate the distribution. Shape (B, T, H)
            horizon (int, optional): Number of steps to consider. Defaults to model horizon.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: Evaluation of the distribution at the points of Shape (B*H) and scale_tensors (empty list)
        """
        batch_size, seq_len, _ = last_hidden_state.shape
        horizon = self._get_horizon(points.size(-1))
        alpha, core, beta = self.get_umps_params(
            last_hidden_state
        )  # (B, T, R), (B, T, H, R, V, R), (B, T, R)
        p_tilde, scale_factors = select_from_umps_tensor(
            alpha=alpha.reshape(batch_size * seq_len, self.rank),
            beta=beta.reshape(batch_size * seq_len, self.rank),
            core=core.reshape(
                batch_size * seq_len,
                self.rank,
                self.vocab_size,
                self.rank,
            ),
            indices=points.reshape(batch_size * seq_len, -1),
        )  # (batch_size, n_vocab)
        return p_tilde.reshape(batch_size, seq_len), [
            s.reshape(batch_size, seq_len) for s in scale_factors
        ]

    def get_norm_consts(
        self, last_hidden_state: torch.Tensor, horizon: int, **kwargs
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Get the normalization constants for the BT distributions.

        Args:
            last_hidden_state (torch.Tensor): Hidden states of the transformer of shape (B, T, D)

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: Norm constants and scale tensors
        """
        horizon = self._get_horizon(horizon)
        alpha, core, beta = self.get_umps_params(
            last_hidden_state
        )  # (B, T, R), (B, T, R, V, R), (B, T, R)
        batch_size, seq_len, _ = last_hidden_state.shape
        z, scale_factors = sum_umps_tensorV2(
            alpha=alpha.reshape(batch_size * seq_len, self.rank),
            beta=beta.reshape(batch_size * seq_len, self.rank),
            core=core.reshape(
                batch_size * seq_len, self.rank, self.vocab_size, self.rank
            ),
            n_core_repititions=horizon,
        )
        return z.reshape(batch_size, seq_len), [
            s.reshape(batch_size, seq_len) for s in scale_factors
        ]

    def evaluate_at_points_and_get_norm_consts(
        self,
        last_hidden_state: torch.Tensor,
        points: torch.Tensor,
        **kwargs,
    ):
        """Evaluate the distribution at the given points and get the normalization constants.

        Args:
            last_hidden_state (torch.Tensor): Hidden states of the transformer of shape (B, T, D)
            points (torch.Tensor): Points to evaluate the distribution. Shape (B, T, H)

        Returns:
            tuple:
                - torch.Tensor: Unormalized distribution `p_tilde` at the points of shape (B, T)
                - list: Scale factors for `p_tilde`
                - torch.Tensor: Normalization constants `z` of shape (B, T)
                - list: Scale factors for `z`
        """
        batch_size, seq_len, _ = last_hidden_state.shape
        horizon = self._get_horizon(points.size(-1))

        alpha, core, beta = self.get_umps_params(
            last_hidden_state
        )  # (B, T, R), (B, T, H, R, V, R), (B, T, R)
        p_tilde, p_scale_factors = select_from_umps_tensor(
            alpha=alpha.reshape(batch_size * seq_len, self.rank),
            beta=beta.reshape(batch_size * seq_len, self.rank),
            core=core.reshape(
                batch_size * seq_len,
                self.rank,
                self.vocab_size,
                self.rank,
            ),
            indices=points.reshape(batch_size * seq_len, -1),
        )  # (batch_size, n_vocab)

        z, z_scale_factors = sum_umps_tensorV2(
            alpha=alpha.reshape(batch_size * seq_len, self.rank),
            beta=beta.reshape(batch_size * seq_len, self.rank),
            core=core.reshape(
                batch_size * seq_len, self.rank, self.vocab_size, self.rank
            ),
            n_core_repititions=horizon,
        )
        return (
            p_tilde.reshape(batch_size, seq_len),
            [s.reshape(batch_size, seq_len) for s in p_scale_factors],
            z.reshape(batch_size, seq_len),
            [s.reshape(batch_size, seq_len) for s in z_scale_factors],
        )

    def sample(
        self,
        hidden_state: torch.Tensor,
        horizon: Optional[int] = None,
        do_sample: bool = False,
        top_k: int = 200,
        **kwargs,
    ) -> torch.Tensor:
        horizon = self._get_horizon(horizon)
        batch_size = hidden_state.size(0)
        dvc = hidden_state.device
        y_hat = torch.empty(batch_size, 0, device=dvc, dtype=torch.long)
        alpha, core, beta = self.get_umps_params(
            hidden_state[:, -1:, :]
        )  # (B, 1, R), (B, 1, R, V, R), (B, 1, R)
        for h in range(horizon):
            ops_tensor = torch.cat(
                (
                    y_hat,  # selection
                    -1  # free leg
                    * torch.ones(batch_size, 1, dtype=torch.long, device=dvc),
                    -2  # marginalization
                    * torch.ones(
                        batch_size, (horizon - h - 1), dtype=torch.long, device=dvc
                    ),
                ),
                dim=1,
            )  # (B, T)

            p_ops_tilde, _ = select_margin_umps_tensor_batched(
                alpha=alpha,
                beta=beta,
                core=core,
                ops=ops_tensor,
            )  # (B, V), (B, T)
            if do_sample:
                next_token = sample_topk(p_ops_tilde, top_k, num_samples=1)
            else:  # greedy sampling
                next_token = sample_topk(p_ops_tilde, 1, num_samples=1)
            y_hat = torch.cat([y_hat, next_token], dim=1)
        return y_hat  # (B, H)
