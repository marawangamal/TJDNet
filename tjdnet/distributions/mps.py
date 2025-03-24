from typing import List, Optional, Tuple
import torch

from tjdnet.distributions._base import BaseDistConfig, BaseDistribution
from tjdnet.tensorops.mps import (
    select_from_mps_tensor,
    select_margin_mps_tensor,
    sum_mps_tensor,
)
from tjdnet.utils import sample_topk


# TODO: try one-hot instead of ones for alpha and beta
class MPSDist(BaseDistribution):
    def __init__(self, config: BaseDistConfig, **kwargs):
        config.param_net.out_dim = config.horizon * (
            config.rank * config.vocab_size * config.rank
        )
        super().__init__(config)
        self.alpha = torch.ones(config.rank) * 0.1
        self.beta = torch.ones(config.rank) * 0.1

    def _get_params(self, last_hidden_state: torch.Tensor, **kwargs):
        return self.param_func(last_hidden_state)  # (B, T, HRVR)

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
        core = self._get_params_from_cache(last_hidden_state, use_cache, save_cache)
        alpha = (self.alpha.reshape(1, 1, self.rank).repeat(batch_size, seq_len, 1)).to(
            last_hidden_state.device
        )  # (B, T, R)
        beta = (self.beta.reshape(1, 1, self.rank).repeat(batch_size, seq_len, 1)).to(
            last_hidden_state.device
        )  # (B, T, R)
        return alpha, core, beta

    def sample(
        self,
        hidden_state: torch.Tensor,
        horizon: Optional[int] = None,
        do_sample: bool = False,
        top_k: int = 200,
        **kwargs,
    ) -> torch.Tensor:
        batch_size = hidden_state.size(0)
        assert batch_size == 1, "Batch size must be 1 for generation"
        horizon = self._get_horizon(horizon)
        dvc = hidden_state.device
        y_hat = torch.empty(batch_size, 0, device=dvc, dtype=torch.long)
        alpha, core, beta = self.get_mps_params(
            hidden_state[:, -1:, :],
        )  # (B, 1, R), (B, 1, HRVR), (B, 1, R)
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
            p_ops_tilde, _ = select_margin_mps_tensor(
                alpha=alpha.reshape(self.rank),
                beta=beta.reshape(self.rank),
                core=core.reshape(
                    self.horizon,
                    self.rank,
                    self.vocab_size,
                    self.rank,
                )[:horizon],
                ops=ops_tensor.reshape(-1),
            )  # (V,), (T,)
            if do_sample:
                next_token = sample_topk(
                    p_ops_tilde.reshape(1, -1), top_k, num_samples=1
                )
            else:  # greedy sampling
                next_token = sample_topk(p_ops_tilde.reshape(1, -1), 1, num_samples=1)
            y_hat = torch.cat([y_hat, next_token], dim=1)
        return y_hat  # (B, H)

    def get_dist(
        self,
        hidden_state: torch.Tensor,
        ops: torch.Tensor,
        use_cache: bool = False,
        save_cache: bool = False,
    ):
        """Get distribution specified by ops.

        Args:
            hidden_state (torch.Tensor): Last hidden state of the transformer of shape (D)
            ops (torch.Tensor): Operation codes of shape (T,) specifying:
                -2: marginalize mode (sum reduction)
                -1: keep mode as free index
                [0,V): select index v in mode
        """
        alpha, core, beta = self.get_mps_params(
            hidden_state.reshape(1, 1, -1),
            use_cache=use_cache,
            save_cache=save_cache,
        )  # (1, 1, R), (1, 1, H, R, V, R), (1, 1, R)
        return select_margin_mps_tensor(
            alpha=alpha.reshape(self.rank),
            beta=beta.reshape(self.rank),
            core=core.reshape(
                self.horizon,
                self.rank,
                self.vocab_size,
                self.rank,
            ),
            ops=ops,
        )

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
        alpha, core, beta = self.get_mps_params(
            last_hidden_state,
        )  # (B, T, R), (B, T, H, R, V, R), (B, T, R)
        p_tilde, scale_factors = select_from_mps_tensor(
            alpha=alpha.reshape(batch_size * seq_len, self.rank),
            beta=beta.reshape(batch_size * seq_len, self.rank),
            core=core.reshape(
                batch_size * seq_len,
                self.horizon,
                self.rank,
                self.vocab_size,
                self.rank,
            )[:, :horizon],
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
        alpha, core, beta = self.get_mps_params(
            last_hidden_state
        )  # (B, T, R), (B, T, H, R, V, R), (B, T, R)
        batch_size, seq_len, _ = last_hidden_state.shape
        z, scale_factors = sum_mps_tensor(
            alpha=alpha.reshape(batch_size * seq_len, self.rank),
            beta=beta.reshape(batch_size * seq_len, self.rank),
            core=core.reshape(
                batch_size * seq_len,
                self.horizon,
                self.rank,
                self.vocab_size,
                self.rank,
            )[:, :horizon],
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
        alpha, core, beta = self.get_mps_params(
            last_hidden_state,
        )  # (B, T, R), (B, T, H, R, V, R), (B, T, R)
        p_tilde, p_scale_factors = select_from_mps_tensor(
            alpha=alpha.reshape(batch_size * seq_len, self.rank),
            beta=beta.reshape(batch_size * seq_len, self.rank),
            core=core.reshape(
                batch_size * seq_len,
                self.horizon,
                self.rank,
                self.vocab_size,
                self.rank,
            )[:, :horizon],
            indices=points.reshape(batch_size * seq_len, -1),
        )  # (batch_size, n_vocab)

        z, z_scale_factors = sum_mps_tensor(
            alpha=alpha.reshape(batch_size * seq_len, self.rank),
            beta=beta.reshape(batch_size * seq_len, self.rank),
            core=core.reshape(
                batch_size * seq_len,
                self.horizon,
                self.rank,
                self.vocab_size,
                self.rank,
            )[:, :horizon],
        )
        return (
            p_tilde.reshape(batch_size, seq_len),
            [s.reshape(batch_size, seq_len) for s in p_scale_factors],
            z.reshape(batch_size, seq_len),
            [s.reshape(batch_size, seq_len) for s in z_scale_factors],
        )
