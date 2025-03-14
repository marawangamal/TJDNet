from typing import List, Tuple
import torch

from tjdnet.distributions._base import BaseDistConfig, BaseDistribution
from tjdnet.tensorops.mps import (
    sample_from_mps_tensorV1,
    select_from_mps_tensor,
    select_margin_mps_tensor,
    sum_mps_tensor,
)


# TODO: dont apply positivity function to alpha and beta and use 1hot instead of ones
class MPSDist(BaseDistribution):
    def __init__(self, config: BaseDistConfig, **kwargs):
        config.param_net.out_dim = config.horizon * (
            config.rank * config.vocab_size * config.rank
        )
        super().__init__(config)
        self.alpha = torch.ones(config.rank) * 0.1
        self.beta = torch.ones(config.rank) * 0.1

    def _get_params(self, last_hidden_state: torch.Tensor, **kwargs):
        """Get trainable parameters from the last hidden state.

        Args:
            last_hidden_state (torch.Tensor): Last hidden state of the transformer of shape (B, T, D)

        Returns:
            torch.Tensor: MPS parameters of shape (B, T, 2*R + H*R*V*R)
        """
        core = self.param_func_core(last_hidden_state)
        core = self.positivity_func(core)  # (B, T, HRVR)
        return core

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
            - torch.Tensor: Alpha of shape (B, T, R)
            - torch.Tensor: Core of shape (B, T, H, R, V, R)
            - torch.Tensor: Beta of shape (B, T, R)

        """
        batch_size, seq_len, _ = last_hidden_state.size()
        core = self._get_params_from_cache(last_hidden_state, use_cache, save_cache)
        alpha = (
            self.positivity_func(self.alpha)
            .reshape(1, 1, self.rank)
            .repeat(batch_size, seq_len, 1)
        ).to(
            last_hidden_state.device
        )  # (B, T, R)
        beta = (
            self.positivity_func(self.beta)
            .reshape(1, 1, self.rank)
            .repeat(batch_size, seq_len, 1)
        ).to(
            last_hidden_state.device
        )  # (B, T, R)
        return alpha, core, beta

    def generate(self, last_hidden_state: torch.Tensor, horizon: int, **kwargs):
        """Generate sequences given an input tensor.

        Args:
            input_ids (torch.Tensor): Previous tokens of shape (B, T)
            horizon (int): Horizon of the generation (Must be <= Horizon of the model)

        Returns:
            torch.Tensor: Generated sequences of shape (B, H)
        """
        # Cannot generate sequences longer than `horizon`
        horizon = self._get_horizon(horizon)
        batch_size, _, _ = last_hidden_state.size()
        assert batch_size == 1, "Batch size must be 1 for generation"
        alpha, core, beta = self.get_mps_params(
            last_hidden_state[:, -1:, :],
        )  # (1, 1, R), (1, 1, H, R, V, R), (1, 1, R)
        return torch.stack(
            [
                sample_from_mps_tensorV1(
                    alpha=alpha.reshape(self.rank),  # (B, T, R)
                    beta=beta.reshape(self.rank),  # (B, T, R)
                    core=core.reshape(
                        self.horizon,
                        self.rank,
                        self.vocab_size,
                        self.rank,
                    )[:horizon],
                )
            ]
        )  # (B, H)

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
