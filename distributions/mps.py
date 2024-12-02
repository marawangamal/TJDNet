from typing import List, Optional, Tuple
import torch
import torch.autograd.profiler as profiler

from distributions._base import BaseDistribution
from tensorops.common import sample_from_tensor_dist
from tensorops.mps import (
    sample_from_mps_tensor,
    select_from_mps_tensor,
    materialize_mps_tensor,
    sum_mps_tensor,
)


class MPSDist(BaseDistribution):
    def __init__(
        self,
        n_embd: int,
        vocab_size,
        rank: int,
        horizon: int,
        positivity_func: str = "exp",
    ):
        super().__init__(horizon)
        self.rank = rank
        self.vocab_size = vocab_size
        self.horizon = horizon
        self.positivity_func: torch.nn.Module = {
            "sq": lambda x: x**2,
            "abs": lambda x: torch.abs(x),
            "exp": torch.exp,
        }[positivity_func]
        self.tensor_train_size = rank + horizon * (rank * vocab_size * rank) + rank
        self.param_func = torch.nn.Linear(n_embd, self.tensor_train_size)

    def _get_pos_params(self, last_hidden_state: torch.Tensor):
        batch_size, seq_len, _ = last_hidden_state.shape
        params = self.positivity_func(
            self.param_func(last_hidden_state)
        )  # (B, T, R + R + HRVR)
        alpha = params[:, :, : self.rank]
        beta = params[:, :, self.rank : self.rank * 2]
        core = params[:, :, self.rank * 2 :].reshape(
            batch_size, seq_len, self.horizon, self.rank, self.vocab_size, self.rank
        )
        return (
            alpha,  # (B, T, R)
            core,  # (B, T, H, R, V, R)
            beta,  # (B, T, R)
        )

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
        alpha, core, beta = self._get_pos_params(
            last_hidden_state[:, -1:, :]
        )  # (B, 1, R), (B, 1, H, R, V, R), (B, 1, R)
        # OLD:
        # p_tilde = materialize_mps_tensor(
        #     alpha=alpha.reshape(batch_size * 1, self.rank),
        #     beta=beta.reshape(batch_size * 1, self.rank),
        #     core=core.reshape(
        #         batch_size * 1, self.horizon, self.rank, self.vocab_size, self.rank
        #     )[:, :horizon],
        # )  # (B, V, V, ..., V)  `horizon` times
        # return torch.stack(
        #     [sample_from_tensor_dist(p_tilde_b, 1) for p_tilde_b in p_tilde]
        # ).reshape(
        #     batch_size, -1
        # )  # (B, H)

        # NEW:
        return torch.stack(
            [
                sample_from_mps_tensor(
                    alpha=alpha.reshape(self.rank),
                    beta=beta.reshape(self.rank),
                    core=core.reshape(
                        self.horizon,
                        self.rank,
                        self.vocab_size,
                        self.rank,
                    )[:horizon],
                )
            ]
        )  # (B, H)

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
        alpha, core, beta = self._get_pos_params(
            last_hidden_state
        )  # (B, T, R), (B, T, H, R, V, R), (B, T, R)
        with profiler.record_function("select_from_mps_tensor"):
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
            return p_tilde, scale_factors

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
        alpha, core, beta = self._get_pos_params(
            last_hidden_state
        )  # (B, T, R), (B, T, H, R, V, R), (B, T, R)
        batch_size, seq_len, _ = last_hidden_state.shape
        with profiler.record_function("normalize_mps_tensor"):
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
            return z, scale_factors
