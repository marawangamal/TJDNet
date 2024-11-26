from typing import List, Tuple
import torch
import torch.autograd.profiler as profiler

from distributions._base import BaseDistribution
from utils.tensorops.common import sample_from_tensor_dist
from utils.tensorops.mps import (
    select_from_umps_tensor,
    umps_materialize_batched,
    umps_select_marginalize_batched,
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
        super().__init__()
        self.rank = rank
        self.vocab_size = vocab_size
        self.horizon = horizon
        self.positivity_func: torch.nn.Module = {
            "sq": lambda x: x**2,
            "abs": lambda x: torch.abs(x),
            "exp": torch.exp,
        }[positivity_func]
        self.tensor_train_size = rank + (rank * vocab_size * rank) + rank
        self.param_func = torch.nn.Linear(n_embd, self.tensor_train_size)

    def _get_pos_params(self, last_hidden_state: torch.Tensor):
        params = self.positivity_func(
            self.param_func(last_hidden_state)
        )  # (B, T, R*H*V)

        alpha, core, beta = torch.split(
            params,
            [self.rank, self.rank * self.vocab_size * self.rank, self.rank],
            dim=-1,
        )  # (B, T, R), (B, T, R*V*R), (B, T, R)
        return alpha, core, beta

    def generate(self, last_hidden_state: torch.Tensor):
        """Generate sequences given an input tensor.

        Args:
            input_ids (torch.Tensor): Previous tokens of shape (B, T)
            horizon (int): Horizon of the generation (Must be <= Horizon of the model)
        """
        # Cannot generate sequences longer than `horizon`
        alpha, core, beta = self._get_pos_params(
            last_hidden_state[:, -1:, :]
        )  # (B, 1, V*R*H) we only need the Tth hidden state
        p_tilde = umps_materialize_batched(
            alpha=alpha.reshape(-1, self.rank),
            beta=beta.reshape(-1, self.rank),
            core=core.reshape(-1, self.rank, self.vocab_size, self.rank),
            n_core_repititions=self.horizon,
        )  # (B, V, V, ..., V)  `horizon` times
        return torch.stack(
            [sample_from_tensor_dist(p_tilde_b, 1) for p_tilde_b in p_tilde]
        )  # (B, H)

    def evaluate_at_points(
        self,
        last_hidden_state: torch.Tensor,
        points: torch.Tensor,
        is_normalized=False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Evaluate the distribution at the given points.

        Args:
            last_hidden_state (torch.Tensor): Hidden states of the transformer of shape (B, T, D)
            points (torch.Tensor): Points to evaluate the distribution. Shape (B, T, H)
            is_normalized (bool, optional): Whether the points are normalized. Defaults to False.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: Evaluation of the distribution at the points of Shape (B*H) and scale_tensors (empty list)
        """
        batch_size, seq_len, _ = last_hidden_state.shape
        alpha, core, beta = self._get_pos_params(last_hidden_state)
        # (B, T, R*H*V) => (B, T)
        with profiler.record_function("select_from_mps_tensor"):
            p_tilde, scale_factors = select_from_umps_tensor(
                alpha=alpha.reshape(batch_size * seq_len, self.rank),
                beta=beta.reshape(batch_size * seq_len, self.rank),
                core=core.reshape(
                    batch_size * seq_len, self.rank, self.vocab_size, self.rank
                ),
                indices=points.reshape(batch_size * seq_len, -1),
            )  # (batch_size, n_vocab)
            return p_tilde, scale_factors

    def get_norm_consts(
        self, last_hidden_state: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Get the normalization constants for the BT distributions.

        Args:
            last_hidden_state (torch.Tensor): Hidden states of the transformer of shape (B, T, D)

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: Norm constants and scale tensors
        """
        alpha, core, beta = self._get_pos_params(last_hidden_state)
        batch_size, seq_len, _ = last_hidden_state.shape
        z, scale_factors = umps_select_marginalize_batched(
            alpha=alpha.reshape(batch_size * seq_len, self.rank),
            beta=beta.reshape(batch_size * seq_len, self.rank),
            core=core.reshape(
                batch_size * seq_len, self.rank, self.vocab_size, self.rank
            ),
            operation_map=torch.ones(
                (batch_size * seq_len, self.horizon), device=alpha.device
            )
            * -1,
        )
        return z, scale_factors
