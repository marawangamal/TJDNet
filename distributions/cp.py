from typing import List, Tuple
import torch
import torch.autograd.profiler as profiler

from distributions.base import BaseDistribution
from utils.tensorops.common import sample_from_tens
from utils.tensorops.cp import (
    materialize_cp_tensor,
    select_from_cp_tensor,
    sum_cp_tensor,
)


class CPDist(BaseDistribution):
    def __init__(
        self,
        n_embd: int,
        vocab_size,
        rank: int,
        horizon: int,
        positivity_func: str = "exp",
    ):
        """CP Distribution

        Args:
            n_embd (int): Embedding dimension
            vocab_size (int): Vocabulary size
            rank (int): Rank of the CP decomposition
            horizon (int): Horizon of the model (Number of tokens to predict)
        """
        super().__init__()
        assert horizon == 2, "Only horizon=2 is supported for now"
        self.param_func = torch.nn.Linear(n_embd, rank * horizon * vocab_size)
        self.horizon = horizon
        self.vocab_size = vocab_size
        self.rank = rank
        self.positivity_func: torch.nn.Module = {
            "sq": lambda x: x**2,
            "abs": lambda x: torch.abs(x),
            "exp": torch.exp,
        }[positivity_func]

    def _get_pos_params(self, last_hidden_state: torch.Tensor):
        return self.positivity_func(self.param_func(last_hidden_state))

    def generate(self, last_hidden_state: torch.Tensor):
        """Generate sequences given an input tensor.

        Args:
            input_ids (torch.Tensor): Previous tokens of shape (B, T)
            horizon (int): Horizon of the generation (Must be <= Horizon of the model)
        """
        # Cannot generate sequences longer than `horizon`
        params = self._get_pos_params(
            last_hidden_state[:, -1:, :]
        )  # (B, 1, V*R*H) we only need the Tth hidden state
        p_tilde = materialize_cp_tensor(
            params.reshape(
                -1,
                self.rank,
                self.horizon,
                self.vocab_size,
            ).permute(0, 2, 3, 1)
            # params.reshape(-1, self.horizon, self.vocab_size, self.rank)
        )  # (B, V**H)
        p_tilde = p_tilde.reshape(
            -1, *([self.vocab_size] * self.horizon)
        )  # (B, V, V, ..., V)
        return sample_from_tens(p_tilde[0], 1)  # (B, H)

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
            Tuple[torch.Tensor, List[torch.Tensor]]: Evaluation of the distribution at the points of Shape (B, H) and scale_tensors (empty list)
        """
        # Get indexed distribution
        params = self._get_pos_params(last_hidden_state)  # (B, T, R*H*V)
        batch_size, seq_len, _ = last_hidden_state.size()

        # (B, T, R*H*V) => (B, T)
        with profiler.record_function("select_from_cp_tensor"):
            p_tilde = select_from_cp_tensor(
                params.reshape(
                    batch_size * seq_len, self.rank, self.horizon, self.vocab_size
                ),
                points.reshape(batch_size * seq_len, self.horizon),
            )
            return p_tilde, []  # (B*T)

    def get_norm_consts(
        self, last_hidden_state: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Get the normalization constants for the BT distributions.

        Args:
            last_hidden_state (torch.Tensor): Hidden states of the transformer of shape (B, T, D)

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: Norm constants and scale tensors
        """
        # Get indexed distribution
        params = self.positivity_func(
            self.param_func(last_hidden_state)
        )  # (B, T, R*H*V)
        batch_size, seq_len, _ = last_hidden_state.size()
        norm_consts = sum_cp_tensor(
            tensor=params.reshape(
                batch_size * seq_len, self.rank, self.horizon, self.vocab_size
            ),
        )
        return (
            norm_consts,  # (B*T)
            [],
        )
