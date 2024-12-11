from typing import List, Tuple
import torch
import torch.autograd.profiler as profiler

from distributions._base import BaseDistribution
from tensorops.common import sample_from_tensor_dist


class BaseDist(BaseDistribution[torch.Tensor]):
    def __init__(
        self,
        n_embd: int,
        vocab_size,
        positivity_func: str = "exp",
        horizon: int = 1,
        **kwargs,
    ):
        """Basic 1D entropy distribution

        Args:
            n_embd (int): Embedding dimension
            vocab_size (int): Vocabulary size
            rank (int): Rank of the CP decomposition
            horizon (int): Horizon of the model (Number of tokens to predict)
        """
        super().__init__(horizon=1)
        assert horizon == 1, "Only horizon=1 is supported for now"
        self.param_func = torch.nn.Linear(n_embd, vocab_size)
        self.vocab_size = vocab_size
        self.rank = 1
        self.horizon = horizon
        self.positivity_func: torch.nn.Module = {
            "sq": lambda x: x**2,
            "abs": lambda x: torch.abs(x),
            "exp": torch.exp,
        }[positivity_func]

    def get_params(self, last_hidden_state: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.positivity_func(self.param_func(last_hidden_state))

    def get_dist(
        self,
        hidden_state: torch.Tensor,
        ops: torch.Tensor,
        **kwargs,
    ):
        """Get distribution specified by ops."""
        assert ops.size(-1) == 1, "Only 1D points are supported"
        p_tilde = self.get_params(hidden_state)  # (B, 1, V)
        p_tilde = p_tilde.reshape(-1)  # (B, V)
        return p_tilde, []

    def generate(
        self, last_hidden_state: torch.Tensor, horizon: int, **kwargs
    ) -> torch.Tensor:
        """Generate sequences given an input tensor.

        Args:
            input_ids (torch.Tensor): Previous tokens of shape (B, T)
        """
        # Cannot generate sequences longer than `horizon`
        p_tilde = self.get_params(
            last_hidden_state[:, -1:, :]
        )  # (B, 1, V) we only need the Tth hidden state
        p_tilde = p_tilde.reshape(-1, self.vocab_size)  # (B, V)
        return torch.stack(
            [sample_from_tensor_dist(p_tilde_b, 1) for p_tilde_b in p_tilde]
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
            points (torch.Tensor): Points to evaluate the distribution. Shape (B, T, 1)
            is_normalized (bool, optional): Whether the points are normalized. Defaults to False.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: Evaluation of the distribution at the points of Shape (B, 1) and scale_tensors (empty list)
        """
        # Get indexed distribution
        assert points.size(-1) == 1, "Only 1D points are supported"
        p_tilde = self.get_params(last_hidden_state)  # (B, T, V)
        batch_size, seq_len, _ = last_hidden_state.size()

        # (B, T, R*H*V) => (B, T)
        with profiler.record_function("select_from_base_tensor"):
            p_tilde_select = torch.gather(
                p_tilde.reshape(batch_size * seq_len, self.vocab_size),
                dim=1,
                index=points.reshape(batch_size * seq_len, self.horizon),
            )  # (B*T, 1)
            return p_tilde_select.reshape(batch_size, seq_len), []  # (B*T)

    def get_norm_consts(
        self, last_hidden_state: torch.Tensor, horizon: int, **kwargs
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Get the normalization constants for the BT distributions.

        Args:
            last_hidden_state (torch.Tensor): Hidden states of the transformer of shape (B, T, D)

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: Norm constants and scale tensors
        """
        # Get indexed distribution
        batch_size, seq_len, _ = last_hidden_state.size()
        p_tilde = self.positivity_func(self.param_func(last_hidden_state))  # (B, T, V)
        norm_consts = torch.sum(p_tilde, dim=-1).reshape(-1)  # (B*T)
        return (
            norm_consts.reshape(batch_size, seq_len),
            [],
        )
