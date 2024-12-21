from typing import List, Tuple
import torch

from distributions._base import BaseDistribution
from tensorops.umps import (
    sample_from_umps_tensor,
    select_from_umps_tensor,
    select_margin_umps_tensor,
    sum_umps_tensorV2,
)


class UMPSDist(BaseDistribution):
    def __init__(
        self,
        n_embd: int,
        vocab_size,
        rank: int,
        horizon: int,
        positivity_func: str = "exp",
        hidden_dim: int = 256,
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
        self.tensor_train_size = rank + (rank * vocab_size * rank) + rank
        self.param_func = torch.nn.Sequential(
            torch.nn.Linear(n_embd, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, self.tensor_train_size),
        )

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
        alpha, core, beta = self.get_umps_params(
            hidden_state.reshape(1, 1, -1),
            use_cache=use_cache,
            save_cache=save_cache,
        )  # (1, 1, R), (1, 1, R, V, R), (1, 1, R)
        return select_margin_umps_tensor(
            alpha=alpha.reshape(self.rank),
            beta=beta.reshape(self.rank),
            core=core.reshape(
                self.rank,
                self.vocab_size,
                self.rank,
            ),
            ops=ops,
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
        alpha, core, beta = self.get_umps_params(
            last_hidden_state[:, -1:, :]
        )  # (B, 1, R), (B, 1, R, V, R), (B, 1, R)
        return torch.stack(
            [
                sample_from_umps_tensor(
                    alpha=alpha.reshape(self.rank),
                    beta=beta.reshape(self.rank),
                    core=core.reshape(self.rank, self.vocab_size, self.rank),
                    horizon=horizon,
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
