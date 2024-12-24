from typing import List, Tuple
from git import Optional
import torch
import torch.autograd.profiler as profiler

from distributions._base import BaseDistConfig, BaseDistribution
from tensorops.cp import (
    sample_from_cp_tensor,
    select_from_cp_tensor,
    select_margin_cp_tensor,
    sum_cp_tensor,
)


class CPDist(BaseDistribution):
    def __init__(self, config: BaseDistConfig, **kwargs):
        """CP Distribution

        Args:
            n_embd (int): Embedding dimension
            vocab_size (int): Vocabulary size
            rank (int): Rank of the CP decomposition
            horizon (int): Horizon of the model (Number of tokens to predict)
        """
        config.param_net.out_dim = config.rank * config.horizon * config.vocab_size
        super().__init__(config)

    def _get_params(
        self, last_hidden_state: torch.Tensor, horizon: Optional[int] = None, **kwargs
    ):
        batch_size, seq_len, _ = last_hidden_state.size()
        params = self.param_func(last_hidden_state)
        params_reshaped = params.reshape(
            batch_size, seq_len, self.rank, self.horizon, self.vocab_size
        )
        if horizon is not None:
            return params_reshaped[:, :, :, :horizon, :]  # (B, T, R, H, V)
        return params_reshaped  # (B, T, R, H*, V)  // H* is model level horizon

    def generate(
        self, last_hidden_state: torch.Tensor, horizon: Optional[int] = None, **kwargs
    ) -> torch.Tensor:
        """Generate sequences given an input tensor.

        Args:
            last_hidden_state (torch.Tensor): Hidden states of the transformer of shape (B, T, D)
            input_ids (torch.Tensor): Previous tokens of shape (B, T)
        """
        # Cannot generate sequences longer than `horizon`
        batch_size, seq_len, _ = last_hidden_state.size()
        assert batch_size == 1, "Only batch size 1 is supported"
        horizon = self._get_horizon(horizon)
        # print(f"Generating {horizon} tokens")
        params = self._get_params(
            last_hidden_state[:, -1:, :],
            horizon,
        )  # (B, 1, R, H, V) we only need the Tth hidden state

        # OPTION 1: Explicitly materialize the CP tensor
        # p_tilde = materialize_cp_tensor(
        #     # (B, 1, R, H, V) => (B, H, V, R)
        #     params.reshape(
        #         -1,
        #         self.rank,
        #         horizon,
        #         self.vocab_size,
        #     )
        # )  # (B, V, V, ..., V) `horizon` times
        # return torch.stack(
        #     [sample_from_tensor_dist(p_tilde_b, num_samples=1) for p_tilde_b in p_tilde]
        # ).reshape(
        #     batch_size, horizon
        # )  # (B, H)

        # OPTION 2: Sample directly using CP representation
        return torch.stack(
            [
                sample_from_cp_tensor(
                    params.reshape(
                        self.rank,
                        horizon,
                        self.vocab_size,
                    )
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
            use_cache (bool, optional): Whether to use cached values. Defaults to False.
        """
        params = self._get_params_from_cache(
            hidden_state.reshape(1, 1, -1), use_cache, save_cache
        )  # (1, 1, R, H, V)
        return select_margin_cp_tensor(
            cp_params=params.reshape(self.rank, self.horizon, self.vocab_size),
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
            Tuple[torch.Tensor, List[torch.Tensor]]: Evaluation of the distribution at the points of Shape (B, H) and scale_tensors (empty list)
        """
        # Get indexed distribution
        batch_size, seq_len, _ = last_hidden_state.size()
        horizon = points.size(-1)
        params = self._get_params(last_hidden_state, horizon)  # (B, T, R, H, V)
        # (B, T, R, H, V) => (B, T)
        with profiler.record_function("select_from_cp_tensor"):
            p_tilde = select_from_cp_tensor(
                params.reshape(
                    batch_size * seq_len, self.rank, horizon, self.vocab_size
                ),
                points.reshape(batch_size * seq_len, horizon),
            )
            return p_tilde.reshape(batch_size, seq_len), []  # (B,T)

    def get_norm_consts(
        self, last_hidden_state: torch.Tensor, horizon: Optional[int] = None, **kwargs
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Get the normalization constants for the BT distributions.

        Args:
            last_hidden_state (torch.Tensor): Hidden states of the transformer of shape (B, T, D)

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: Norm constants and scale tensors
        """
        batch_size, seq_len, _ = last_hidden_state.size()
        horizon = self._get_horizon(horizon)
        # Get indexed distribution
        params = self._get_params(last_hidden_state, horizon)  # (B, T, R, H, V)
        with profiler.record_function("normalize_cp_tensor"):
            norm_consts = sum_cp_tensor(
                cp_params=params.reshape(
                    batch_size * seq_len, self.rank, horizon, self.vocab_size
                ),
            )
            return (
                norm_consts.reshape(batch_size, seq_len),  # (B, T)
                [],
            )

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
        # Get indexed distribution
        batch_size, seq_len, _ = last_hidden_state.size()
        horizon = points.size(-1)
        params = self._get_params(last_hidden_state, horizon)  # (B, T, R, H, V)
        p_tilde = select_from_cp_tensor(
            params.reshape(batch_size * seq_len, self.rank, horizon, self.vocab_size),
            points.reshape(batch_size * seq_len, horizon),
        )
        norm_consts = sum_cp_tensor(
            cp_params=params.reshape(
                batch_size * seq_len, self.rank, horizon, self.vocab_size
            ),
        )
        return (
            p_tilde.reshape(batch_size, seq_len),
            [],
            norm_consts.reshape(batch_size, seq_len),
            [],
        )
