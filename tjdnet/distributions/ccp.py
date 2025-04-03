from typing import List, Tuple
from git import Optional
import torch
import torch.autograd.profiler as profiler

from tjdnet.distributions._base import BaseDistConfig, BaseDistribution
from tjdnet.tensorops.ccp import select_margin_ccp_tensor_batched
from tjdnet.tensorops.cp import select_margin_cp_tensor_batched, sum_cp_tensor
from tjdnet.utils import sample_topk


class CCPDist(BaseDistribution):
    def __init__(self, config: BaseDistConfig, **kwargs):
        """Compressed CP Distribution

        Args:
            config (BaseDistConfig): Configuration object.

        """
        config.param_net.out_dim = (
            config.rank * config.horizon * config.vocab_size_compr
        )
        super().__init__(config)
        self.unembed = torch.nn.Parameter(
            torch.randn(
                config.vocab_size_compr,
                config.vocab_size,
                dtype=torch.float32,
            )
        )

    def _get_params(
        self, last_hidden_state: torch.Tensor, horizon: Optional[int] = None, **kwargs
    ):
        batch_size, seq_len, _ = last_hidden_state.size()
        params = self.param_func(last_hidden_state)
        params_reshaped = params.reshape(
            batch_size, seq_len, self.rank, self.horizon, self.vocab_size_compr
        )
        if horizon is not None:
            return params_reshaped[:, :, :, :horizon, :]  # (B, T, R, H, Vc)
        return params_reshaped  # (B, T, R, H', Vc)

    def _get_ccp_params(
        self, last_hidden_state: torch.Tensor, horizon: Optional[int] = None, **kwargs
    ):
        cp_params = self._get_params(last_hidden_state)  # (B, T, R, H', Vc)
        return cp_params, torch.exp(self.unembed)  # (B, T, R, H', Vc), (Vc, V)

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
        cp_params, cp_decode = self._get_ccp_params(hidden_state[:, -1:, :])
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
            p_ops_tilde, _ = select_margin_ccp_tensor_batched(
                cp_params=cp_params.squeeze(1),  # (B, 1, R, H, V) => (B, R, H, V)
                cp_decode=cp_decode,  # (Vc, V)
                ops=ops_tensor,
            )  # (B, V), (B, T)
            if do_sample:
                next_token = sample_topk(p_ops_tilde, top_k, num_samples=1)
            else:  # greedy sampling
                next_token = sample_topk(p_ops_tilde, 1, num_samples=1)
            y_hat = torch.cat([y_hat, next_token], dim=1)
        return y_hat  # (B, H)

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
        cp_params, cp_decode = self._get_ccp_params(
            last_hidden_state, horizon=horizon
        )  # (B, T, R, H, Vc), (Vc, V)
        p_tilde, _ = select_margin_ccp_tensor_batched(
            cp_params=cp_params.reshape(
                batch_size * seq_len, self.rank, horizon, self.vocab_size_compr
            ),
            cp_decode=cp_decode,
            ops=points.reshape(batch_size * seq_len, horizon),
        )  # (BT,), (BT, T)
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
        # params = self._get_params(last_hidden_state, horizon)  # (B, T, R, H, V)
        cp_params, cp_decode = self._get_ccp_params(
            last_hidden_state, horizon=horizon
        )  # (B, T, R, H, Vc), (Vc, V)
        with profiler.record_function("normalize_cp_tensor"):
            # norm_consts = sum_cp_tensor(
            #     cp_params=params.reshape(
            #         batch_size * seq_len, self.rank, horizon, self.vocab_size
            #     ),
            # )
            norm_consts, _ = select_margin_ccp_tensor_batched(
                cp_params=cp_params.reshape(
                    batch_size * seq_len, self.rank, horizon, self.vocab_size_compr
                ),
                cp_decode=cp_decode,
                ops=torch.full(
                    (batch_size * seq_len, horizon),
                    -2,
                    dtype=torch.long,
                    device=last_hidden_state.device,
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
        # params = self._get_params(last_hidden_state, horizon)  # (B, T, R, H, V)
        # p_tilde = select_from_cp_tensor(
        #     params.reshape(batch_size * seq_len, self.rank, horizon, self.vocab_size),
        #     points.reshape(batch_size * seq_len, horizon),
        # )
        cp_params, cp_decode = self._get_ccp_params(
            last_hidden_state, horizon
        )  # (B, T, R, H, Vc), (Vc, V)
        p_tilde, _ = select_margin_ccp_tensor_batched(
            cp_params=cp_params.reshape(
                batch_size * seq_len, self.rank, horizon, self.vocab_size_compr
            ),
            cp_decode=cp_decode,
            ops=points.reshape(batch_size * seq_len, horizon),
        )  # (BT,), (BT, T)
        norm_consts, _ = select_margin_ccp_tensor_batched(
            cp_params=cp_params.reshape(
                batch_size * seq_len, self.rank, horizon, self.vocab_size_compr
            ),
            cp_decode=cp_decode,
            ops=torch.full(
                (batch_size * seq_len, horizon),
                -2,
                dtype=torch.long,
                device=last_hidden_state.device,
            ),
        )
        return (
            p_tilde.reshape(batch_size, seq_len),
            [],
            norm_consts.reshape(batch_size, seq_len),
            [],
        )
