from typing import Optional
import torch

from tjdnet.distributions._tjdist import (
    BaseDistConfig,
    BaseDistFromLinearConfig,
    TJDist,
)
from tjdnet.distributions.tpnet import TensorParamNetConfig
from tjdnet.tensorops.mps import select_margin_mps_tensor_batched
from tjdnet.utils import sample_topk


# TODO: try one-hot instead of ones for alpha and beta
class MPSDist(TJDist):
    def __init__(self, config: BaseDistConfig, bypass_config: bool = False, **kwargs):
        if not bypass_config:
            config.param_net.out_dim_encoder = (
                config.horizon * config.rank * config.rank
            )
            config.param_net.out_dim_decoder = config.vocab_size
        else:
            print("WARNING: bypassing config for MPSDist")
        super().__init__(config)
        self.alpha = torch.ones(config.rank) * 0.1
        self.beta = torch.ones(config.rank) * 0.1
        self.dist_config = config

    def forward(self, last_hidden_state: torch.Tensor, **kwargs):
        return self.param_func(last_hidden_state)  # (B, T, HRR, V)

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
        core = self._get_params_from_cache(
            last_hidden_state, use_cache, save_cache
        )  # (B, T, HRR, V)
        alpha = (self.alpha.reshape(1, 1, self.rank).repeat(batch_size, seq_len, 1)).to(
            last_hidden_state.device
        )  # (B, T, R)
        beta = (self.beta.reshape(1, 1, self.rank).repeat(batch_size, seq_len, 1)).to(
            last_hidden_state.device
        )  # (B, T, R)
        return (
            alpha,
            core.reshape(
                batch_size,
                seq_len,
                self.dist_config.horizon,
                self.rank,
                self.rank,
                self.vocab_size,
            ).permute(0, 1, 2, 3, 5, 4),
            beta,
        )

    @classmethod
    def from_pretrained(
        cls, linear: torch.nn.Linear, config: BaseDistFromLinearConfig, **kwargs
    ):
        """Create an MPS distribution from a linear layer.

        Args:
            linear (torch.nn.Linear): Linear layer to use as a base. Shape: (D, V)
            config (BaseDistFromLinearConfig): Configuration for the distribution.

        Returns:
            CPDist: CP distribution with the given configuration.
        """

        n_emb, vocab_size = linear.weight.shape
        if linear.bias is not None:
            raise Warning("CPDist: Skiping bias initialization.")

        return cls(
            config=BaseDistConfig(
                vocab_size=vocab_size,
                horizon=config.horizon,
                rank=config.rank,
                param_net=config.param_net,
            ),
            **kwargs,
        )

    def sample(
        self,
        hidden_state: torch.Tensor,
        horizon: Optional[int] = None,
        do_sample: bool = False,
        top_k: int = 200,
        **kwargs,
    ):
        horizon = self.get_horizon(horizon)
        batch_size = hidden_state.size(0)
        dvc = hidden_state.device
        y_hat = torch.empty(batch_size, 0, device=dvc, dtype=torch.long)
        alpha, core, beta = self.get_mps_params(
            hidden_state[:, -1:, :],
        )  # (B, 1, R), (B, 1, H, R, V, R), (B, 1, R)
        py_list = []
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
            p_ops_tilde, _ = select_margin_mps_tensor_batched(
                alpha=alpha.squeeze(1),
                beta=beta.squeeze(1),
                core=core.squeeze(1)[:horizon],
                ops=ops_tensor,
            )  # (V,), (T,)
            py_list.append(p_ops_tilde)
            if do_sample:
                next_token = sample_topk(p_ops_tilde, top_k, num_samples=1)
            else:  # greedy sampling
                next_token = sample_topk(p_ops_tilde, 1, num_samples=1)
            y_hat = torch.cat([y_hat, next_token], dim=1)
        py = torch.stack(py_list, dim=1)  # (B, H, V)
        return y_hat, py  # (B, H)

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
        horizon = self.get_horizon(points.size(-1))
        alpha, core, beta = self.get_mps_params(
            last_hidden_state,
        )  # (B, T, R), (B, T, H, R, V, R), (B, T, R)

        prepared_alpha = alpha.reshape(batch_size * seq_len, self.rank)
        prepared_beta = beta.reshape(batch_size * seq_len, self.rank)
        prepared_core = core.reshape(
            batch_size * seq_len,
            self.horizon,
            self.rank,
            self.vocab_size,
            self.rank,
        )[:, :horizon]

        p_tilde, p_tilde_scale_factors = select_margin_mps_tensor_batched(
            alpha=prepared_alpha,
            beta=prepared_beta,
            core=prepared_core,
            ops=points.reshape(batch_size * seq_len, horizon),
        )  # (BT,), (BT, T)
        norm_consts, norm_consts_scale_factors = select_margin_mps_tensor_batched(
            alpha=prepared_alpha,
            beta=prepared_beta,
            core=prepared_core,
            ops=torch.full(
                (batch_size * seq_len, horizon),
                -2,
                dtype=torch.long,
                device=last_hidden_state.device,
            ),
        )

        assert len(p_tilde_scale_factors) == len(
            norm_consts_scale_factors
        ), "Scale factors for p_tilde and norm_consts should have the same length"

        if (
            len(p_tilde_scale_factors) == 0
            and len(norm_consts_scale_factors) == 0
            and not torch.all(p_tilde <= norm_consts)
        ):
            # If no scale factors are used, and p_tilde is less than norm_consts throw an error
            raise ValueError(
                "p_tilde is less than norm_consts, please check the input points."
            )

        return (
            p_tilde.reshape(batch_size, seq_len),
            [s.reshape(batch_size, seq_len) for s in p_tilde_scale_factors],
            norm_consts.reshape(batch_size, seq_len),
            [s.reshape(batch_size, seq_len) for s in norm_consts_scale_factors],
        )
