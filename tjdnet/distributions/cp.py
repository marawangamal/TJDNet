from typing import Callable
from git import Optional
import torch

from tjdnet.distributions._tjdist import (
    BaseDistConfig,
    BaseDistFromLinearConfig,
    TJDist,
)

from tjdnet.tensorops.cp import (
    select_margin_cp_tensor_batched,
    select_margin_cp_tensor_decoder_batched,
)
from tjdnet.utils import sample_topk


class CPDist(TJDist):
    def __init__(self, config: BaseDistConfig, **kwargs):
        super().__init__(config)
        self.pos_func = {
            "sq": lambda x: x**2,
            "abs": lambda x: torch.abs(x),
            "exp": torch.exp,
            "none": lambda x: x,
        }[config.positivity_func]
        self.w = torch.nn.Linear(
            in_features=config.in_dim,
            out_features=config.hidden_dim * config.rank * config.horizon,
        )
        self.decoder = torch.nn.Parameter(
            torch.randn(config.hidden_dim, config.vocab_size)
        )

    @classmethod
    def from_linear(
        cls, linear: torch.nn.Linear, config: BaseDistFromLinearConfig, **kwargs
    ):
        """Create a CP distribution from a linear layer.

        Args:
            linear (torch.nn.Linear): Linear layer to use as a base. Shape: (D, V)
            config (BaseDistFromLinearConfig): Configuration for the distribution.

        Returns:
            CPDist: CP distribution with the given configuration.
        """

        n_emb, vocab_size = linear.weight.shape
        if linear.bias is not None:
            raise Warning("CPDist: Skiping bias initialization.")

        obj = cls(
            config=BaseDistConfig(
                vocab_size=vocab_size,
                horizon=config.horizon,
                rank=config.rank,
                in_dim=n_emb,
                hidden_dim=n_emb,
            ),
            **kwargs,
        )

        # Initialize the parameters in obj.tensor_param_net
        # with the parameters from the linear layer
        obj.w.linear.weight.data = linear.weight.data
        return obj

    def get_params(self, x: torch.Tensor, **kwargs):
        params = self.pos_func(self.w(x))  # (B, RHd)
        return params.reshape(
            -1, self.rank, self.horizon, self.config.hidden_dim
        ), self.pos_func(self.decoder)

    def sample(
        self,
        x: torch.Tensor,
        # (B, D) -> (B,)
        sample_fn: Callable[[torch.Tensor], torch.Tensor],
        horizon: Optional[int] = None,
        return_logits: bool = False,
        **kwargs,
    ):
        horizon = self.get_horizon(horizon)
        batch_size = x.size(0)
        dvc = x.device
        y_hat = torch.empty(batch_size, 0, device=dvc, dtype=torch.long)
        cp_params, cp_decoder = self.get_params(x)  # (B, R, H, d), (d, V)
        py_tilde_list = []
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
            #  (B, R, H, V) -> (B, V)
            p_ops_tilde, _ = select_margin_cp_tensor_decoder_batched(
                cp_params=cp_params,
                ops=ops_tensor,
                cp_decoder=cp_decoder,
            )  # (B, V), (B,) * T
            py_tilde_list.append(p_ops_tilde)
            next_token = sample_fn(p_ops_tilde).unsqueeze(1)  # (B,1)

            y_hat = torch.cat([y_hat, next_token], dim=1)
        py_tilde = torch.stack(py_tilde_list, dim=1)  # (B, H, V)
        if return_logits:  # don't normalize
            return y_hat, py_tilde
        return y_hat, py_tilde / py_tilde.sum(dim=-1, keepdim=True)  # (B, H)

    def evaluate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        **kwargs,
    ):
        # Get indexed distribution
        horizon = self.horizon
        B = x.size(0)
        cp_params, cp_decoder = self.get_params(x)  # (B, R, H, d), (d, V)
        p_tilde, p_tilde_scale_factors = select_margin_cp_tensor_decoder_batched(
            cp_params=cp_params,
            cp_decoder=cp_decoder,
            ops=y.reshape(B, horizon),
        )  # (B,), (B, H)
        norm_consts, norm_consts_scale_factors = (
            select_margin_cp_tensor_decoder_batched(
                cp_params=cp_params,
                cp_decoder=cp_decoder,
                ops=torch.full(
                    (B, horizon),
                    -2,
                    dtype=torch.long,
                    device=x.device,
                ),
            )
        )
        return (p_tilde, p_tilde_scale_factors, norm_consts, norm_consts_scale_factors)
