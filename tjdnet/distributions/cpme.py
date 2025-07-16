import os, datetime
from typing import Callable, Optional
import torch

from tjdnet.distributions._base import BaseDistConfig
from tjdnet.distributions._tjdist import BaseDistConfig, TJDist

from tjdnet.tensorops.cp import select_margin_cp_tensor_batched_w_decoder


def safe_exp(x: torch.Tensor) -> torch.Tensor:
    """Safe exponential function to avoid overflow."""
    return torch.exp(torch.clamp(x, max=20.0))  # Clamp to


class CPME(TJDist):
    def __init__(self, config: BaseDistConfig, **kwargs):
        """CP Distribution

        Args:
            n_embd (int): Embedding dimension
            vocab_size (int): Vocabulary size
            rank (int): Rank of the CP decomposition
            horizon (int): Horizon of the model (Number of tokens to predict)
        """
        super().__init__(config)

        # === config
        self.param_func = None
        self.config = config
        H, R, D, V = (
            self.config.horizon,
            self.config.rank,
            self.config.embedding_dim,
            self.config.vocab_size,
        )
        self.positivity_func = {
            "sq": lambda x: x**2,
            "abs": lambda x: torch.abs(x),
            "exp": torch.exp,
            "safe_exp": safe_exp,
            "sigmoid": torch.sigmoid,
            "relu": torch.relu,
            "leaky_relu": torch.nn.functional.leaky_relu,
            "none": lambda x: x,
        }[config.positivity_func]

        # === params
        self.w_cp = torch.nn.Linear(D, R * H, bias=False)
        self.decoder = torch.nn.Parameter(torch.randn(D, V))

    @property
    def cp_decoder(self):
        return self.positivity_func(self.decoder)

    @classmethod
    def from_pretrained(cls, linear: torch.nn.Linear, config: BaseDistConfig, **kwargs):
        raise NotImplementedError("CPDist does not support from_pretrained")

    def get_params(self, x: torch.Tensor, **kwargs):
        B, R, H = (x.size(0), self.config.rank, self.config.horizon)
        params = self.w_cp(x).reshape(B, R, H, -1)  # (B, R, H, d)
        return self.positivity_func(params)  # (B, R, H, d)

    def sample(
        self,
        x: torch.Tensor,
        # (B, D) -> (B,)
        sample_fn: Callable[[torch.Tensor], torch.Tensor],
        horizon: Optional[int] = None,
        return_logits: bool = False,
        refine: bool = False,
        **kwargs,
    ):
        """Computes P(yh|x, y1:h-1) for h in [1, H].

        Args:
            x (torch.Tensor): Input features. Shape (B, D). (i.e., last hidden state)
            sample_fn (Callable): Sampling function.
            horizon (Optional[int]): Horizon for sampling. Must be <= self.horizon.
            return_logits (bool): Whether to return logits or probabilities.
            refine (bool): Whether to refine the sampling process.

        Returns:
            tuple:
                - Evaluation of the distribution at the points of shape (B, H).
                - Probabilities of shape (B, H, V) or logits of shape (B, H, V).
        """
        H = (
            min(horizon, self.config.horizon)
            if horizon is not None
            else self.config.horizon
        )
        B = x.size(0)
        dvc = x.device

        # Output tokens will be placed in `y_hat`
        y_hat = torch.empty(B, 0, device=dvc, dtype=torch.long)
        model_head_params = self.get_params(x)  # (B, R, H, d)
        py_tilde_list = []

        # Autoregressive sampling
        # Operations tensor (B, T). Describes batch operations to perform on the CP tensor
        # modelled by `model_head_params`.
        # Example:
        #  y_hat = [[1, 2, 3]]  # (B, T)
        #  ops_tensor = [[1, 2, -2]]  # (B, T)
        #  p_ops_tilde = A^{(1))_1} * A^{(2)}_2 * (𝜮_r A^{(3)}_r)
        for h in range(H):
            ops_tensor = torch.cat(
                (
                    y_hat,  # selection
                    -1 * torch.ones(B, 1, dtype=torch.long, device=dvc),  # free leg
                    -2  # marginalization
                    * torch.ones(B, (H - h - 1), dtype=torch.long, device=dvc),
                ),
                dim=1,
            )  # (B, T)
            #  (B, R, H, V) -> (B, V)
            p_ops_tilde, _ = select_margin_cp_tensor_batched_w_decoder(
                cp_params=model_head_params,
                ops=ops_tensor,
                decoder=self.cp_decoder,
            )  # (B, V), (B,) * T
            py_tilde_list.append(p_ops_tilde)
            next_token = sample_fn(p_ops_tilde).unsqueeze(1)  # (B,1)

            y_hat = torch.cat([y_hat, next_token], dim=1)
        py_tilde = torch.stack(py_tilde_list, dim=1)  # (B, H, V)
        if return_logits:  # don't normalize
            return y_hat, py_tilde
        return y_hat, py_tilde / py_tilde.sum(dim=-1, keepdim=True)  # (B, H)

    def _describe_tensor_cp_eff(self, tensor: torch.Tensor, name: str):
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        print(
            f"CP_EFFDiag - {name}: {tensor.shape}, min={tensor.min().item():.3f}, max={tensor.max().item():.3f}, "
            f"NaN={nan_count}/{tensor.numel()}, Inf={inf_count}/{tensor.numel()}"
        )

    def evaluate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        **kwargs,
    ):
        # Get indexed distribution
        H = self.config.horizon
        B = x.size(0)
        params = self.get_params(x)  # (B, R, H, d)
        p_tilde, p_tilde_scale_factors = select_margin_cp_tensor_batched_w_decoder(
            cp_params=params,
            ops=y.reshape(B, H),
            decoder=self.cp_decoder,
        )  # (B,), (B, H)

        # Check for NaNs in p_tilde or its scale factors
        if torch.isnan(p_tilde).any() or any(
            torch.isnan(sf).any()
            for sf in p_tilde_scale_factors
            if isinstance(sf, torch.Tensor)
        ):
            print("=== CP_EFF DIAGNOSTICS ===")
            os.makedirs("debug", exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_filepath = f"debug/cp_eff_nan_debug_{ts}.pt"
            torch.save(
                {
                    "params": params.cpu(),
                    "y": y.cpu(),
                    "p_tilde": (
                        p_tilde.cpu() if isinstance(p_tilde, torch.Tensor) else p_tilde
                    ),
                    "p_tilde_scale_factors": [
                        sf.cpu() if isinstance(sf, torch.Tensor) else sf
                        for sf in p_tilde_scale_factors
                    ],
                    "cp_decoder": (
                        self.cp_decoder.cpu()
                        if isinstance(self.cp_decoder, torch.Tensor)
                        else self.cp_decoder
                    ),
                    "param_func.w.weight": self.w_cp.weight.data.cpu(),
                    "x": x.cpu(),
                },
                debug_filepath,
            )
            print("NaN detected in p_tilde or its scale factors!")
            self._describe_tensor_cp_eff(params, "params")
            self._describe_tensor_cp_eff(y.reshape(B, H), "ops")
            self._describe_tensor_cp_eff(self.cp_decoder, "cp_decoder")
            print(f"Saved debug tensors to {debug_filepath}")
            print("=== END CP_EFF DIAGNOSTICS ===")

        # Marginalize over all tokens
        norm_consts, norm_consts_scale_factors = (
            select_margin_cp_tensor_batched_w_decoder(
                cp_params=params,
                ops=torch.full(
                    (B, H),
                    -2,
                    dtype=torch.long,
                    device=x.device,
                ),
                decoder=self.cp_decoder,
            )
        )
        return (p_tilde, p_tilde_scale_factors, norm_consts, norm_consts_scale_factors)
