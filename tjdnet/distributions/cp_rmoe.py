from typing import Callable, Optional
import torch

from tjdnet.distributions._base import BaseDistConfig
from tjdnet.distributions._tjdist import BaseDistConfig, TJDist

from tjdnet.distributions._tpnet import safe_exp
from tjdnet.tensorops.cp import select_margin_cp_tensor_batched


class CPRMoEDist(TJDist):
    def __init__(self, config: BaseDistConfig, bypass_config=False, **kwargs):
        super().__init__(config)
        self.param_func = None
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
        self.config = config
        R, D, H = config.rank, config.embedding_dim, config.horizon

        # Encoder
        self.w_encoder_experts = torch.nn.ModuleList(
            [torch.nn.Linear(D, H * D) for _ in range(R)]
        )

        # Decoder
        self.w_decoder = torch.nn.Linear(D, config.vocab_size)

    @classmethod
    def from_pretrained(cls, linear: torch.nn.Linear, config: BaseDistConfig, **kwargs):
        raise NotImplementedError("CPDist does not support from_pretrained")

    def get_params(self, x: torch.Tensor, **kwargs):
        # Setup
        B, R, H, D, V = (
            x.size(0),
            self.config.rank,
            self.config.horizon,
            self.config.embedding_dim,
            self.config.vocab_size,
        )

        # Encoder: (B, D) -> (B, D, R', H)
        rmask = torch.randint(0, 2, (R,), dtype=torch.bool, device=x.device)
        rmask[torch.randint(R, (1,), device=x.device)] = True  # guarantee ≥1 True

        # choose `self.n_active` experts randomly
        expert_ids = torch.multinomial(
            torch.arange(R, dtype=torch.float), num_samples=self.config.rank_active
        ).to(torch.long)

        # Apply experts
        p_keep = self.config.rank_active / R
        gamma = 1 / p_keep if self.training and p_keep > 0 else 1.0
        z = gamma * torch.stack(
            [self.w_encoder_experts[eid](x) for eid in expert_ids.tolist()], dim=1
        )  # (B, R', H*D)

        # Decoder: (B, R', H*D) -> (B, R', H, V)
        z = self.positivity_func(self.w_decoder(z.reshape(B, -1, H, D)))

        return z  # (B, R', H, V)

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
        dvc = x.device

        # Output tokens will be placed in `y_hat`
        model_head_params = self.get_params(x)  # (B, R, H, V)
        B, R, H, V = model_head_params.shape
        y_hat = torch.empty(B, 0, device=dvc, dtype=torch.long)

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
            p_ops_tilde, _ = select_margin_cp_tensor_batched(
                cp_params=model_head_params,
                ops=ops_tensor,
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
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"Batch size mismatch: z.shape[0]={x.shape[0]}, y.shape[0]={y.shape[0]}"
            )
        # Get indexed distribution
        params = self.get_params(x)  # (B, R, H, V)
        B, R, H, V = params.shape
        p_tilde, p_tilde_scale_factors = select_margin_cp_tensor_batched(
            cp_params=params.reshape(B, R, H, V),
            ops=y.reshape(B, H),
        )  # (B,), (B, H)
        norm_consts, norm_consts_scale_factors = select_margin_cp_tensor_batched(
            cp_params=params.reshape(B, R, H, V),
            ops=torch.full(
                (B, H),
                -2,
                dtype=torch.long,
                device=x.device,
            ),
        )
        return (p_tilde, p_tilde_scale_factors, norm_consts, norm_consts_scale_factors)
