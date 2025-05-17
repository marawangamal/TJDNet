from typing import Callable, Optional
import torch

from tjdnet.distributions._tjdist import (
    BaseDistFromLinearConfig,
    TJDist,
    BaseDistConfig,
)
from tjdnet.utils import sample_topk


class BaseDist(TJDist):
    def __init__(
        self,
        config: BaseDistConfig,
        **kwargs,
    ):
        """Basic 1D entropy distribution

        Args:
            n_embd (int): Embedding dimension
            vocab_size (int): Vocabulary size
            rank (int): Rank of the CP decomposition
            horizon (int): Horizon of the model (Number of tokens to predict)
        """
        super().__init__(config)
        assert config.horizon == 1, "Only horizon=1 is supported for now"
        self.pos_func = {
            "sq": lambda x: x**2,
            "abs": lambda x: torch.abs(x),
            "exp": torch.exp,
            "none": lambda x: x,
        }[config.positivity_func]
        self.w = torch.nn.Linear(
            in_features=config.in_dim, out_features=config.vocab_size
        )

    def get_params(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        p_tilde = self.pos_func(self.w(x))  # (B, V)
        return p_tilde.squeeze(-1)  # (B, V)

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
            raise Warning("BaseDist: Skiping bias initialization.")

        return cls(
            config=BaseDistConfig(
                vocab_size=vocab_size,
                horizon=config.horizon,
                rank=config.rank,
                in_dim=n_emb,
                hidden_dim=n_emb,
                positivity_func=config.positivity_func,
            ),
            **kwargs,
        )

    def sample(
        self,
        x: torch.Tensor,  # (B, T, D)
        sample_fn: Callable[[torch.Tensor], torch.Tensor],
        horizon: Optional[int] = None,
        return_logits: bool = False,
        **kwargs,
    ):
        if horizon and horizon > 1:
            raise ValueError("Horizon must be 1 for base distribution")
        model_head_params = self.get_params(x)  # (B, V)
        y_hat = sample_fn(model_head_params).unsqueeze(1)  # (B, 1)
        py_tilde = model_head_params.unsqueeze(1)
        if return_logits:  # don't normalize
            return y_hat, py_tilde  # (B, 1), (B, 1, V)
        return y_hat, py_tilde / py_tilde.sum(dim=-1, keepdim=True)  # (B, H)

    def evaluate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        **kwargs,
    ):
        # Get indexed distribution
        assert y.size(-1) == 1, "Only 1D points are supported"
        p_tilde = self.get_params(x)  # (B, T, V)
        batch_size, _ = x.size()

        # (B, T, R*H*V) => (B, T)
        p_tilde_select = torch.gather(
            p_tilde.reshape(batch_size, self.vocab_size),
            dim=1,
            index=y.reshape(batch_size, self.horizon),
        )  # (B*T, 1)
        norm_consts = torch.sum(p_tilde, dim=-1).reshape(-1)  # (B*T)
        return (
            p_tilde_select.reshape(batch_size),
            [],
            norm_consts.reshape(batch_size),
            [],
        )
