from typing import Optional
import torch

from tjdnet.distributions._base import (
    BaseDistFromLinearConfig,
    BaseDistribution,
    BaseDistConfig,
)
from tjdnet.distributions.tpnet import TensorParamNetConfig
from tjdnet.utils import sample_topk


class BaseDist(BaseDistribution):
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
        assert config.horizon == 1, "Only horizon=1 is supported for now"
        config.param_net.use_decoder = False  # Set TPNet decoder to False
        config.param_net.out_dim_encoder = config.vocab_size
        config.param_net.hidden_dim = 1
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.rank = config.rank
        self.horizon = config.horizon

    def forward(self, last_hidden_state: torch.Tensor, **kwargs) -> torch.Tensor:
        p_tilde = self.param_func(last_hidden_state)  # (B, T, 1, V)
        return p_tilde.squeeze(-1)  # (B, T, V)

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
                param_net=config.param_net,
            ),
            **kwargs,
        )

    def sample(
        self,
        hidden_state: torch.Tensor,  # (B, T, D)
        horizon: Optional[int] = None,
        do_sample: bool = False,
        top_k: int = 200,
        **kwargs,
    ):
        if horizon and horizon > 1:
            raise ValueError("Horizon must be 1 for base distribution")
        model_head_params = self.forward(hidden_state[:, -1:, :])  # (B, 1, V)
        y_hat = sample_topk(
            model_head_params.squeeze(1), top_k=top_k if do_sample else 1  # topk/greedy
        )  # (B, 1)
        py = model_head_params
        return y_hat, py  # (B, 1), (B, 1, V)

    def evaluate_at_points_and_get_norm_consts(
        self,
        last_hidden_state: torch.Tensor,
        points: torch.Tensor,
        **kwargs,
    ):
        """Evaluate the distribution at the given points.

        Args:
            last_hidden_state (torch.Tensor): Hidden states of the transformer of shape (B, T, D)
            points (torch.Tensor): Points to evaluate the distribution. Shape (B, T, 1)
            is_normalized (bool, optional): Whether the points are normalized. Defaults to False.

        Returns:
            tuple:
                - torch.Tensor: Unormalized distribution `p_tilde` at the points of shape (B, T)
                - list: Scale factors for `p_tilde`
                - torch.Tensor: Normalization constants `z` of shape (B, T)
                - list: Scale factors for `z`
        """
        # Get indexed distribution
        assert points.size(-1) == 1, "Only 1D points are supported"
        p_tilde = self.forward(last_hidden_state)  # (B, T, V)
        batch_size, seq_len, _ = last_hidden_state.size()

        # (B, T, R*H*V) => (B, T)
        p_tilde_select = torch.gather(
            p_tilde.reshape(batch_size * seq_len, self.vocab_size),
            dim=1,
            index=points.reshape(batch_size * seq_len, self.horizon),
        )  # (B*T, 1)
        norm_consts = torch.sum(p_tilde, dim=-1).reshape(-1)  # (B*T)
        return (
            p_tilde_select.reshape(batch_size, seq_len),
            [],
            norm_consts.reshape(batch_size, seq_len),
            [],
        )
