from typing import Callable, Optional
import torch

from tjdnet.distributions._base import (
    AbstractDist,
    BaseDistConfig,
    BaseDistConfig,
    BaseDistOutput,
)


class STPDist(AbstractDist):
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
        super().__init__()

        assert config.horizon == 1, "Horizon must be 1 for STPDist"

        self.vocab_size = config.vocab_size
        self.rank = config.rank
        self.horizon = config.horizon
        self.decoder = torch.nn.Linear(
            config.embedding_dim,
            config.vocab_size,
            bias=False,
        )

    @classmethod
    def from_pretrained(cls, linear: torch.nn.Linear, config: BaseDistConfig, **kwargs):
        if linear.bias is not None:
            raise ValueError("Linear bias is not supported for STPDist")
        w = linear.weight  # (V, D)
        obj = cls(config)
        obj.decoder.weight = w
        return obj

    def get_output_embeddings(self):
        return self.decoder.weight  # (V, D)

    def set_output_embeddings(self, new_embeddings):
        self.decoder.weight = new_embeddings

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
        logits_p = self.decoder(x)  # (B, V)
        py = torch.nn.functional.softmax(logits_p, dim=-1)  # (B, V)
        y_hat = sample_fn(py).unsqueeze(1)  # (B, 1)
        py = py.unsqueeze(1)  # (B, 1, V)
        if return_logits:
            return y_hat, py
        return y_hat, py / py.sum(dim=-1, keepdim=True)

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        """Computes loss for CPB distribution.

        Args:
            x (torch.Tensor): Input features. Shape (B, D). (i.e., last hidden state)
            y (torch.Tensor): Target labels. Shape (B, H).

        Returns:
            torch.Tensor: Computed loss. Shape (B,).
        """
        # preds = torch.nn.functional.softmax(self.decoder(x), dim=-1)  # (B, V)
        # p_tilde_select = preds.gather(dim=-1, index=y).squeeze(-1)  # (B,)
        # return -(p_tilde_select.log())

        # CE loss (should be equivalent)
        preds = self.decoder(x)  # (B, V)
        if y is not None:
            loss = torch.nn.functional.cross_entropy(
                preds, y.squeeze(-1), reduction="none"
            )
            return BaseDistOutput(loss=loss, logits=preds)
        return BaseDistOutput(logits=preds)
