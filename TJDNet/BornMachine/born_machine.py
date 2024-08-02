from typing import Tuple
import torch
import torch.nn as nn

from TJDNet.BornMachine.base_tensorized_distribution import BaseTensorizedDistribution


class BornMachine:
    def __init__(self, alpha: torch.Tensor, core: torch.Tensor, beta: torch.Tensor):
        pass

    def select(self, y: torch.Tensor) -> torch.Tensor:
        """Selection operation for the Born Machine.

        Args:
            y (torch.Tensor): Select vector. Shape: (seq_len,)

        Returns:
            torch.Tensor: Evaluation of the Born Machine. Shape: (1,)
        """
        raise NotImplementedError

    def get_normalization_constant(self) -> torch.Tensor:
        """Get the normalization constant of the Born Machine.

        Returns:
            torch.Tensor: Normalization constant. Shape: (1,)
        """
        raise NotImplementedError


class Sequence2BornOutput:
    """Output of the Sequence2Born Layer.

    Attributes:
        p_tilde (torch.Tensor): Unnormalized probability of a sequence. Shape: (batch_size,)
        norm_const (torch.Tensor): Normalization constant of the Born Machine. Shape: (batch_size,)
        loss (torch.Tensor): Loss function. Shape: (batch_size,) or (1,)
    """

    def __init__(
        self, p_tilde: torch.Tensor, norm_const: torch.Tensor, loss: torch.Tensor
    ):
        self.p_tilde = p_tilde
        self.norm_const = norm_const
        self.loss = loss


class Sequence2Born(BaseTensorizedDistribution):
    r"""Born Machine Layer.

    Projects a varaible-length sequence of embeddings to a Born Machine Tensor Network.

    Args:
        n_emb (int): Embedding dimension.
        n_vocab (int): Vocab size.
        rank (int, optional): Rank of the Born Machine Tensor Network. Defaults to 2.

    Examples::

        >>> S2B = Sequence2Born(64, 8)
        >>> input = torch.randn(8, 8, 64)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([8,])
    """

    def __init__(self, n_emb: int, n_vocab: int, rank: int = 2, *args, **kwargs):
        super(Sequence2Born, self).__init__()
        self.n_emb = n_emb
        self.n_vocab = n_vocab
        self.n_born_machine_params = n_vocab * rank * rank + 2 * rank
        self.w_alpha = nn.Parameter(torch.empty(n_emb, rank))
        self.w_beta = nn.Parameter(torch.empty(n_emb, rank))
        self.w_core = nn.Parameter(torch.empty(n_emb, n_vocab * rank * rank))

    def select(self, y: torch.Tensor) -> torch.Tensor:
        """Selection operation for the Born Machine Layer.

        Args:
            y (torch.Tensor): Select vector. Shape: (seq_len,)

        Returns:
            torch.Tensor: Evaluation of the Born Machine Layer. Shape: (1,)
        """
        assert torch.all(y < self.n_vocab), "Invalid selection vector."
        raise NotImplementedError

    def sample(self, max_len: int) -> torch.Tensor:
        """Sample operation for the Born Machine Layer.

        Args:
            max_len (int): Maximum length of the sequence to sample.

        Returns:
            torch.LongTensor: Sampled sequence. Shape: (max_len,)
        """
        raise NotImplementedError

    def _get_born_params(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_reshaped = x.view(-1, self.n_emb)
        alpha = x_reshaped @ self.w_alpha
        beta = x_reshaped @ self.w_beta
        core = x_reshaped @ self.w_core
        return alpha, core, beta

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, eps=1e-6
    ) -> Sequence2BornOutput:
        """Forward pass for the Born Machine Layer.

        Args:
            x (torch.Tensor): Input sequence. Shape: (batch_size, seq_len, n_emb)

        Returns:
            torch.Tensor: Loss function. Shape: (1,)
        """
        assert x.size(-1) == self.n_emb, "Invalid embedding dimension."
        alpha, core, beta = self._get_born_params(x)
        bm = BornMachine(alpha=alpha, core=core, beta=beta)
        p_tilde = bm.select(y)
        norm_const = bm.get_normalization_constant()
        loss = -torch.log(p_tilde + eps) + torch.log(norm_const)
        output = Sequence2BornOutput(p_tilde, norm_const, loss)
        return output
