import torch


class BaseTensorizedDistribution(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(BaseTensorizedDistribution, self).__init__()

    def select(self, y: torch.Tensor) -> torch.Tensor:
        # y: (seq_len,)
        # output: (1,)
        raise NotImplementedError

    def sample(self, max_len: int) -> torch.Tensor:
        # output: (max_len,)
        raise NotImplementedError

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, n_emb)
        # y: (batch_size, seq_len)
        # output: (1,)  -- loss function
        raise NotImplementedError


class Sequence2Born(BaseTensorizedDistribution):
    """Born Machine Layer.

    Projects a sequence onto the parameter space of a Born Machine.

    Args:
        torch (_type_): _description_
    """

    def __init__(self, n_emb, n_vocab, seq_len):
        super(Sequence2Born, self).__init__()
        self.n_emb = n_emb
        self.n_vocab = n_vocab
        self.seq2btparams = torch.nn.Linear(n_emb, n_vocab)

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

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Born Machine Layer.

        Args:
            x (torch.Tensor): Input sequence. Shape: (batch_size, seq_len, n_emb)

        Returns:
            torch.Tensor: Loss function. Shape: (1,)
        """
        raise NotImplementedError
