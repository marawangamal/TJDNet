import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class CPRegressor(nn.Module):
    """
    CP regression with rank-R factors learned by Adam.

        ŷ = Σ_r w_r ∏_{m=1..H} A_m[i_m, r] + b
    """

    def __init__(self, vocab_size: int, horizon: int, rank: int, device="cpu"):
        super().__init__()
        self.V, self.H, self.R = vocab_size, horizon, rank
        self.to(device)

        self.factors = nn.ParameterList(
            [
                nn.Parameter(torch.randn(self.V, self.R, device=device))
                for _ in range(self.H)
            ]
        )
        self.weights = nn.Parameter(torch.ones(self.R, device=device))
        self.bias = nn.Parameter(torch.zeros(1, device=device))

    # -------- forward & predict --------------------------------------------
    def forward(self, coords: torch.Tensor) -> torch.Tensor:  # coords: (N, H) int64
        prod = torch.ones(coords.size(0), self.R, device=coords.device)
        for m in range(self.H):
            prod *= self.factors[m][coords[:, m]]
        return (prod * self.weights).sum(1) + self.bias

    def predict(self, coords: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self(coords)

    # -------- simple .fit  --------------------------------------------------
    def fit(
        self,
        X: torch.Tensor,  # (N, H) int64 coordinates
        y: torch.Tensor,  # (N,)
        lr: float = 1e-2,
        epochs: int = 1000,
        batch_size: int = 512,
    ):
        dl = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for _ in range(epochs):
            for xb, yb in dl:
                opt.zero_grad()
                loss_fn(self(xb), yb).backward()
                opt.step()
        return self
