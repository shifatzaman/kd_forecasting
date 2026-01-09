import torch
import torch.nn as nn
import torch.nn.functional as F

from teachers.base import TeacherModel
from core.dataset import make_windows


class DLinear(nn.Module):
    """
    DLinear: Decomposition-Linear model for time series forecasting.

    Decomposes input into:
      - Trend (moving average)
      - Seasonal / residual

    Input:  [B, lookback, 1]
    Output: [B, horizon]
    """

    def __init__(
        self,
        lookback: int,
        horizon: int,
        kernel_size: int = 25,
    ):
        super().__init__()

        self.lookback = lookback
        self.horizon = horizon
        self.kernel_size = kernel_size

        # Linear projections for trend and seasonal components
        self.linear_trend = nn.Linear(lookback, horizon)
        self.linear_seasonal = nn.Linear(lookback, horizon)

    def moving_average(self, x):
        """
        x: [B, lookback]
        returns: [B, lookback]
        """
        pad = self.kernel_size // 2

        # reflect padding avoids edge artifacts
        x = F.pad(x.unsqueeze(1), (pad, pad), mode="reflect")
        x = F.avg_pool1d(x, kernel_size=self.kernel_size, stride=1)

        return x.squeeze(1)

    def forward(self, x):
        """
        x: [B, lookback, 1]
        """
        # remove channel dimension
        x = x.squeeze(-1)  # [B, lookback]

        # trend + seasonal decomposition
        trend = self.moving_average(x)
        seasonal = x - trend

        # linear forecasts
        out_trend = self.linear_trend(trend)
        out_seasonal = self.linear_seasonal(seasonal)

        return out_trend + out_seasonal  # [B, horizon]


class DLinearTeacher(TeacherModel):
    """
    DLinear teacher wrapper following TeacherModel interface.
    """

    def __init__(
        self,
        lookback: int,
        horizon: int,
        epochs: int,
        kernel_size: int = 25,
        lr: float = 1e-3,
        device: str | None = None,
    ):
        self.lookback = lookback
        self.horizon = horizon
        self.epochs = epochs
        self.lr = lr

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DLinear(
            lookback=lookback,
            horizon=horizon,
            kernel_size=kernel_size,
        ).to(self.device)

    def fit(self, series):
        """
        Train DLinear on a single univariate series.
        """
        X, Y = make_windows(
            series,
            lookback=self.lookback,
            horizon=self.horizon,
        )

        X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(self.device)
        Y = torch.tensor(Y, dtype=torch.float32).to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        loss_fn = nn.L1Loss()

        self.model.train()
        for _ in range(self.epochs):
            preds = self.model(X)
            loss = loss_fn(preds, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    @torch.no_grad()
    def predict(self, X):
        """
        X: numpy array [N, lookback]
        Returns: numpy array [N, horizon]
        """
        self.model.eval()

        X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(self.device)
        preds = self.model(X)

        return preds.cpu().numpy()