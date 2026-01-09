
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.dataset import make_windows
from teachers.base import TeacherModel

class DLinear(nn.Module):
    def __init__(self, lookback=24, horizon=6, kernel=25):
        super().__init__()
        self.linear_t = nn.Linear(lookback, horizon)
        self.linear_s = nn.Linear(lookback, horizon)
        self.kernel = kernel

    def forward(self, x):
        x = x.squeeze(-1)
        pad = self.kernel // 2
        trend = F.avg_pool1d(
            F.pad(x.unsqueeze(1), (pad, pad), mode="reflect"),
            self.kernel, stride=1
        ).squeeze(1)
        seasonal = x - trend
        return self.linear_t(trend) + self.linear_s(seasonal)

class DLinearTeacher(TeacherModel):
    def __init__(self, lookback=24, horizon=6, epochs=50):
        self.model = DLinear(lookback, horizon)
        self.epochs = epochs

    def fit(self, series):
        X, Y = make_windows(series)
        X = torch.tensor(X).unsqueeze(-1)
        Y = torch.tensor(Y)
        opt = torch.optim.AdamW(self.model.parameters(), 1e-3)
        loss = nn.L1Loss()
        for _ in range(self.epochs):
            pred = self.model(X)
            l = loss(pred, Y)
            opt.zero_grad()
            l.backward()
            opt.step()

    def predict(self, X):
        with torch.no_grad():
            X = torch.tensor(X).unsqueeze(-1)
            return self.model(X).numpy()
