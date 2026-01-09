
import torch.nn as nn

class MLPStudent(nn.Module):
    def __init__(self, lookback=24, horizon=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(lookback, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, horizon)
        )

    def forward(self, x):
        return self.net(x.squeeze(-1))
