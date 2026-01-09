
import torch
import torch.nn as nn
from core.dataset import make_windows
from teachers.base import TeacherModel

class PatchTST(nn.Module):
    def __init__(self, lookback=24, horizon=6, patch=4, d=64):
        super().__init__()
        self.np = lookback // patch
        self.embed = nn.Linear(patch, d)
        enc = nn.TransformerEncoderLayer(d, 4, 128, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, 2)
        self.head = nn.Linear(d*self.np, horizon)

    def forward(self, x):
        x = x.squeeze(-1)
        patches = x.unfold(1, self.embed.in_features, self.embed.in_features)
        z = self.encoder(self.embed(patches))
        return self.head(z.reshape(z.size(0), -1))

class PatchTSTTeacher(TeacherModel):
    def __init__(self, epochs=60):
        self.model = PatchTST()
        self.epochs = epochs

    def fit(self, series):
        X, Y = make_windows(series)
        X = torch.tensor(X).unsqueeze(-1)
        Y = torch.tensor(Y)
        opt = torch.optim.AdamW(self.model.parameters(), 1e-3)
        loss = nn.L1Loss()
        for _ in range(self.epochs):
            p = self.model(X)
            l = loss(p, Y)
            opt.zero_grad()
            l.backward()
            opt.step()

    def predict(self, X):
        with torch.no_grad():
            return self.model(torch.tensor(X).unsqueeze(-1)).numpy()
