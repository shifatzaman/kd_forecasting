import numpy as np
import torch

from core.dataset import (
    load_series,
    normalize,
    split_series,
    make_windows,
    compute_regime_scores,
)
from core.metrics import mae, rmse, smape, horizon_metrics
from core.kd_trainer_simple import train_student_kd_simple

from teachers.dlinear import DLinearTeacher
from teachers.patchtst import PatchTSTTeacher
from utils.uncertainty import compute_weights


# Original simple MLP
class SimpleMLPStudent(torch.nn.Module):
    def __init__(self, lookback=24, horizon=6):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(lookback, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, horizon)
        )

    def forward(self, x):
        return self.net(x.squeeze(-1))


# =========================
# CONFIG - ORIGINAL
# =========================
DATA_PATH = "data/Wfp_rice.csv"

LOOKBACK = 48
HORIZON = 6
PATCH_LEN = 4

VAL_SIZE = 24
TEST_SIZE = 24

TEACHER_EPOCHS = 200
STUDENT_EPOCHS = 200
ALPHA0 = 0.5

SEED = 42


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(SEED)

    # -------------------------
    # Load & normalize
    # -------------------------
    series = load_series(DATA_PATH)
    series_norm, mean, std = normalize(series)

    train, val, test = split_series(series_norm, VAL_SIZE, TEST_SIZE)

    # context-aware windows
    val_ctx = np.concatenate([train[-LOOKBACK:], val])
    test_ctx = np.concatenate([np.concatenate([train, val])[-LOOKBACK:], test])

    Xtr, Ytr = make_windows(train, LOOKBACK, HORIZON)
    Xva, Yva = make_windows(val_ctx, LOOKBACK, HORIZON)
    Xte, Yte = make_windows(test_ctx, LOOKBACK, HORIZON)

    print(f"Training windows: {len(Xtr)}")
    print(f"Test windows: {len(Xte)}")

    # -------------------------
    # Regime scores (TRAIN ONLY)
    # -------------------------
    regime_scores = compute_regime_scores(train, LOOKBACK)
    regime_scores = regime_scores[: len(Xtr)]

    # -------------------------
    # Teachers
    # -------------------------
    teachers = [
        DLinearTeacher(LOOKBACK, HORIZON, TEACHER_EPOCHS),
        PatchTSTTeacher(LOOKBACK, HORIZON, TEACHER_EPOCHS, PATCH_LEN),
    ]

    tr_preds, va_preds = [], []

    print("\nTraining teachers...")
    for i, t in enumerate(teachers):
        print(f"  Teacher {i+1}/2...", end=" ", flush=True)
        t.fit(train)
        tr_preds.append(t.predict(Xtr))
        va_preds.append(t.predict(Xva))
        print("✓")

    weights = compute_weights(Yva, va_preds)
    print("Teacher weights:", weights)

    # -------------------------
    # Student (Original Simple KD)
    # -------------------------
    student = SimpleMLPStudent(LOOKBACK, HORIZON)

    # Simple KD trainer (no fancy features)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.tensor(Xtr, dtype=torch.float32).unsqueeze(-1).to(device)
    Y = torch.tensor(Ytr, dtype=torch.float32).to(device)
    T = torch.tensor(np.stack(tr_preds, axis=1), dtype=torch.float32).to(device)
    W = torch.tensor(weights, dtype=torch.float32).view(1, -1, 1).to(device)
    R = torch.tensor(regime_scores, dtype=torch.float32).view(-1, 1).to(device)

    student = student.to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)

    H = Y.shape[1]
    horizon_weights = torch.linspace(1.0, 0.5, H, device=device).view(1, -1)

    print("\nTraining student...")
    student.train()
    for e in range(STUDENT_EPOCHS):
        alpha = ALPHA0 * (1 - e / (STUDENT_EPOCHS - 1))

        preds = student(X)
        teacher_target = (T * W).sum(dim=1)

        loss_sup = (horizon_weights * torch.abs(preds - Y)).mean()
        loss_kd = (R * horizon_weights * (preds - teacher_target) ** 2).mean()
        loss = (1 - alpha) * loss_sup + alpha * loss_kd

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (e + 1) % 50 == 0:
            print(f"  Epoch {e+1}/{STUDENT_EPOCHS}, Loss: {loss.item():.4f}")

    # -------------------------
    # Evaluate
    # -------------------------
    student.eval()
    with torch.no_grad():
        preds_norm = student(
            torch.tensor(Xte, dtype=torch.float32).unsqueeze(-1).to(device)
        ).cpu().numpy()

    preds = preds_norm * std + mean
    Yte_denorm = Yte * std + mean

    print("\n" + "="*60)
    print("RESULTS (Original Configuration)")
    print("="*60)
    print(f"Overall MAE:   {mae(Yte_denorm, preds):.4f}")
    print(f"Overall RMSE:  {rmse(Yte_denorm, preds):.4f}")
    print(f"Overall sMAPE: {smape(Yte_denorm, preds):.4f}")

    print("\nHorizon-wise metrics:")
    for h, m in horizon_metrics(Yte_denorm, preds).items():
        print(f"  {h}: MAE={m['MAE']:.4f}, RMSE={m['RMSE']:.4f}, sMAPE={m['sMAPE']:.4f}")


if __name__ == "__main__":
    main()
