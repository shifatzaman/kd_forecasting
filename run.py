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
from core.kd_trainer import train_student_kd

from teachers.dlinear import DLinearTeacher
from teachers.patchtst import PatchTSTTeacher
from students.mlp import MLPStudent
from utils.uncertainty import compute_weights


# =========================
# CONFIG
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

    for t in teachers:
        t.fit(train)
        tr_preds.append(t.predict(Xtr))
        va_preds.append(t.predict(Xva))

    weights = compute_weights(Yva, va_preds)
    print("Teacher weights:", weights)

    # -------------------------
    # Student (Bounded Regime-Aware KD)
    # -------------------------
    student = MLPStudent(LOOKBACK, HORIZON)

    student = train_student_kd(
        X=Xtr,
        Y=Ytr,
        teacher_preds=tr_preds,
        weights=weights,
        student=student,
        epochs=STUDENT_EPOCHS,
        regime_scores=regime_scores,
        alpha0=ALPHA0,
    )

    # -------------------------
    # Evaluate
    # -------------------------
    student.eval()
    with torch.no_grad():
        preds_norm = student(
            torch.tensor(Xte, dtype=torch.float32).unsqueeze(-1)
        ).cpu().numpy()

    preds = preds_norm * std + mean
    Yte_denorm = Yte * std + mean

    print("\nOverall MAE:", mae(Yte_denorm, preds))
    print("Overall RMSE:", rmse(Yte_denorm, preds))
    print("Overall sMAPE:", smape(Yte_denorm, preds))

    print("\nHorizon-wise metrics:")
    for h, m in horizon_metrics(Yte_denorm, preds).items():
        print(h, m)

    print("\nTraining complete (bounded regime-aware KD).")


if __name__ == "__main__":
    main()