# run.py
# Single-commodity, 6-month horizon forecasting with multi-teacher KD
# Teachers: DLinear + PatchTST
# Student: MLP
# Weighting: uncertainty-aware (validation-based)
# Fixes included: normalization + context-aware val/test windows + horizon-weighted KD loss

import os
import numpy as np
import torch

from core.dataset import load_series, split_series, make_windows, normalize
from core.metrics import mae, rmse, smape, horizon_metrics
from core.kd_trainer import train_student_kd

from teachers.dlinear import DLinearTeacher
from teachers.patchtst import PatchTSTTeacher
from students.mlp import MLPStudent
from utils.uncertainty import compute_weights


# =========================
# CONFIG (EDIT THESE)
# =========================
DATA_PATH = os.environ.get("DATA_PATH", "data/Wfp_rice.csv")

LOOKBACK = int(os.environ.get("LOOKBACK", 48))
HORIZON = int(os.environ.get("HORIZON", 6))

VAL_SIZE = int(os.environ.get("VAL_SIZE", 24))
TEST_SIZE = int(os.environ.get("TEST_SIZE", 24))

TEACHER_EPOCHS = int(os.environ.get("TEACHER_EPOCHS", 200))
STUDENT_EPOCHS = int(os.environ.get("STUDENT_EPOCHS", 200))

PATCH_LEN = int(os.environ.get("PATCH_LEN", 4))

SEED = int(os.environ.get("SEED", 42))


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def assert_shapes(X, Y, lookback, horizon, name):
    assert X.ndim == 2 and Y.ndim == 2, f"{name}: expected 2D arrays"
    assert X.shape[1] == lookback, f"{name}: lookback mismatch: {X.shape[1]} != {lookback}"
    assert Y.shape[1] == horizon, f"{name}: horizon mismatch: {Y.shape[1]} != {horizon}"
    assert len(X) > 0 and len(Y) > 0, f"{name}: empty windows (increase data or reduce lookback/horizon)"


def main():
    set_seed(SEED)

    # Basic compatibility checks
    assert LOOKBACK > 0 and HORIZON > 0
    assert VAL_SIZE > 0 and TEST_SIZE > 0
    assert LOOKBACK % PATCH_LEN == 0, "LOOKBACK must be divisible by PATCH_LEN for PatchTST"

    # =========================
    # LOAD + NORMALIZE
    # =========================
    series = load_series(DATA_PATH)
    series_norm, mean, std = normalize(series)

    # =========================
    # SPLIT
    # =========================
    train, val, test = split_series(series_norm, val_size=VAL_SIZE, test_size=TEST_SIZE)

    # Context-aware windows (IMPORTANT):
    # - val windows need last LOOKBACK points from train
    # - test windows need last LOOKBACK points from (train+val)
    val_ctx = np.concatenate([train[-LOOKBACK:], val])
    test_ctx = np.concatenate([np.concatenate([train, val])[-LOOKBACK:], test])

    Xtr, Ytr = make_windows(train, lookback=LOOKBACK, horizon=HORIZON)
    Xva, Yva = make_windows(val_ctx, lookback=LOOKBACK, horizon=HORIZON)
    Xte, Yte = make_windows(test_ctx, lookback=LOOKBACK, horizon=HORIZON)

    assert_shapes(Xtr, Ytr, LOOKBACK, HORIZON, "TRAIN")
    assert_shapes(Xva, Yva, LOOKBACK, HORIZON, "VAL")
    assert_shapes(Xte, Yte, LOOKBACK, HORIZON, "TEST")

    # =========================
    # TEACHERS
    # =========================
    teachers = [
        DLinearTeacher(lookback=LOOKBACK, horizon=HORIZON, epochs=TEACHER_EPOCHS),
        PatchTSTTeacher(lookback=LOOKBACK, horizon=HORIZON, epochs=TEACHER_EPOCHS, patch_len=PATCH_LEN),
    ]

    tr_preds, va_preds = [], []

    # Train teachers on TRAIN ONLY, evaluate on VAL windows
    for t in teachers:
        t.fit(train)
        tr_preds.append(t.predict(Xtr))  # KD targets on train windows
        va_preds.append(t.predict(Xva))  # weight computation on val windows

    # =========================
    # UNCERTAINTY-AWARE WEIGHTS
    # =========================
    weights = compute_weights(Yva, va_preds)
    print("Teacher weights:", weights)

    # =========================
    # STUDENT + KD TRAINING
    # =========================
    student = MLPStudent(lookback=LOOKBACK, horizon=HORIZON)

    student = train_student_kd(
        X=Xtr,
        Y=Ytr,
        teacher_preds=tr_preds,
        weights=weights,
        student=student,
        epochs=STUDENT_EPOCHS,
        alpha0=0.7,  # decays inside trainer
    )

    # =========================
    # EVALUATE ON TEST
    # =========================
    student.eval()
    with torch.no_grad():
        Xte_t = torch.tensor(Xte, dtype=torch.float32).unsqueeze(-1)
        preds_norm = student(Xte_t).cpu().numpy()

    # De-normalize for metrics on original price scale
    preds = preds_norm * std + mean
    Yte_denorm = Yte * std + mean

    print("\nOverall MAE:", mae(Yte_denorm, preds))
    print("Overall RMSE:", rmse(Yte_denorm, preds))
    print("Overall sMAPE:", smape(Yte_denorm, preds))

    print("\nHorizon-wise metrics:")
    hm = horizon_metrics(Yte_denorm, preds)
    for h, m in hm.items():
        print(h, m)

    print("\nTraining complete.")


if __name__ == "__main__":
    main()