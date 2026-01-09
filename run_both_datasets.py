import numpy as np
import torch

from core.dataset import (
    load_series,
    normalize,
    split_series,
    make_windows,
    compute_regime_scores,
    augment_data,
)
from core.metrics import mae, rmse, smape, horizon_metrics
from core.kd_trainer import train_student_kd

from teachers.dlinear import DLinearTeacher
from teachers.patchtst import PatchTSTTeacher
from teachers.nlinear import NLinearTeacher
from students.mlp import MLPStudent
from utils.uncertainty import compute_weights


# =========================
# CONFIG
# =========================
DATASETS = {
    "rice": "data/Wfp_rice.csv",
    "wheat": "data/Wfp_wheat.csv",
}

LOOKBACK = 60
HORIZON = 6
PATCH_LEN = 4

VAL_SIZE = 24
TEST_SIZE = 24

TEACHER_EPOCHS = 300
STUDENT_EPOCHS = 400
ALPHA0 = 0.6
LEARNING_RATE = 5e-4

# Student architecture
HIDDEN_DIM = 256
N_BLOCKS = 3
DROPOUT = 0.15

SEED = 42


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment(dataset_name, data_path):
    """Run the full experiment pipeline for a given dataset."""

    print("\n" + "=" * 60)
    print(f"RUNNING EXPERIMENT: {dataset_name.upper()}")
    print("=" * 60)

    set_seed(SEED)

    # -------------------------
    # Load & normalize
    # -------------------------
    series = load_series(data_path)
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
    # Data Augmentation
    # -------------------------
    print(f"Original training size: {len(Xtr)}")
    Xtr_aug, Ytr_aug = augment_data(Xtr, Ytr, noise_std=0.015, scale_range=(0.97, 1.03))
    print(f"Augmented training size: {len(Xtr_aug)}")

    # Expand regime scores for augmented data
    regime_scores_aug = np.tile(regime_scores, 3)

    # -------------------------
    # Teachers (ensemble of 3 diverse models)
    # -------------------------
    teachers = [
        NLinearTeacher(LOOKBACK, HORIZON, TEACHER_EPOCHS),
        DLinearTeacher(LOOKBACK, HORIZON, TEACHER_EPOCHS),
        PatchTSTTeacher(LOOKBACK, HORIZON, TEACHER_EPOCHS, PATCH_LEN),
    ]

    tr_preds, va_preds = [], []

    print("\nTraining teachers...")
    for i, t in enumerate(teachers):
        teacher_name = t.__class__.__name__.replace("Teacher", "")
        print(f"  [{i+1}/{len(teachers)}] {teacher_name}...", end=" ", flush=True)
        t.fit(train)
        tr_preds.append(t.predict(Xtr_aug))
        va_preds.append(t.predict(Xva))
        print("✓")

    weights = compute_weights(Yva, va_preds)
    print(f"\nTeacher weights: {weights}")

    # -------------------------
    # Student (Bounded Regime-Aware KD)
    # -------------------------
    student = MLPStudent(
        lookback=LOOKBACK,
        horizon=HORIZON,
        hidden_dim=HIDDEN_DIM,
        n_blocks=N_BLOCKS,
        dropout=DROPOUT,
    )

    print("\nTraining student with knowledge distillation...")
    student = train_student_kd(
        X=Xtr_aug,
        Y=Ytr_aug,
        teacher_preds=tr_preds,
        weights=weights,
        student=student,
        epochs=STUDENT_EPOCHS,
        regime_scores=regime_scores_aug,
        alpha0=ALPHA0,
        lr=LEARNING_RATE,
        patience=50,
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

    # Compute metrics
    test_mae = mae(Yte_denorm, preds)
    test_rmse = rmse(Yte_denorm, preds)
    test_smape = smape(Yte_denorm, preds)

    print("\n" + "-" * 60)
    print(f"RESULTS FOR {dataset_name.upper()}")
    print("-" * 60)
    print(f"Overall MAE:   {test_mae:.4f}")
    print(f"Overall RMSE:  {test_rmse:.4f}")
    print(f"Overall sMAPE: {test_smape:.4f}")

    print("\nHorizon-wise metrics:")
    for h, m in horizon_metrics(Yte_denorm, preds).items():
        print(f"  {h}: MAE={m['MAE']:.4f}, RMSE={m['RMSE']:.4f}, sMAPE={m['sMAPE']:.4f}")

    return {
        "dataset": dataset_name,
        "mae": test_mae,
        "rmse": test_rmse,
        "smape": test_smape,
    }


def main():
    results = []

    for dataset_name, data_path in DATASETS.items():
        result = run_experiment(dataset_name, data_path)
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for res in results:
        status = "✓ PASS" if res["mae"] < 2.0 else "✗ FAIL"
        print(f"{res['dataset'].upper():6s}: MAE = {res['mae']:.4f} {status}")

    print("\nTraining complete for all datasets.")


if __name__ == "__main__":
    main()
