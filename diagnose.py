import numpy as np
import torch

from core.dataset import load_series, normalize, split_series, make_windows
from teachers.nlinear import NLinearTeacher
from teachers.dlinear import DLinearTeacher
from core.metrics import mae

# Simple test
DATA_PATH = "data/Wfp_rice.csv"
LOOKBACK = 60
HORIZON = 6
VAL_SIZE = 24
TEST_SIZE = 24

# Load data
series = load_series(DATA_PATH)
print(f"Series shape: {series.shape}")
print(f"Series range: [{series.min():.2f}, {series.max():.2f}]")
print(f"Series mean: {series.mean():.2f}, std: {series.std():.2f}")

# Normalize
series_norm, mean, std = normalize(series)
print(f"\nNormalized range: [{series_norm.min():.2f}, {series_norm.max():.2f}]")
print(f"Mean: {mean:.2f}, Std: {std:.2f}")

# Split
train, val, test = split_series(series_norm, VAL_SIZE, TEST_SIZE)
print(f"\nTrain size: {len(train)}, Val: {len(val)}, Test: {len(test)}")

# Make windows
Xtr, Ytr = make_windows(train, LOOKBACK, HORIZON)
print(f"Training windows: {len(Xtr)}")

# Test teacher
print("\n" + "="*60)
print("Testing NLinear Teacher")
print("="*60)
teacher = NLinearTeacher(LOOKBACK, HORIZON, epochs=50)
teacher.fit(train)
preds = teacher.predict(Xtr[:5])  # First 5 predictions

print(f"Teacher predictions shape: {preds.shape}")
print(f"Teacher predictions (normalized):")
for i in range(min(3, len(preds))):
    print(f"  Sample {i}: {preds[i]}")
    print(f"  Actual {i}: {Ytr[i]}")
    print(f"  MAE: {np.abs(preds[i] - Ytr[i]).mean():.4f}")

# Denormalize and check
preds_denorm = preds * std + mean
Ytr_denorm = Ytr[:5] * std + mean
print(f"\nDenormalized predictions:")
for i in range(min(3, len(preds_denorm))):
    print(f"  Sample {i}: {preds_denorm[i]}")
    print(f"  Actual {i}: {Ytr_denorm[i]}")
    print(f"  MAE: {np.abs(preds_denorm[i] - Ytr_denorm[i]).mean():.4f}")

# Test student
print("\n" + "="*60)
print("Testing Student Model")
print("="*60)
from students.mlp import MLPStudent

student = MLPStudent(LOOKBACK, HORIZON, hidden_dim=256, n_blocks=3, dropout=0.15)
print(f"Student model created")

# Random prediction
student.eval()
with torch.no_grad():
    X_test = torch.tensor(Xtr[:3], dtype=torch.float32).unsqueeze(-1)
    student_preds = student(X_test).cpu().numpy()

print(f"Student predictions (before training): {student_preds.shape}")
print(f"Sample predictions:")
for i in range(len(student_preds)):
    print(f"  Sample {i}: mean={student_preds[i].mean():.4f}, std={student_preds[i].std():.4f}")
