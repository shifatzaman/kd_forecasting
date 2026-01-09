"""
Simplified baseline test - minimal configuration to verify setup works
"""
import numpy as np
import torch

from core.dataset import load_series, normalize, split_series, make_windows
from core.metrics import mae, rmse, smape
from teachers.dlinear import DLinearTeacher
from students.mlp import MLPStudent

# Minimal config
DATA_PATH = "data/Wfp_rice.csv"
LOOKBACK = 24  # Much smaller
HORIZON = 6
VAL_SIZE = 24
TEST_SIZE = 24
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

print("="*60)
print("BASELINE TEST - RICE DATASET")
print("="*60)

# Load
series = load_series(DATA_PATH)
print(f"✓ Loaded {len(series)} data points")
print(f"  Range: [{series.min():.2f}, {series.max():.2f}]")

# Normalize
series_norm, mean, std = normalize(series)
print(f"✓ Normalized (mean={mean:.2f}, std={std:.2f})")

# Split
train, val, test = split_series(series_norm, VAL_SIZE, TEST_SIZE)
print(f"✓ Split: train={len(train)}, val={len(val)}, test={len(test)}")

# Windows
val_ctx = np.concatenate([train[-LOOKBACK:], val])
test_ctx = np.concatenate([np.concatenate([train, val])[-LOOKBACK:], test])

Xtr, Ytr = make_windows(train, LOOKBACK, HORIZON)
Xva, Yva = make_windows(val_ctx, LOOKBACK, HORIZON)
Xte, Yte = make_windows(test_ctx, LOOKBACK, HORIZON)
print(f"✓ Windows: train={len(Xtr)}, val={len(Xva)}, test={len(Xte)}")

# Train simple teacher
print("\nTraining DLinear teacher (100 epochs)...")
teacher = DLinearTeacher(LOOKBACK, HORIZON, epochs=100)
teacher.fit(train)
teacher_preds_test = teacher.predict(Xte)
print("✓ Teacher trained")

# Evaluate teacher
teacher_preds_denorm = teacher_preds_test * std + mean
Yte_denorm = Yte * std + mean
teacher_mae = mae(Yte_denorm, teacher_preds_denorm)
print(f"  Teacher MAE: {teacher_mae:.4f}")

# Train simple student (NO KD, just supervised)
print("\nTraining simple MLP student (supervised only, 200 epochs)...")
student = MLPStudent(LOOKBACK, HORIZON, hidden_dim=64, n_blocks=1, dropout=0.1)
student = student.to('cpu')

X_train = torch.tensor(Xtr, dtype=torch.float32).unsqueeze(-1)
Y_train = torch.tensor(Ytr, dtype=torch.float32)

optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)

student.train()
for epoch in range(200):
    preds = student(X_train)
    loss = torch.abs(preds - Y_train).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"  Epoch {epoch+1}/200, Loss: {loss.item():.4f}")

print("✓ Student trained")

# Evaluate student
student.eval()
with torch.no_grad():
    student_preds_norm = student(torch.tensor(Xte, dtype=torch.float32).unsqueeze(-1)).cpu().numpy()

student_preds_denorm = student_preds_norm * std + mean
student_mae = mae(Yte_denorm, student_preds_denorm)
student_rmse = rmse(Yte_denorm, student_preds_denorm)
student_smape = smape(Yte_denorm, student_preds_denorm)

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Teacher (DLinear) MAE: {teacher_mae:.4f}")
print(f"Student (MLP)     MAE: {student_mae:.4f}")
print(f"Student (MLP)    RMSE: {student_rmse:.4f}")
print(f"Student (MLP)   sMAPE: {student_smape:.4f}")
print("="*60)

if student_mae < 2.0:
    print("✓ SUCCESS: MAE < 2.0")
else:
    print(f"✗ FAIL: MAE = {student_mae:.4f} (target: < 2.0)")
    print("\nTroubleshooting suggestions:")
    print("  1. Check if data is properly normalized")
    print("  2. Verify model architecture")
    print("  3. Try different learning rates")
    print("  4. Increase training epochs")
