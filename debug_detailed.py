"""
Detailed debugging script to understand what's happening
"""
import numpy as np
import torch
import pandas as pd

from core.dataset import load_series, normalize, split_series, make_windows

# Load rice data
DATA_PATH = "data/Wfp_rice.csv"
LOOKBACK = 24
HORIZON = 6
VAL_SIZE = 24
TEST_SIZE = 24

print("="*70)
print("DETAILED DEBUG ANALYSIS")
print("="*70)

# 1. Load and inspect raw data
df = pd.read_csv(DATA_PATH)
print(f"\n1. RAW DATA")
print(f"   Shape: {df.shape}")
print(f"   Columns: {df.columns.tolist()}")
print(f"   First 5 rows:")
print(df.head())
print(f"   Last 5 rows:")
print(df.tail())
print(f"   Price statistics:")
print(df['price'].describe())

# 2. Load series
series = load_series(DATA_PATH)
print(f"\n2. LOADED SERIES")
print(f"   Length: {len(series)}")
print(f"   Min: {series.min():.2f}, Max: {series.max():.2f}")
print(f"   Mean: {series.mean():.2f}, Std: {series.std():.2f}")
print(f"   First 10 values: {series[:10]}")

# 3. Normalize
series_norm, mean, std = normalize(series)
print(f"\n3. NORMALIZED SERIES")
print(f"   Mean used: {mean:.4f}")
print(f"   Std used: {std:.4f}")
print(f"   Normalized min: {series_norm.min():.4f}, max: {series_norm.max():.4f}")
print(f"   First 10 normalized: {series_norm[:10]}")

# 4. Split
train, val, test = split_series(series_norm, VAL_SIZE, TEST_SIZE)
print(f"\n4. SPLIT DATA")
print(f"   Train: {len(train)} samples")
print(f"   Val: {len(val)} samples")
print(f"   Test: {len(test)} samples")
print(f"   Train range: [{train.min():.4f}, {train.max():.4f}]")
print(f"   Test range: [{test.min():.4f}, {test.max():.4f}]")

# 5. Windows
Xtr, Ytr = make_windows(train, LOOKBACK, HORIZON)
test_ctx = np.concatenate([np.concatenate([train, val])[-LOOKBACK:], test])
Xte, Yte = make_windows(test_ctx, LOOKBACK, HORIZON)

print(f"\n5. WINDOWS")
print(f"   Training windows: {len(Xtr)}")
print(f"   Test windows: {len(Xte)}")
print(f"   Sample X shape: {Xtr[0].shape}")
print(f"   Sample Y shape: {Ytr[0].shape}")
print(f"\n   First training window:")
print(f"   X (input): {Xtr[0][:5]}...")  # First 5 of lookback
print(f"   Y (target): {Ytr[0]}")

# 6. Check if a simple persistence model works
print(f"\n6. PERSISTENCE MODEL (baseline)")
print(f"   Predicts: each horizon value = last input value")
persistence_preds = np.repeat(Xte[:, -1:], HORIZON, axis=1)
persistence_mae_norm = np.abs(persistence_preds - Yte).mean()
print(f"   Persistence MAE (normalized): {persistence_mae_norm:.4f}")

# Denormalize
persistence_preds_denorm = persistence_preds * std + mean
Yte_denorm = Yte * std + mean
persistence_mae = np.abs(persistence_preds_denorm - Yte_denorm).mean()
print(f"   Persistence MAE (original scale): {persistence_mae:.4f}")

# 7. Check mean prediction
mean_preds = np.full_like(Yte, 0.0)  # Predict normalized mean (0)
mean_mae_norm = np.abs(mean_preds - Yte).mean()
print(f"\n7. MEAN MODEL")
print(f"   Predicts: all values = 0 (normalized mean)")
print(f"   Mean MAE (normalized): {mean_mae_norm:.4f}")
mean_preds_denorm = mean_preds * std + mean
mean_mae = np.abs(mean_preds_denorm - Yte_denorm).mean()
print(f"   Mean MAE (original scale): {mean_mae:.4f}")

# 8. Check simple linear extrapolation
print(f"\n8. LINEAR EXTRAPOLATION")
print(f"   Predicts: trend from last 2 values")
linear_preds = []
for i in range(len(Xte)):
    last_val = Xte[i, -1]
    second_last = Xte[i, -2]
    trend = last_val - second_last
    pred = np.array([last_val + trend * (h+1) for h in range(HORIZON)])
    linear_preds.append(pred)
linear_preds = np.array(linear_preds)
linear_mae_norm = np.abs(linear_preds - Yte).mean()
print(f"   Linear MAE (normalized): {linear_mae_norm:.4f}")
linear_preds_denorm = linear_preds * std + mean
linear_mae = np.abs(linear_preds_denorm - Yte_denorm).mean()
print(f"   Linear MAE (original scale): {linear_mae:.4f}")

print(f"\n" + "="*70)
print(f"BASELINE COMPARISON")
print(f"="*70)
print(f"Persistence:  {persistence_mae:.4f}")
print(f"Mean:         {mean_mae:.4f}")
print(f"Linear trend: {linear_mae:.4f}")
print(f"\nTarget: < 2.0")
print(f"\nIf these baselines are > 2.0, the task may be very difficult")
print(f"or there may be data quality issues.")
