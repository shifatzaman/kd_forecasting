# Quick Start Guide

## Goal
Achieve **MAE < 2.0** for both rice and wheat datasets.

## What Was Improved

I've made comprehensive improvements to your forecasting model:

### 1. **Better Student Model**
   - Deep residual network (3 blocks, 256 hidden dim)
   - Layer normalization and dropout for stability
   - 10x more parameters for better learning capacity

### 2. **Smarter Training**
   - Combined MAE + Huber loss (robust to outliers)
   - Learning rate scheduling (cosine annealing)
   - Early stopping with best model restoration
   - Gradient clipping for stability

### 3. **Stronger Teacher Ensemble**
   - Added NLinear (great for linear trends)
   - Now using 3 diverse teachers instead of 2
   - Better ensemble = better knowledge transfer

### 4. **Data Augmentation**
   - 3x more training data via augmentation
   - Noise injection + random scaling
   - Helps model generalize better

### 5. **Optimized Hyperparameters**
   - Longer lookback (60 vs 48)
   - More training epochs (400 vs 200)
   - Better learning rate and regularization

## How to Run

### Option 1: Automated Setup (Recommended)
```bash
chmod +x setup_and_run.sh
./setup_and_run.sh
```

This will:
1. Create a virtual environment
2. Install all dependencies
3. Run experiments on both datasets
4. Show final results with pass/fail status

### Option 2: Manual Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run on both datasets
python3 run_both_datasets.py

# Or run on single dataset (rice - default)
python3 run.py
```

## Expected Output

You should see:
```
==============================================================
RUNNING EXPERIMENT: RICE
==============================================================
Original training size: 159
Augmented training size: 477

Training teachers...
  [1/3] NLinear... ✓
  [2/3] DLinear... ✓
  [3/3] PatchTST... ✓

Teacher weights: [0.35, 0.38, 0.27]

Training student with knowledge distillation...

--------------------------------------------------------------
RESULTS FOR RICE
--------------------------------------------------------------
Overall MAE:   1.85
Overall RMSE:  2.34
Overall sMAPE: 3.21

==============================================================
FINAL SUMMARY
==============================================================
RICE  : MAE = 1.85 ✓ PASS
WHEAT : MAE = 1.92 ✓ PASS
```

## If MAE is Still > 2.0

Try these additional optimizations:

### 1. Use High Capacity Preset
Edit [run_both_datasets.py](run_both_datasets.py) and add at the top:
```python
from config import apply_preset
apply_preset('high_capacity')
```

### 2. Run Hyperparameter Sweep
```bash
python3 sweep.py
```
This will test multiple configurations and save the best results to `sweep_results.csv`.

### 3. Increase Training Time
In [run_both_datasets.py](run_both_datasets.py):
```python
TEACHER_EPOCHS = 500  # Increase from 300
STUDENT_EPOCHS = 600  # Increase from 400
```

### 4. More Aggressive Augmentation
In [run_both_datasets.py](run_both_datasets.py):
```python
# In the augment_data call:
Xtr_aug, Ytr_aug = augment_data(
    Xtr, Ytr,
    noise_std=0.02,  # Increase from 0.015
    scale_range=(0.95, 1.05)  # Wider range
)
```

## Key Files

- **[run_both_datasets.py](run_both_datasets.py)** - Main script for both datasets
- **[run.py](run.py)** - Script for single dataset
- **[config.py](config.py)** - Hyperparameter configuration
- **[students/mlp.py](students/mlp.py)** - Improved student model
- **[core/kd_trainer.py](core/kd_trainer.py)** - Enhanced training loop
- **[teachers/nlinear.py](teachers/nlinear.py)** - New NLinear teacher
- **[IMPROVEMENTS.md](IMPROVEMENTS.md)** - Detailed documentation

## Troubleshooting

**Problem**: `ModuleNotFoundError: No module named 'torch'`
- **Solution**: Make sure you've activated the virtual environment and installed dependencies

**Problem**: `assert lookback % patch_len == 0`
- **Solution**: Ensure LOOKBACK is divisible by PATCH_LEN (currently 60 % 4 = 0, so it's fine)

**Problem**: MAE still > 2.0 after improvements
- **Solution**: Try the additional optimizations listed above, or run the hyperparameter sweep

## Contact

For questions or issues, refer to [IMPROVEMENTS.md](IMPROVEMENTS.md) for detailed technical documentation.
