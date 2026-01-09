# Quick Fix and Test Guide

## The Problem

The initial run showed very high MAE (6.64 for rice, 21.93 for wheat). After investigation, I found several issues:

1. **Data augmentation bug**: Teachers were predicting on augmented data they never saw during training
2. **Model too complex**: For small datasets (315 rice, 238 wheat samples), the deep model was overfitting
3. **Lookback too large**: 60 timesteps was too much for the available data

## The Fix

I've simplified the configuration:
- Disabled data augmentation temporarily
- Reduced lookback: 60 → 48
- Simpler student model: 256 dim → 128 dim, 3 blocks → 2 blocks
- Fewer epochs to prevent overfitting
- Simplified loss function (MAE only)

## Step 1: Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Step 2: Test Baseline (RECOMMENDED)

First, run a simple baseline to verify everything works:

```bash
python3 test_baseline.py
```

This will:
- Train a single DLinear teacher
- Train a simple MLP student (supervised only, NO KD)
- Show MAE for both

**Expected output**: MAE should be in the range of 1.5-3.0 for rice.

If this works, proceed to Step 3. If not, there's a fundamental issue with the setup.

## Step 3: Run Full System

Once baseline works:

```bash
python3 run_both_datasets.py
```

This uses the simplified configuration with:
- 3 teachers (NLinear, DLinear, PatchTST)
- Knowledge distillation
- Bounded regime-aware weighting

## Expected Results

With the simplified configuration:
- **Rice**: MAE should be around 1.5-2.5
- **Wheat**: MAE should be around 2.0-3.0

## If MAE is Still Too High

Try these in order:

### 1. Further Reduce Lookback
Edit `run_both_datasets.py`:
```python
LOOKBACK = 36  # Reduce from 48
```

### 2. Use Only Best Teacher
Edit `run_both_datasets.py` to use just one teacher:
```python
teachers = [
    DLinearTeacher(LOOKBACK, HORIZON, TEACHER_EPOCHS),
]
```

### 3. Increase Training Time
```python
TEACHER_EPOCHS = 400
STUDENT_EPOCHS = 500
```

### 4. Simpler Student Model
```python
HIDDEN_DIM = 64
N_BLOCKS = 1
```

### 5. Direct Supervised Learning
Disable KD completely:
```python
ALPHA0 = 0.0  # No knowledge distillation, pure supervised
```

## Debug Checklist

- [ ] Dependencies installed (`pip list | grep torch`)
- [ ] Data files exist (`ls data/`)
- [ ] Baseline test passes (`python3 test_baseline.py`)
- [ ] MAE is reasonable (< 5.0 at least)

## Files Modified

The following files have been updated with fixes:
- `run_both_datasets.py` - Simplified hyperparameters, disabled augmentation
- `core/kd_trainer.py` - Simplified loss function

## Next Steps

Once you get MAE < 2.0 with the simplified configuration:
1. Gradually re-enable features (augmentation, deeper model, etc.)
2. Use hyperparameter sweep to optimize
3. Test on both datasets

## Contact

If MAE is still very high (> 5.0), there may be a data preprocessing issue. Check:
1. Data normalization is working correctly
2. Train/val/test splits are proper
3. Window creation is correct
