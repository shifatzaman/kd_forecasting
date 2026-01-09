# Model Improvements for MAE < 2.0

This document outlines the improvements made to achieve MAE < 2.0 for both rice and wheat datasets.

## Summary of Changes

### 1. Enhanced Student Architecture ([students/mlp.py](students/mlp.py))

**Previous**: Shallow 2-layer MLP (lookback → 128 → 64 → horizon)

**Improved**: Deep residual network with:
- **Residual blocks** with layer normalization for better gradient flow
- **Increased capacity**: 256 hidden dimensions, 3 residual blocks
- **Regularization**: Dropout (0.15) to prevent overfitting
- **Better architecture**: Input projection → Residual blocks → Multi-stage output head

**Impact**: More expressive model that can learn complex temporal patterns while maintaining stable training.

### 2. Improved Training Process ([core/kd_trainer.py](core/kd_trainer.py))

**New features**:
- **Combined loss function**: 70% MAE + 30% Huber loss for robustness to outliers
- **Learning rate scheduling**: Cosine annealing (lr decay from 5e-4 to 5e-6)
- **Gradient clipping**: Max norm of 1.0 for training stability
- **Early stopping**: Patience of 50 epochs with best model restoration
- **Weight decay**: L2 regularization (1e-4) to prevent overfitting

**Impact**: More stable training, better convergence, prevents overfitting.

### 3. Enhanced Teacher Ensemble

**Previous**: 2 teachers (DLinear, PatchTST)

**Improved**: 3 diverse teachers:
1. **NLinear** ([teachers/nlinear.py](teachers/nlinear.py)): Simple, effective for linear trends
2. **DLinear**: Trend-seasonal decomposition
3. **PatchTST**: Transformer-based for complex patterns

**Impact**: Better ensemble diversity leads to more robust knowledge distillation.

### 4. Data Augmentation ([core/dataset.py](core/dataset.py))

**New augmentation strategies**:
- **Gaussian noise injection**: σ = 0.015 (1.5% noise)
- **Random scaling**: Uniformly sampled from [0.97, 1.03]
- **3x training data**: Original + 2 augmented versions

**Impact**: Better generalization, reduced overfitting on small datasets.

### 5. Optimized Hyperparameters

| Parameter | Previous | Improved | Rationale |
|-----------|----------|----------|-----------|
| Lookback | 48 | 60 | More historical context |
| Teacher Epochs | 200 | 300 | Better teacher convergence |
| Student Epochs | 200 | 400 | More distillation time |
| Alpha0 | 0.5 | 0.6 | Balanced KD vs supervised |
| Learning Rate | 1e-3 | 5e-4 | More stable training |
| Hidden Dim | 128 | 256 | Increased model capacity |

## How to Run

### Single Dataset (Rice - default)
```bash
python3 run.py
```

### Both Datasets (Recommended)
```bash
python3 run_both_datasets.py
```

This will train and evaluate on both rice and wheat datasets, showing:
- Training progress for each teacher
- Teacher ensemble weights
- Final MAE, RMSE, sMAPE for each dataset
- Pass/fail status (MAE < 2.0)

## Expected Results

With these improvements, you should achieve:
- **Rice dataset**: MAE < 2.0
- **Wheat dataset**: MAE < 2.0

The improvements focus on:
1. **Model capacity**: Deeper, more expressive architecture
2. **Training stability**: Better optimization and regularization
3. **Ensemble quality**: Diverse teachers with better predictions
4. **Data efficiency**: Augmentation to make better use of limited data

## Key Files Modified

- [students/mlp.py](students/mlp.py) - Enhanced student architecture
- [core/kd_trainer.py](core/kd_trainer.py) - Improved training loop
- [teachers/nlinear.py](teachers/nlinear.py) - New NLinear teacher (NEW FILE)
- [core/dataset.py](core/dataset.py) - Added data augmentation
- [run.py](run.py) - Updated with optimized hyperparameters
- [run_both_datasets.py](run_both_datasets.py) - Evaluation script (NEW FILE)

## Installation

If you haven't installed dependencies yet:

```bash
pip3 install -r requirements.txt
```

## Next Steps (Optional Further Improvements)

If MAE is still not below 2.0, consider:

1. **Hyperparameter tuning**: Run [sweep.py](sweep.py) with expanded search space
2. **More augmentation**: Add time warping, magnitude warping
3. **Test-time augmentation**: Average predictions over multiple augmented inputs
4. **Ensemble student**: Train multiple students and average their predictions
5. **Cross-validation**: Use multiple train/val/test splits
6. **Feature engineering**: Add temporal features (day of week, month, etc.)
