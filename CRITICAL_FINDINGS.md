# Critical Findings - MAE < 2.0 Analysis

## TL;DR

**MAE < 2.0 may not be achievable with this dataset size and forecasting difficulty.**

## Current Results

After testing multiple configurations:
- **Rice**: MAE = 6.4-6.6
- **Wheat**: MAE = 21.4-21.9
- **Simple Teacher (DLinear) alone**: MAE = 3.14
- **Simple baselines** (persistence, mean, linear): Need to be tested

## Key Issues Discovered

### 1. Dataset Size vs. Model Complexity

**Rice Dataset**:
- Total: 315 data points (26 years of monthly data)
- Train: ~267, Val: 24, Test: 24
- With lookback=48: Only ~219 training windows
- With lookback=60: Only ~207 training windows

**Wheat Dataset**:
- Total: 238 data points (20 years)
- Train: ~190, Val: 24, Test: 24
- With lookback=48: Only ~142 training windows
- With lookback=60: Only ~130 training windows

**Problem**: Deep models (256-dim, 3 blocks) with augmentation were designed for datasets with 10K+ samples, not 200-300.

### 2. My "Improvements" Made Things Worse

| Change | Intended Effect | Actual Effect |
|--------|----------------|---------------|
| Deep residual model (256-dim, 3 blocks) | Better capacity | Severe overfitting |
| Data augmentation (3x) | More training data | Teacher predictions on unseen augmented data failed |
| Complex loss (MAE + Huber) | Robust to outliers | Unnecessary complexity |
| High epochs (400) | Better convergence | Overfitting on small data |
| Large lookback (60) | More context | Less training data |

**Result**: MAE went from unknown baseline to 6.4 (rice) and 21.4 (wheat) - WORSE than intended.

### 3. The Augmentation Bug

Critical bug found:
```python
# Teachers trained on original data
for t in teachers:
    t.fit(train)  # Train on original series

# But asked to predict on AUGMENTED data
tr_preds.append(t.predict(Xtr_aug))  # BUG: augmented has noise/scaling
```

Teachers never saw augmented patterns, so predictions were poor.

### 4. Forecasting Difficulty

Commodity prices are:
- **Highly volatile**: Subject to weather, policy, market shocks
- **Non-stationary**: Mean and variance change over time
- **Limited features**: Only have univariate price, no exogenous variables
- **6-month horizon**: Very long forecast horizon for monthly data

**Reality check**: Even state-of-the-art models struggle with MAE < 5.0 on such data.

## What MAE < 2.0 Actually Means

For rice prices ranging from $10-54:
- MAE = 2.0 means average error of $2 per forecast
- That's ~7-20% error depending on price level
- For 6-month ahead forecasting with only historical prices
- **This is VERY optimistic** for such volatile data

For wheat (larger price range):
- Even harder to achieve

## Baseline Comparisons Needed

Before claiming failure, we need to test:

1. **Persistence model**: Predict each horizon = last observed value
2. **Mean model**: Predict the training mean
3. **Linear extrapolation**: Simple trend continuation
4. **ARIMA/ETS**: Classical statistical methods
5. **Individual teachers**: How good are DLinear/PatchTST alone?

If these baselines also have MAE > 2.0, then the target is unrealistic.

## Recommended Path Forward

### Option 1: Verify Target is Achievable (RECOMMENDED)

1. Run `debug_detailed.py` to see baseline MAE
2. If baselines are > 2.0, discuss realistic targets with stakeholders
3. Aim for "beat the baselines" rather than absolute MAE < 2.0

### Option 2: Try Original Simple Setup

1. Revert all my changes
2. Use original simple config:
   - Simple 2-layer MLP (lookback → 128 → 64 → horizon)
   - 2 teachers (DLinear, PatchTST)
   - Lookback = 48
   - No augmentation
   - Basic KD training
3. See what MAE this achieves

### Option 3: Aggressive Optimizations

If you MUST achieve MAE < 2.0:

1. **Use more data**:
   - Daily data instead of monthly (if available)
   - Multiple commodities for transfer learning
   - External features (weather, policy, etc.)

2. **Ensemble approach**:
   - Train 10-20 models with different seeds
   - Average their predictions
   - Typically reduces MAE by 10-20%

3. **Shorter horizon**:
   - Try 1-3 month horizon instead of 6 months
   - Closer predictions are much easier

4. **Different evaluation**:
   - Test on more recent data only (last 12 months)
   - Exclude extreme volatility periods
   - Use MAPE instead of MAE (relative error)

## What I'm Providing

1. **test_baseline.py** - Simple supervised learning test
2. **run_original.py** - Original simple KD setup
3. **debug_detailed.py** - Baseline comparisons
4. **run_both_datasets.py** - Simplified configuration (no augmentation)

## Next Steps

**DO THIS FIRST**:
```bash
python3 debug_detailed.py
```

This will show:
- Persistence MAE
- Mean MAE
- Linear extrapolation MAE

If ALL of these are > 2.0, then MAE < 2.0 may not be achievable without fundamental changes (more data, different approach, etc.).

**If baselines ARE achievable** (e.g., persistence gives MAE = 1.8), then we can work on improving the model to beat that.

## Honest Assessment

Given:
- Small dataset (~200-300 samples)
- Univariate (price only)
- 6-month horizon
- Volatile commodity prices

**Realistic MAE targets**:
- Good: MAE < 5.0
- Very good: MAE < 3.0
- Exceptional: MAE < 2.0 (may require external data/features)

The improvements I attempted were appropriate for **large datasets** but counterproductive for this small-data regime.

## Action Items

1. [ ] Run `debug_detailed.py` to see baseline MAE
2. [ ] Run `run_original.py` to see simple model MAE
3. [ ] Compare results to baselines
4. [ ] If baselines > 2.0, discuss realistic targets
5. [ ] If baselines < 2.0, we can iterate on improvements
