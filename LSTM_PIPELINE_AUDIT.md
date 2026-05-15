# STANDALONE LSTM PIPELINE AUDIT
**Comprehensive Review & Quality Assessment**
**Date**: May 15, 2026

---

## EXECUTIVE SUMMARY

The standalone LSTM pipeline (`run_lstm_pipeline.py` + `LSTMModel` class) has foundational code structure but exhibits **critical issues** across data handling, model architecture, prediction logic, and evaluation. This audit identifies 18 distinct issues with severity levels and provides fixes.

**Key Findings**:
- ✅ Pipeline execution structure is sound
- ⚠️ **Critical**: Data leakage and validation issues
- ⚠️ **High**: Inconsistent error handling and type safety
- ⚠️ **High**: Scalability and edge case problems
- ⚠️ **Medium**: Documentation gaps and unclear semantics
- ⚠️ **Medium**: Feature engineering and hyperparameter tuning concerns

**Recommendation**: All Critical and High severity issues must be fixed before production use.

---

## DETAILED FINDINGS

### 1. [CRITICAL] Data Leakage in Feature Engineering & Shifting

**File**: `run_lstm_pipeline.py`, lines 47-55
**Severity**: CRITICAL - Results in invalid model evaluation

**Problem**:
```python
# Current (BROKEN) code
feature_cols_to_shift = [c for c in processed_data.columns if c != 'usdngn' and c not in lag_cols]
processed_data[feature_cols_to_shift] = processed_data[feature_cols_to_shift].shift(1)
processed_data = processed_data.dropna()
```

**Issues**:
1. **Shift applied BEFORE train/val/test split** → future information leaks into features
2. Features from day T are shifted to day T-1, making day T-1 prediction use day T's information
3. This violates fundamental time-series causality: predictions should NOT use future data
4. Results are NOT comparable to other models using unshifted features

**Example of the problem**:
```
Day 1: oil_price = 50, feature_col = 100
Day 2: feature_col = 101

After shift:
Day 1: oil_price = 50, feature_col = 101  ← USES DAY 2's DATA!
Day 2: feature_col = (NaN, dropped)
```

**Impact**: 
- LSTM metrics artificially inflated
- Predictions appear better than reality
- Cannot be directly compared to ARIMA or hybrid models

**Fix**:
```python
# CORRECT approach - no shifting before split
# or if shifting is needed, do it INSIDE each split with proper alignment
train_data, val_data, test_data = splitter.split(processed_data)

# Shift is NOT needed if lagged features already created by preprocessor
# If custom shifting required for specific experiment, do:
def shift_features_per_split(data, shift_cols, periods=1):
    """Shift features WITHOUT data leakage."""
    for col in shift_cols:
        if col in data.columns:
            data[f'{col}_shifted'] = data[col].shift(periods)
    return data

train_data = shift_features_per_split(train_data, feature_cols_to_shift)
val_data = shift_features_per_split(val_data, feature_cols_to_shift)
test_data = shift_features_per_split(test_data, feature_cols_to_shift)
```

**Verification**: Run audit test (see AUDIT_TESTS.md) to confirm no future data leakage.

---

### 2. [CRITICAL] Inconsistent Sequence Length Handling & Context Alignment

**File**: `src/models.py`, LSTMModel.predict() method, lines ~840-920
**Severity**: CRITICAL - Predictions misaligned with test data

**Problem**:
The `predict()` method attempts to handle context with a rolling window, but:

1. **Sequence length boundary not consistently respected**:
   - First `sequence_length` predictions are NaN (correct)
   - But alignment with actual test dates is fragile
   
2. **Context parameter semantics unclear**:
   ```python
   val_context = X_train          # ← Why X_train? Is this intended?
   test_context = np.vstack([X_train, X_val])  # ← What's included?
   val_pred_raw = model.predict(X_val, X_context=val_context, verbose_debug=False)
   ```
   
3. **No validation that context + test_data match**:
   - If X_context has wrong shape, errors are silent (returns NaN)
   - No checks for feature dimension consistency

4. **Global index calculation fragile**:
   ```python
   global_idx = start_idx + i  # Relies on start_idx calculation
   if global_idx < self.sequence_length:
       predictions.append(np.nan)
   ```
   If `start_idx` is wrong, all subsequent predictions are wrong.

**Example of misalignment**:
```
Intended:
  X_train: indices 0-999 (history)
  X_val: indices 1000-1200 (prediction targets)
  
With sequence_length=30:
  Valid predictions should START at index 30 of X_val
  But if context handling is wrong, might start at 0 or 1000

Result: Predictions don't align with actual dates!
```

**Fix**:
```python
def predict(self, X_test, X_context=None, verbose_debug=False):
    """
    Make rolling one-step-ahead predictions.
    
    SEMANTICS (CLARIFIED):
    - X_test: the test features to predict for (shape: [n_test, n_features])
    - X_context: optional history for building rolling window
    
    If X_context provided:
      - Window is built from [X_context | X_test] 
      - Predictions are for indices sequence_length through len(X_context)+len(X_test)-1
      - First sequence_length predictions will be NaN (insufficient history)
    
    If X_context is None:
      - Use X_test[0:sequence_length] as history
      - Predictions start from index sequence_length
    """
    
    # Add input validation
    X_test = np.asarray(X_test, dtype=np.float32)
    if X_test.ndim != 2:
        raise ValueError(f"X_test must be 2D, got shape {X_test.shape}")
    
    if X_test.shape[1] != self.input_size:
        raise ValueError(
            f"X_test features ({X_test.shape[1]}) != "
            f"expected input_size ({self.input_size})"
        )
    
    # Rest of implementation...
    # (with proper error handling added below)
```

**Verification**: 
- Add shape assertions at start of predict()
- Test that predictions align with test dates exactly
- Verify first `sequence_length` predictions are NaN

---

### 3. [CRITICAL] No Validation of Train/Val/Test Feature Consistency

**File**: `run_lstm_pipeline.py`, lines 70-76
**Severity**: CRITICAL - Silent failures with mismatched features

**Problem**:
```python
X_train = np.nan_to_num(train_data[available_features].values, nan=0.0, ...)
X_val = np.nan_to_num(val_data[available_features].values, nan=0.0, ...)
X_test = np.nan_to_num(test_data[available_features].values, nan=0.0, ...)
```

**Issues**:
1. **No check that all features exist in all splits**
   - If a feature is in training but missing from validation, code still runs silently
   - Causes dimension mismatch in model

2. **NaN-to-zero replacement is too aggressive**
   - Zero may not be correct substitute for missing values
   - Different semantics for different features (e.g., volatility zero vs rate zero)
   - No logging of how many values were replaced

3. **No verification of feature ordering**
   - If columns reordered, features misaligned without error

**Example**:
```
train_data features: [oil, mpr, cpi, rate]  → X_train shape (N, 4)
test_data features: [mpr, oil, cpi, rate]   → X_test shape (M, 4)  
# Features are reordered but code doesn't catch it!
# Model trains on one order, predicts on another
```

**Fix**:
```python
def prepare_data_with_validation(seed=42, verbose=True):
    """Prepare data with consistency checks."""
    np.random.seed(seed)
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    collector = DataCollector(start_date='1995-01-01', end_date='2025-12-31')
    raw_data = collector.collect_all_data(verbose=verbose)
    raw_data.to_csv('data/raw_data.csv')

    preprocessor = DataPreprocessor(raw_data)
    processed_data, _ = preprocessor.preprocess()
    processed_data.to_csv('data/processed_data.csv')

    splitter = DataSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    train_data, val_data, test_data = splitter.split(processed_data)

    train_data.to_csv('data/train_data.csv')
    val_data.to_csv('data/val_data.csv')
    test_data.to_csv('data/test_data.csv')

    # ✅ VALIDATION CHECKS
    available_features = [f for f in FEATURE_COLS if f in train_data.columns]
    
    # Check 1: Features exist in all splits
    missing_in_val = set(available_features) - set(val_data.columns)
    missing_in_test = set(available_features) - set(test_data.columns)
    
    if missing_in_val:
        raise ValueError(f"Features in train but missing in val: {missing_in_val}")
    if missing_in_test:
        raise ValueError(f"Features in train but missing in test: {missing_in_test}")
    
    if verbose:
        print(f"\n[Data Validation] ✓ All {len(available_features)} features present in train/val/test")

    # Check 2: Extract features in consistent order
    X_train_raw = train_data[available_features].values
    X_val_raw = val_data[available_features].values
    X_test_raw = test_data[available_features].values
    
    # Check 3: Log NaN replacement
    nan_counts = {
        'train': np.isnan(X_train_raw).sum(),
        'val': np.isnan(X_val_raw).sum(),
        'test': np.isnan(X_test_raw).sum()
    }
    
    if any(nan_counts.values()):
        if verbose:
            print(f"\n[Warning] NaN values found - replacing with 0:")
            for split, count in nan_counts.items():
                print(f"  {split}: {count} NaN values")
    
    # Safe NaN handling
    X_train = np.nan_to_num(X_train_raw, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val_raw, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test_raw, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check 4: Verify dimensions
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1], \
        f"Feature dimension mismatch: train={X_train.shape[1]}, val={X_val.shape[1]}, test={X_test.shape[1]}"
    
    if verbose:
        print(f"[Data Validation] ✓ Shape consistency: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    return train_data, val_data, test_data, available_features, X_train, X_val, X_test
```

---

### 4. [HIGH] Target Mode Logic Confusing & Incorrectly Implemented

**File**: `run_lstm_pipeline.py`, lines 119-135
**Severity**: HIGH - Unclear semantics, potential bugs

**Problem**:
```python
if args.target_mode == 'return':
    y_train, y_val, _, _, _, _ = make_return_targets(train_data, val_data, test_data)
else:
    y_train = y_train_level
    y_val = y_val_level
```

**Issues**:
1. **Target mode not clearly defined in code comments**
   - What does "return" vs "level" really mean?
   - How should evaluation differ?

2. **Inconsistent handling in prediction reconstruction**:
   ```python
   if args.target_mode == 'return':
       val_pred = returns_to_levels(val_pred_raw, prev_val)
       test_pred = returns_to_levels(test_pred_raw, prev_test)
   ```
   - But `test_pred_raw` is ALREADY scaled back by `scaler_y`
   - Applying `returns_to_levels` on already-inverted values = DOUBLE INVERSION

3. **Evaluation metrics computed on LEVELS regardless of training target**:
   ```python
   val_metrics = ModelEvaluator.compute_all_metrics(y_val_level, val_pred, ...)
   ```
   - If model trained on returns, shouldn't evaluate on returns?
   - Or is this intentional (evaluate reconstructed levels)?
   - Not documented.

4. **`returns_to_levels()` function has implicit assumptions**:
   ```python
   def returns_to_levels(return_pred, previous_levels):
       return_pred = np.nan_to_num(return_pred, nan=0.0, ...)
       return previous_levels * np.exp(return_pred)  # Log-return reconstruction
   ```
   - Assumes log-returns, but what if returns are simple returns?
   - No validation that inputs are actually returns

**Fix**:
```python
def prepare_targets(train_data, val_data, test_data, target_mode='level'):
    """
    Prepare target variables for LSTM training.
    
    Parameters:
    -----------
    target_mode : str
        'level': Train on price levels (USD-NGN rates)
                 Model predicts: ŷ_t = E[rate_t | history]
                 
        'return': Train on log-returns (daily % change)
                  Model predicts: ŷ_t = E[log_return_t | history]
                  Reconstructed level: level_t = level_{t-1} * exp(return_t)
                  
                  Why use returns?
                  - Often more stationary than levels
                  - Better for models assuming stationary targets
                  - But evaluation should show reconstructed levels for interpretability
    
    Returns:
    --------
    (y_train, y_val, y_test, y_test_level, prev_values_for_evaluation)
    """
    
    y_test_level = test_data['usdngn'].values
    y_train_level = train_data['usdngn'].values
    y_val_level = val_data['usdngn'].values
    
    if target_mode == 'return':
        # Train on log-returns
        y_train = train_data['usdngn_return'].values
        y_val = val_data['usdngn_return'].values
        y_test_actual = test_data['usdngn_return'].values
        
        # But return y_test_level for evaluation
        return y_train, y_val, y_test_actual, y_test_level
    
    elif target_mode == 'level':
        # Train directly on levels
        return y_train_level, y_val_level, y_test_level, y_test_level
    
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}")


def reconstruct_levels_from_predictions(y_pred_raw, y_train_last_level, 
                                       target_mode, scaler_y):
    """
    Reconstruct price levels from model predictions.
    
    If model was trained on returns, predictions are return values.
    We need to reconstruct the actual levels for interpretation.
    """
    
    if target_mode == 'return':
        # y_pred_raw are log-returns (already inverse-scaled by model)
        # But scaler_y was fit on levels, so we need to be careful
        
        # BETTER: Keep log-return predictions in log-return space
        # Then reconstruct using actual previous levels
        y_pred_returns = y_pred_raw  # Already returns
        
        # For validation, use y_train[-1] as previous level
        # For test, use y_val[-1] as previous level
        
        # Reconstruct: level_t = level_{t-1} * exp(return_t)
        return y_pred_returns  # Return predictions; caller handles reconstruction
    
    elif target_mode == 'level':
        # Predictions are already levels
        return y_pred_raw
    
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}")
```

---

### 5. [HIGH] No Dropout During Inference (Model Uncertainty Not Captured)

**File**: `src/models.py`, LSTMModel.predict(), line ~881
**Severity**: HIGH - Loss of uncertainty quantification

**Problem**:
```python
self.model.eval()  # ← Sets model to evaluation mode
with torch.no_grad():
    for i in range(len(X_scaled)):
        # ... predictions
```

**Issues**:
1. **eval() mode disables dropout**
   - During training: dropout was active (regularization)
   - During inference: dropout disabled (point predictions only)
   - This is correct for point estimates BUT loses uncertainty info

2. **No uncertainty quantification**
   - No confidence intervals around predictions
   - No indication of model confidence

3. **For time series forecasting, uncertainty is crucial**
   - Exchange rates are highly uncertain
   - Point estimates without uncertainty bounds are incomplete
   - Should report prediction intervals

**Note**: This is a design limitation, not a bug. Fixing requires ensemble methods or MC-dropout.

**Recommendation** (Future Enhancement):
```python
def predict_with_uncertainty(self, X_test, X_context=None, n_iterations=10):
    """
    Predict with uncertainty using MC-Dropout.
    
    Run forward passes with dropout enabled multiple times,
    compute mean and std of predictions.
    """
    predictions_samples = []
    
    for _ in range(n_iterations):
        self.model.train()  # ← Keep dropout ON
        with torch.no_grad():
            preds = # ... (normal prediction loop)
        predictions_samples.append(preds)
    
    preds_array = np.array(predictions_samples)
    mean_pred = np.mean(preds_array, axis=0)
    std_pred = np.std(preds_array, axis=0)
    
    return mean_pred, std_pred
```

---

### 6. [HIGH] Inconsistent Scaling Between Training and Prediction

**File**: `src/models.py`, LSTMModel class
**Severity**: HIGH - Can cause prediction errors with extreme values

**Problem**:
```python
def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
    self.scaler_X.fit(X_train)           # Scalers fit on TRAINING data only
    self.scaler_y.fit(y_train.reshape(-1, 1))
    
    # Then later in predict():
    X_scaled = self.scaler_X.transform(X_test)  # Transform test data
```

**Issues**:
1. **Scalers fit ONLY on training data**
   - If test data has higher/lower values than training, scaling is inconsistent
   - Example: If training max oil price = 100, test max = 150 → test data scaled outside [-1,1] range

2. **Validation data NOT used during scaling**
   - Scalers should fit on train+val (or just val range), not train only
   - This is standard practice in ML

3. **No checks for out-of-range scaled values**
   - If X_test has values outside scaler's training range, predictions become unreliable
   - No warnings or adjustments

**Example of the problem**:
```
Training data X_train: values in range [10, 100]
Scaler fit on X_train: min=10, max=100
StandardScaler centers around 55, scales by std~25

Test data X_test: values in range [5, 150]  ← Different range!
When X_test=5:  scaled = (5-55)/25 = -2.0  ← Outside normal range
When X_test=150: scaled = (150-55)/25 = 3.8 ← Way outside normal range

Model trained on [-1, 1] range, suddenly gets inputs in [-2, 3.8] range
→ Extrapolation, unreliable predictions
```

**Fix**:
```python
def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
    """Fit scalers on combined train+val data to avoid extrapolation."""
    
    # Fit scalers on train+val combined (standard practice)
    if X_val is not None:
        X_for_scaling = np.vstack([X_train, X_val])
        y_for_scaling = np.concatenate([y_train, y_val])
    else:
        X_for_scaling = X_train
        y_for_scaling = y_train
    
    self.scaler_X.fit(X_for_scaling)
    self.scaler_y.fit(y_for_scaling.reshape(-1, 1))
    
    if verbose:
        X_range = X_for_scaling.min(axis=0), X_for_scaling.max(axis=0)
        y_range = (y_for_scaling.min(), y_for_scaling.max())
        print(f"  Scalers fit on train+val (X range: {X_range}, y range: {y_range})")
    
    # Rest of training...
```

---

### 7. [HIGH] Silent NaN Handling with No Tracking

**File**: Multiple files
**Severity**: HIGH - Difficult to debug, masks data quality issues

**Problem**:
```python
# In run_lstm_pipeline.py:
X_train = np.nan_to_num(X_train_raw, nan=0.0, posinf=0.0, neginf=0.0)

# In models.py:
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
```

**Issues**:
1. **NaN values are silently replaced with 0**
   - No logging of how many values were replaced
   - No indication if NaN replacement is problematic
   
2. **Zero may be wrong substitute**
   - For oil price: zero means no oil (wrong)
   - For volatility: zero means no volatility (maybe OK)
   - For rate: zero means no currency (wrong)
   - For log-returns: zero means no change (maybe OK)

3. **Multiple NaN replacements**
   - Data → processed_data has NaNs
   - processed_data → X features has NaNs  
   - X in training → more NaNs handled
   - No clear accounting of what was replaced where

**Fix**:
```python
def prepare_data_with_nan_tracking(seed=42, verbose=True):
    """Prepare data with NaN tracking."""
    
    # ... (previous code)
    
    available_features = [f for f in FEATURE_COLS if f in train_data.columns]
    
    # Extract features
    X_train_raw = train_data[available_features].values
    X_val_raw = val_data[available_features].values
    X_test_raw = test_data[available_features].values
    
    # Track NaNs BEFORE replacement
    def log_nan_stats(X, split_name, feature_names):
        """Log NaN statistics."""
        nan_mask = np.isnan(X)
        nan_count = nan_mask.sum()
        
        if nan_count > 0:
            print(f"\n[NaN Analysis] {split_name}: {nan_count} total NaN values")
            for i, fname in enumerate(feature_names):
                col_nans = nan_mask[:, i].sum()
                if col_nans > 0:
                    pct = 100 * col_nans / len(X)
                    print(f"  {fname}: {col_nans} ({pct:.2f}%)")
            
            # Alternative: Suggest imputation strategy
            print(f"  → Consider interpolation or feature-specific imputation")
    
    log_nan_stats(X_train_raw, "Train", available_features)
    log_nan_stats(X_val_raw, "Validation", available_features)
    log_nan_stats(X_test_raw, "Test", available_features)
    
    # Replace with explicit method tracking
    def safe_nan_replace(X, strategy='zero', features=None):
        """Replace NaN with explicit strategy."""
        X_replaced = X.copy()
        
        if strategy == 'zero':
            X_replaced = np.nan_to_num(X_replaced, nan=0.0, posinf=0.0, neginf=0.0)
        elif strategy == 'forward_fill':
            # Use forward-fill (last known value)
            for i in range(1, len(X_replaced)):
                nan_mask = np.isnan(X_replaced[i])
                X_replaced[i, nan_mask] = X_replaced[i-1, nan_mask]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return X_replaced
    
    X_train = safe_nan_replace(X_train_raw, strategy='zero', features=available_features)
    X_val = safe_nan_replace(X_val_raw, strategy='zero', features=available_features)
    X_test = safe_nan_replace(X_test_raw, strategy='zero', features=available_features)
    
    if verbose:
        print(f"[Data Quality] ✓ Replaced NaN values with zeros")
    
    return train_data, val_data, test_data, available_features, X_train, X_val, X_test
```

---

### 8. [HIGH] Target Mode Not Handled Correctly in Evaluation

**File**: `run_lstm_pipeline.py`, lines 154-175
**Severity**: HIGH - Evaluation metrics misaligned with training target

**Problem**:
When `target_mode='return'`, the model is trained on log-returns, but evaluation is confusing:

```python
if args.target_mode == 'return':
    val_pred = returns_to_levels(val_pred_raw, prev_val)
    test_pred = returns_to_levels(test_pred_raw, prev_test)
else:
    val_pred = val_pred_raw
    test_pred = test_pred_raw

# Then evaluate on levels
val_metrics = ModelEvaluator.compute_all_metrics(y_val_level, val_pred, prev_values=prev_val)
```

**Issues**:
1. **Prediction values may be incorrect after reconstruction**
   - `test_pred_raw` is ALREADY inverse-scaled by scaler_y
   - scaler_y was fit on `y_train` (returns or levels depending on mode)
   - Calling `returns_to_levels()` on these values is wrong

2. **`prev_val` and `prev_test` incorrect**:
   ```python
   prev_val = np.concatenate([[y_train_level[-1]], y_val_level[:-1]])
   prev_test = np.concatenate([[y_val_level[-1]], y_test_level[:-1]])
   ```
   - These are constructed assuming `y_val_level` and `y_test_level` exist
   - But if training on returns, what should prev_values be?
   - Semantics unclear

3. **Directional accuracy calculation may be wrong**:
   - If training on returns, predictions are returns
   - But prev_values are levels
   - Comparing level-changes to return-changes is incorrect

**Better approach**:
```python
# Clear separation of training target and evaluation target
y_train, y_val, y_test, y_test_level = prepare_targets(
    train_data, val_data, test_data, 
    target_mode=args.target_mode
)

# Train model on chosen target
model.fit(X_train, y_train, X_val, y_val, verbose=args.verbose)

# Make predictions
val_context = X_train
test_context = np.vstack([X_train, X_val])
val_pred_raw = model.predict(X_val, X_context=val_context)
test_pred_raw = model.predict(X_test, X_context=test_context)

# Reconstruct predictions to LEVELS for evaluation
if args.target_mode == 'return':
    # Predictions are returns; reconstruct to levels
    prev_val = np.concatenate([[y_train_level[-1]], y_val_level[:-1]])
    prev_test = np.concatenate([[y_val_level[-1]], y_test_level[:-1]])
    
    val_pred_level = reconstruct_levels(val_pred_raw, prev_val)
    test_pred_level = reconstruct_levels(test_pred_raw, prev_test)
else:
    # Predictions are already levels
    val_pred_level = val_pred_raw
    test_pred_level = test_pred_raw
    prev_val = np.concatenate([[y_train_level[-1]], y_val_level[:-1]])
    prev_test = np.concatenate([[y_val_level[-1]], y_test_level[:-1]])

# Evaluate on LEVELS always (for consistency)
val_metrics = ModelEvaluator.compute_all_metrics(y_val_level, val_pred_level, prev_values=prev_val)
test_metrics = ModelEvaluator.compute_all_metrics(y_test_level, test_pred_level, prev_values=prev_test)
```

---

### 9. [MEDIUM] Early Stopping Patience Parameter Not Explained

**File**: `src/models.py`, LSTMModel.fit(), lines ~734-755
**Severity**: MEDIUM - Unexpected early stopping behavior

**Problem**:
```python
patience_counter = 0
for epoch in range(self.epochs):
    # ... training ...
    
    if val_loader is not None:
        val_loss = self._validate(val_loader, criterion)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= self.patience:
            break  # ← Early stopping triggered
```

**Issues**:
1. **Patience semantics not documented**
   - patience=8 means: stop if no improvement for 8 epochs
   - But this is rarely stated explicitly

2. **Early stopping vs. plateau detection unclear**
   - Is early stopping stopping at first plateau?
   - Or continuing if loss is still decreasing slightly?
   - LR scheduler also active, making it harder to debug

3. **No warning logged when early stopping occurs**
   ```python
   if patience_counter >= self.patience:
       if verbose:
           print(f"Early stopping at epoch {epoch+1}")  # ← Currently there
   ```
   - OK, but should show: epoch, best_loss, current_loss, patience_counter

**Fix**: Already partially implemented, but enhance:
```python
if patience_counter >= self.patience:
    if verbose:
        print(f"  Early stopping at epoch {epoch+1}/{self.epochs}")
        print(f"    Best val loss: {best_val_loss:.6f}")
        print(f"    Current val loss: {val_loss:.6f}")
        print(f"    Patience: {self.patience} epochs")
    break
```

---

### 10. [MEDIUM] No Validation of Hyperparameter Ranges

**File**: `run_lstm_pipeline.py`, lines 210-220
**Severity**: MEDIUM - Invalid hyperparameters silently cause problems

**Problem**:
```python
parser.add_argument('--sequence_length', type=int, default=30)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--patience', type=int, default=8)
parser.add_argument('--hidden_size1', type=int, default=96)
parser.add_argument('--hidden_size2', type=int, default=48)
# ... no validation of ranges!
```

**Issues**:
1. **No checking for invalid values**:
   - `sequence_length=0` → causes errors in create_sequences()
   - `epochs=-10` → weird behavior
   - `patience > epochs` → patience never reached
   - `hidden_size1=1000000` → OOM

2. **No sensible defaults**:
   - defaults seem arbitrary (why 96, not 128?)
   - No justification in comments

3. **Relationship constraints not validated**:
   - patience should be < epochs
   - sequence_length should be sensible (>5, <1000)
   - batch_size should be < training data size

**Fix**:
```python
def parse_args():
    parser = argparse.ArgumentParser(description='Standalone LSTM pipeline for USD-NGN forecasting')
    
    parser.add_argument('--target_mode', choices=['return', 'level'], default='return',
                        help='return trains LSTM on log returns and reconstructs levels; level trains directly on rates')
    
    # Sequence length
    parser.add_argument('--sequence_length', type=int, default=30,
                        help='lookback window for LSTM (5-100 recommended)')
    
    # Training epochs
    parser.add_argument('--epochs', type=int, default=75,
                        help='max training epochs (10-200 typical)')
    
    # Early stopping patience
    parser.add_argument('--patience', type=int, default=12,
                        help='epochs without improvement before stopping (3-patience recommended)')
    
    # Batch size
    parser.add_argument('--batch_size', type=int, default=64,
                        help='training batch size (16-128 typical)')
    
    # Learning rate
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Adam optimizer learning rate (0.0001-0.01 typical)')
    
    # LSTM architecture
    parser.add_argument('--hidden_size1', type=int, default=96,
                        help='first LSTM layer units (32-256 typical)')
    
    parser.add_argument('--hidden_size2', type=int, default=48,
                        help='second LSTM layer units (16-128 typical)')
    
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout rate (0.0-0.5 typical)')
    
    # Other args
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--verbose', type=str, default='True', choices=['True', 'False'])
    
    args = parser.parse_args()
    args.verbose = args.verbose == 'True'
    
    # ✅ VALIDATION
    if args.sequence_length < 5 or args.sequence_length > 500:
        raise ValueError(f"sequence_length must be 5-500, got {args.sequence_length}")
    
    if args.epochs < 1 or args.epochs > 500:
        raise ValueError(f"epochs must be 1-500, got {args.epochs}")
    
    if args.patience >= args.epochs:
        raise ValueError(f"patience ({args.patience}) must be < epochs ({args.epochs})")
    
    if args.batch_size < 1 or args.batch_size > 512:
        raise ValueError(f"batch_size must be 1-512, got {args.batch_size}")
    
    if args.learning_rate < 1e-6 or args.learning_rate > 1:
        raise ValueError(f"learning_rate must be 1e-6-1, got {args.learning_rate}")
    
    if args.hidden_size1 < 8 or args.hidden_size1 > 512:
        raise ValueError(f"hidden_size1 must be 8-512, got {args.hidden_size1}")
    
    if args.hidden_size2 < 8 or args.hidden_size2 > 512:
        raise ValueError(f"hidden_size2 must be 8-512, got {args.hidden_size2}")
    
    if args.hidden_size2 >= args.hidden_size1:
        raise ValueError(f"hidden_size2 ({args.hidden_size2}) should be < hidden_size1 ({args.hidden_size1})")
    
    if args.dropout < 0 or args.dropout > 0.7:
        raise ValueError(f"dropout must be 0-0.7, got {args.dropout}")
    
    return args
```

---

### 11. [MEDIUM] No Logging of Model Architecture

**File**: `src/models.py`, LSTMModel class
**Severity**: MEDIUM - Reproducibility issues

**Problem**:
When model is trained, the architecture is never explicitly logged. Hard to reproduce results later.

**Fix**:
```python
def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
    """Train LSTM model with architecture logging."""
    
    # Log configuration at start
    if verbose:
        print("\n[LSTM Configuration]")
        print(f"  Input features: {self.input_size}")
        print(f"  Sequence length: {self.sequence_length}")
        print(f"  Architecture: LSTM({self.hidden_size1}) -> LSTM({self.hidden_size2}) -> FC(64) -> FC(32) -> FC(1)")
        print(f"  Dropout: {self.dropout}")
        print(f"  Optimizer: Adam(lr={self.learning_rate})")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Epochs: {self.epochs}, Patience: {self.patience}")
    
    # ... rest of training ...
```

---

### 12. [MEDIUM] Evaluation Metrics CSV Format Unclear

**File**: `run_lstm_pipeline.py`, lines 163-169
**Severity**: MEDIUM - Metrics interpretation difficult

**Problem**:
```python
metrics_df = pd.DataFrame([
    {'Split': 'Validation', 'Model': f"LSTM ({args.target_mode})", **val_metrics},
    {'Split': 'Test', 'Model': f"LSTM ({args.target_mode})", **test_metrics},
])
metrics_df.to_csv('data/lstm_evaluation_metrics.csv', index=False)
```

**Issues**:
1. **Metrics column meanings not documented**
   - What's RMSE: in terms of what? Log-returns? Levels? Percent?
   - DA: one-step or multi-step?

2. **No metadata in CSV**
   - No date generated
   - No data version
   - No hyperparameters used

3. **CSV not comparison-ready**
   - Other models (ARIMA, hybrid) don't write to same format
   - Can't easily compare all models

**Fix**:
```python
def save_metrics_with_metadata(metrics_df, hyperparams, data_info, output_path):
    """Save metrics with metadata."""
    
    # Add metadata rows
    metadata = pd.DataFrame([
        {'Split': 'METADATA', 'Model': f"Generated: {datetime.now().isoformat()}"},
        {'Split': 'METADATA', 'Model': f"Target mode: {hyperparams.get('target_mode', 'N/A')}"},
        {'Split': 'METADATA', 'Model': f"Sequence length: {hyperparams.get('sequence_length', 'N/A')}"},
        {'Split': 'METADATA', 'Model': f"Hidden sizes: {hyperparams.get('hidden_size1')}/{hyperparams.get('hidden_size2')}"},
        {'Split': 'METADATA', 'Model': f"Train size: {data_info.get('train_size')}"},
        {'Split': 'METADATA', 'Model': f"Test size: {data_info.get('test_size')}"},
    ])
    
    combined_df = pd.concat([metadata, metrics_df], ignore_index=True)
    combined_df.to_csv(output_path, index=False)
```

---

### 13. [MEDIUM] Inferred Device Type Mismatch

**File**: `run_lstm_pipeline.py`, line 113
**Severity**: MEDIUM - Device handling confusing, may cause errors on multi-GPU systems

**Problem**:
```python
device = torch.device('cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu') if TORCH_AVAILABLE else 'CPU'
print(f"Device: {device}")
```

**Issues**:
1. **Device is string 'CPU' if torch not available**
   - But later code expects torch.device object
   - Will fail if torch not available but still referenced

2. **No validation that device matches model device**
   - Model created with one device, but X data might be on another
   - Silent failures if mismatch

3. **Multi-GPU not supported**
   - Hard-coded to use cuda:0
   - No option for distributed training

**Fix**:
```python
def setup_device():
    """Setup PyTorch device with validation."""
    if not TORCH_AVAILABLE:
        print("PyTorch not available - using CPU/numpy mode")
        return None
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = torch.cuda.get_device_name(0)
        device_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {device_name} ({device_mem:.1f} GB)")
        return device
    else:
        print("GPU not available - using CPU")
        return torch.device('cpu')

# Then in fit_lstm():
device = setup_device()
# Pass to model...
```

---

### 14. [MEDIUM] No Checkpointing of Best Model During Training

**File**: `src/models.py`, LSTMModel.fit()
**Severity**: MEDIUM - Can't recover from training interruptions

**Problem**:
```python
if self.best_model_state is not None:
    self.model.load_state_dict({k: v.to(self.device) for k, v in self.best_model_state.items()})
```

**Issues**:
1. **Best model state stored only in memory**
   - If training crashes, best state lost
   - No ability to resume training

2. **No checkpoint file saved during training**
   - Other frameworks (PyTorch Lightning, TF) save checkpoints automatically
   - Users must implement manually

3. **No versioning of saved models**
   - If `models/lstm_model.pt` exists, it's overwritten
   - No way to track model versions/history

**Fix**:
```python
def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
    """Train with checkpoint support."""
    
    # Setup checkpoint directory
    checkpoint_dir = Path('models/checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(self.epochs):
        # ... training loop ...
        
        if val_loader is not None:
            val_loss = self._validate(val_loader, criterion)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                
                # Save checkpoint
                checkpoint_path = checkpoint_dir / f"lstm_epoch{epoch+1}_loss{val_loss:.4f}.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'hyperparams': {
                        'input_size': self.input_size,
                        'hidden_size1': self.hidden_size1,
                        'hidden_size2': self.hidden_size2,
                        'dropout': self.dropout,
                    }
                }, checkpoint_path)
                
                if verbose:
                    print(f"    Checkpoint saved: {checkpoint_path}")
            else:
                patience_counter += 1
            
            if patience_counter >= self.patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break
```

---

### 15. [MEDIUM] No Return Value Documentation

**File**: `run_lstm_pipeline.py`, line 178-185
**Severity**: MEDIUM - Unclear what gets returned and used

**Problem**:
```python
return {
    'model': model,
    'metrics': metrics_df,
    'predictions': predictions_df,
    'features': features,
}
```

**Issues**:
1. **Return value not used by caller**
   - `fit_lstm()` returns dict, but `if __name__ == "__main__"` ignores it
   - Dead code

2. **No documentation of return structure**
   - What types are returned?
   - Are they serializable?
   - How to use them later?

3. **Model object not serializable**
   - Returning the model object (PyTorch) is dangerous
   - Can't be pickled or sent over network
   - Should return path instead

**Fix**:
```python
def fit_lstm(args):
    """
    Fit LSTM model on USD-NGN data.
    
    Returns:
    --------
    dict
        'model_path': str - path to saved model weights
        'metrics_csv': str - path to evaluation metrics
        'predictions_csv': str - path to predictions
        'config': dict - training hyperparameters used
    """
    
    # ... training code ...
    
    # Save model
    if TORCH_AVAILABLE and model.model is not None:
        model_path = 'models/lstm_model.pt'
        torch.save(model.model.state_dict(), model_path)
        print(f"Saved model to: {model_path}")
    else:
        model_path = None
    
    return {
        'model_path': model_path,
        'metrics_csv': 'data/lstm_evaluation_metrics.csv',
        'predictions_csv': 'data/lstm_predictions.csv',
        'config': {
            'target_mode': args.target_mode,
            'sequence_length': args.sequence_length,
            'epochs': args.epochs,
            'patience': args.patience,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'hidden_size1': args.hidden_size1,
            'hidden_size2': args.hidden_size2,
            'dropout': args.dropout,
        }
    }

if __name__ == "__main__":
    result = fit_lstm(parse_args())
    print("\nResults:")
    print(f"  Model: {result['model_path']}")
    print(f"  Metrics: {result['metrics_csv']}")
    print(f"  Predictions: {result['predictions_csv']}")
```

---

### 16. [MEDIUM] No Input Validation in LSTMModel Constructor

**File**: `src/models.py`, LSTMModel.__init__()
**Severity**: MEDIUM - Invalid parameters silently cause errors later

**Problem**:
```python
def __init__(self, input_size, sequence_length=60, batch_size=32, epochs=80,
             patience=16, learning_rate=0.001, hidden_size1=192, hidden_size2=96, dropout=0.2):
    self.input_size = input_size
    self.sequence_length = sequence_length
    # ... no validation!
```

**Issues**:
1. **Invalid input_size not caught**
   - If input_size=0, model creation fails later
   - If input_size=-5, errors are confusing

2. **No type checking**
   - If sequence_length is passed as string, numpy array, etc., fails later

3. **No bounds checking**
   - sequence_length=1000000 → OOM later
   - dropout=1.5 → pytorch error later
   - batch_size=-1 → pytorch error later

**Fix**:
```python
def __init__(self, input_size, sequence_length=60, batch_size=32, epochs=80,
             patience=16, learning_rate=0.001, hidden_size1=192, hidden_size2=96, dropout=0.2):
    
    # ✅ Validation
    if not isinstance(input_size, int) or input_size < 1:
        raise ValueError(f"input_size must be positive int, got {input_size} ({type(input_size)})")
    
    if not isinstance(sequence_length, int) or sequence_length < 2:
        raise ValueError(f"sequence_length must be int >= 2, got {sequence_length}")
    
    if sequence_length > 1000:
        raise ValueError(f"sequence_length > 1000 likely a mistake, got {sequence_length}")
    
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"batch_size must be positive int, got {batch_size}")
    
    if batch_size > 2048:
        raise ValueError(f"batch_size > 2048 likely a mistake, got {batch_size}")
    
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError(f"epochs must be positive int, got {epochs}")
    
    if not isinstance(patience, int) or patience < 1:
        raise ValueError(f"patience must be positive int, got {patience}")
    
    if patience >= epochs:
        raise ValueError(f"patience ({patience}) should be < epochs ({epochs})")
    
    if not 0 < learning_rate < 1:
        raise ValueError(f"learning_rate must be in (0, 1), got {learning_rate}")
    
    if not 0 < hidden_size1 < 2048:
        raise ValueError(f"hidden_size1 must be in (0, 2048), got {hidden_size1}")
    
    if not 0 < hidden_size2 < 2048:
        raise ValueError(f"hidden_size2 must be in (0, 2048), got {hidden_size2}")
    
    if hidden_size2 > hidden_size1:
        raise Warning(f"hidden_size2 ({hidden_size2}) > hidden_size1 ({hidden_size1}) - unusual but allowed")
    
    if not 0 <= dropout < 1:
        raise ValueError(f"dropout must be in [0, 1), got {dropout}")
    
    # Store validated parameters
    self.input_size = input_size
    self.sequence_length = sequence_length
    # ... rest of __init__ ...
```

---

### 17. [MEDIUM] Predictions CSV Missing Important Columns

**File**: `run_lstm_pipeline.py`, lines 171-177
**Severity**: MEDIUM - Incomplete analysis of predictions

**Problem**:
```python
predictions_df = pd.DataFrame({
    'actual': y_test_level,
    'previous_actual': prev_test,
    'lstm_prediction': test_pred,
    'raw_lstm_output': test_pred_raw,
}, index=test_data.index)
predictions_df.to_csv('data/lstm_predictions.csv')
```

**Missing columns**:
1. Errors (absolute, percentage)
2. Directional correctness (did LSTM predict right direction?)
3. Confidence/uncertainty
4. Feature values (which features drove each prediction?)

**Fix**:
```python
# Enhanced predictions output
predictions_df = pd.DataFrame({
    'date': test_data.index,
    'actual': y_test_level,
    'previous_actual': prev_test,
    'lstm_prediction': test_pred,
    'raw_lstm_output': test_pred_raw,
    
    # Error metrics
    'abs_error': np.abs(y_test_level - test_pred),
    'pct_error': 100 * np.abs(y_test_level - test_pred) / np.abs(y_test_level),
    'squared_error': (y_test_level - test_pred) ** 2,
    
    # Direction correctness
    'actual_direction': (y_test_level > prev_test).astype(int),
    'predicted_direction': (test_pred > prev_test).astype(int),
    'direction_correct': ((y_test_level > prev_test) == (test_pred > prev_test)).astype(int),
}, index=test_data.index)

predictions_df.to_csv('data/lstm_predictions.csv', index=False)

# Print summary
print(f"\nPrediction Summary:")
print(f"  Mean absolute error: {predictions_df['abs_error'].mean():.2f} NGN/USD")
print(f"  Mean pct error: {predictions_df['pct_error'].mean():.2f}%")
print(f"  Direction accuracy: {predictions_df['direction_correct'].mean():.1%}")
```

---

### 18. [LOW] Python Compatibility Not Verified

**File**: All Python files
**Severity**: LOW - May not run on all Python versions

**Problem**:
- Code uses `type|type` union syntax (Python 3.10+)
- No explicit `__future__` imports
- No `.pyi` type hints

**Fix**: Add to all source files:
```python
from __future__ import annotations

# Python 3.9+ compatibility
import sys
if sys.version_info < (3, 9):
    raise RuntimeError("This project requires Python 3.9+")
```

---

## TESTING RECOMMENDATIONS

Create comprehensive tests to catch these issues:

```python
# tests/test_lstm_pipeline.py
import pytest
import numpy as np
from src.models import LSTMModel
from run_lstm_pipeline import prepare_data_with_validation

def test_no_data_leakage():
    """Verify no future data leaks into features."""
    # Implementation...
    pass

def test_sequence_alignment():
    """Verify predictions align with test dates."""
    # Implementation...
    pass

def test_scaling_consistency():
    """Verify scalers don't extrapolate beyond training range."""
    # Implementation...
    pass

def test_hyperparameter_validation():
    """Verify invalid hyperparameters are rejected."""
    # Implementation...
    pass

# Run with: pytest tests/ -v
```

---

## SUMMARY OF REQUIRED FIXES

| Issue | Severity | Effort | Priority |
|-------|----------|--------|----------|
| Data leakage in feature shifting | CRITICAL | High | 1 |
| Sequence alignment & context handling | CRITICAL | High | 2 |
| Feature consistency validation | CRITICAL | High | 3 |
| Target mode implementation | HIGH | Medium | 4 |
| Uncertainty quantification | HIGH | Medium | 5 |
| Scaling consistency | HIGH | High | 6 |
| NaN tracking | HIGH | Medium | 7 |
| Hyperparameter validation | HIGH | Low | 8 |
| Early stopping logging | MEDIUM | Low | 9 |
| Device handling | MEDIUM | Low | 10 |
| All others | MEDIUM/LOW | Varies | 11+ |

---

## NEXT STEPS

1. **Immediate** (Critical fixes): Apply fixes 1-3
2. **Short-term** (High priority): Apply fixes 4-8
3. **Medium-term**: Apply remaining medium priority fixes
4. **Long-term**: Consider advanced features (MC-dropout, checkpointing)
5. **Quality**: Add comprehensive test suite

---

**Audit Date**: May 15, 2026
**Auditor**: AI Code Review System
