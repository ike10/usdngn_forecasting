# LSTM PIPELINE AUDIT - ACTION PLAN
**Prioritized fixes with effort estimates and execution roadmap**

---

## QUICK START

**Recommended approach:**
1. Read: LSTM_PIPELINE_AUDIT.md (issues 1-3 are CRITICAL)
2. Reference: LSTM_PIPELINE_FIXES.md (corrected code)
3. Execute: Follow this action plan step-by-step

**Time to implement all fixes**: 2-3 hours

---

## CRITICAL ISSUES (Must Fix Before Using)

### ISSUE #1: Data Leakage in Feature Engineering
**Severity**: CRITICAL | **Effort**: 30 min | **Impact**: High (invalid results)

**What to do**:
1. Open `run_lstm_pipeline.py`
2. Find lines 47-55 (feature shifting)
3. **DELETE** these lines:
   ```python
   # DELETE THIS:
   feature_cols_to_shift = [c for c in processed_data.columns if c != 'usdngn' and c not in lag_cols]
   processed_data[feature_cols_to_shift] = processed_data[feature_cols_to_shift].shift(1)
   ```
4. Replace `prepare_data()` function with `prepare_data_with_validation()` from LSTM_PIPELINE_FIXES.md
5. Test: `python run_lstm_pipeline.py --verbose True`
6. Verify: No errors about feature dimensions

**Why**: Features were shifted BEFORE train/val/test split, causing future information to leak into training.

---

### ISSUE #2: Sequence Length & Context Alignment
**Severity**: CRITICAL | **Effort**: 45 min | **Impact**: High (misaligned predictions)

**What to do**:
1. Open `src/models.py`
2. Find `LSTMModel.predict()` method (around line 835-920)
3. Add input validation at the start:
   ```python
   # Add this BEFORE processing:
   X_test = np.asarray(X_test, dtype=np.float32)
   if X_test.ndim != 2:
       raise ValueError(f"X_test must be 2D, got shape {X_test.shape}")
   
   if X_test.shape[1] != self.input_size:
       raise ValueError(f"X_test features ({X_test.shape[1]}) != input_size ({self.input_size})")
   ```
4. Replace the entire `predict()` method with the corrected version from LSTM_PIPELINE_FIXES.md
5. Test: `python run_lstm_pipeline.py --verbose True`
6. Verify: Predictions have correct length and no shape errors

**Why**: Predictions were misaligned with test dates due to unclear context handling.

---

### ISSUE #3: Feature Consistency Validation
**Severity**: CRITICAL | **Effort**: 20 min | **Impact**: High (silent failures)

**What to do**:
1. Replace `prepare_data()` function (done in Issue #1)
2. New function `prepare_data_with_validation()` includes this:
   ```python
   # Verify features exist in all splits
   missing_in_val = set(available_features) - set(val_data.columns)
   missing_in_test = set(available_features) - set(test_data.columns)
   
   if missing_in_val:
       raise ValueError(f"Features in train but missing in val: {missing_in_val}")
   if missing_in_test:
       raise ValueError(f"Features in train but missing in test: {missing_in_test}")
   ```
3. Test: `python run_lstm_pipeline.py --verbose True`
4. Verify: Console shows "✓ All X features present in train/val/test"

**Why**: Missing features in validation/test were silently ignored, causing dimension mismatches.

---

## HIGH PRIORITY ISSUES (Fix Next)

### ISSUE #4: Target Mode Implementation  
**Severity**: HIGH | **Effort**: 40 min | **Impact**: Medium (incorrect evaluation)

**What to do**:
1. Replace `prepare_data()` → use `prepare_data_with_validation()`
2. Add new function `prepare_targets()` from LSTM_PIPELINE_FIXES.md
3. Replace `make_return_targets()` call with `prepare_targets()`
4. Replace `returns_to_levels()` with `reconstruct_levels_from_returns()`
5. Update evaluation code in `fit_lstm()` to use new functions
6. Test: `python run_lstm_pipeline.py --target_mode return --verbose True`
7. Verify: Reconstructed levels are reasonable (compare to previous actual)

**Why**: Target mode logic was confusing and caused evaluation metric misalignment.

---

### ISSUE #5: Scaling Consistency
**Severity**: HIGH | **Effort**: 30 min | **Impact**: Medium (extrapolation errors)

**What to do**:
1. Open `src/models.py`
2. Find `LSTMModel.fit()` method
3. Replace this:
   ```python
   # OLD (WRONG):
   self.scaler_X.fit(X_train)
   self.scaler_y.fit(y_train.reshape(-1, 1))
   ```
   With this:
   ```python
   # NEW (CORRECT):
   if X_val is not None:
       X_for_scaling = np.vstack([X_train, X_val])
       y_for_scaling = np.concatenate([y_train, y_val])
   else:
       X_for_scaling = X_train
       y_for_scaling = y_train
   
   self.scaler_X.fit(X_for_scaling)
   self.scaler_y.fit(y_for_scaling.reshape(-1, 1))
   ```
4. Test: `python run_lstm_pipeline.py --verbose True`
5. Verify: Console shows scaler fitting range

**Why**: Scalers fit only on training data caused test data to be extrapolated outside normal ranges.

---

### ISSUE #6: NaN Tracking
**Severity**: HIGH | **Effort**: 25 min | **Impact**: Medium (hidden data quality issues)

**What to do**:
1. Use `prepare_data_with_validation()` from LSTM_PIPELINE_FIXES.md (already includes NaN tracking)
2. Test: `python run_lstm_pipeline.py --verbose True`
3. Verify: Console shows detailed NaN analysis for each split

**Why**: NaN values were silently replaced with zero, masking data quality issues.

---

### ISSUE #7: Input Validation in LSTMModel
**Severity**: HIGH | **Effort**: 20 min | **Impact**: Low-Medium (prevents cryptic errors)

**What to do**:
1. Open `src/models.py`
2. Find `LSTMModel.__init__()`
3. Add validation block from LSTM_PIPELINE_FIXES.md (after line with `def __init__`)
4. Test with invalid parameters: `python run_lstm_pipeline.py --hidden_size1 -5`
5. Verify: Clear error message about invalid parameter

**Why**: Invalid hyperparameters caused confusing errors later in training.

---

### ISSUE #8: Hyperparameter Validation in CLI
**Severity**: HIGH | **Effort**: 15 min | **Impact**: Low-Medium (prevents mistakes)

**What to do**:
1. Open `run_lstm_pipeline.py`
2. Find `parse_args()` function
3. Add validation block at end (before `return args`):
   ```python
   # Validation
   if args.sequence_length < 5 or args.sequence_length > 500:
       raise ValueError(f"sequence_length must be 5-500, got {args.sequence_length}")
   
   if args.patience >= args.epochs:
       raise ValueError(f"patience ({args.patience}) must be < epochs ({args.epochs})")
   
   # ... (more validation in LSTM_PIPELINE_FIXES.md)
   ```
4. Test: `python run_lstm_pipeline.py --patience 100 --epochs 50`
5. Verify: Error message about patience >= epochs

**Why**: Invalid hyperparameters silently caused training failures.

---

## MEDIUM PRIORITY ISSUES (Nice to Have)

### ISSUE #9: Early Stopping Logging
**Severity**: MEDIUM | **Effort**: 10 min | **Impact**: Low

**What to do**:
1. Open `src/models.py`
2. Find early stopping code in `fit()` method
3. Replace:
   ```python
   if patience_counter >= self.patience:
       if verbose:
           print(f"Early stopping at epoch {epoch+1}")
   ```
   With:
   ```python
   if patience_counter >= self.patience:
       if verbose:
           print(f"  ✓ Early stopping at epoch {epoch+1}/{self.epochs}")
           print(f"    Best val loss: {best_val_loss:.6f}")
           print(f"    Current val loss: {val_loss:.6f}")
           print(f"    Patience: {self.patience} epochs")
   ```
4. Test: `python run_lstm_pipeline.py --patience 3 --epochs 20`
5. Verify: Better logging when early stopping occurs

**Why**: Early stopping was silent, hard to debug.

---

### ISSUE #10: Device Setup & Logging
**Severity**: MEDIUM | **Effort**: 15 min | **Impact**: Low

**What to do**:
1. Open `run_lstm_pipeline.py`
2. Add `setup_device()` function from LSTM_PIPELINE_FIXES.md
3. Replace device setup code:
   ```python
   # OLD:
   device = torch.device(...) if TORCH_AVAILABLE else 'CPU'
   
   # NEW:
   device = setup_device()
   ```
4. Test: `python run_lstm_pipeline.py --verbose True`
5. Verify: Better GPU/CPU detection logging

**Why**: Device handling was confusing and error-prone.

---

### ISSUE #11: Enhanced Predictions Output
**Severity**: MEDIUM | **Effort**: 20 min | **Impact**: Low-Medium (better analysis)

**What to do**:
1. Open `run_lstm_pipeline.py`
2. Replace predictions DataFrame code (around line 171-177) with enhanced version from LSTM_PIPELINE_FIXES.md
3. Test: `python run_lstm_pipeline.py --save_model --verbose True`
4. Verify: `data/lstm_predictions.csv` has columns for errors, direction, etc.

**Why**: Original predictions CSV was incomplete for analysis.

---

### ISSUE #12: Model Architecture Logging
**Severity**: MEDIUM | **Effort**: 10 min | **Impact**: Low (reproducibility)

**What to do**:
1. Open `src/models.py`
2. Find `LSTMModel.fit()` method
3. Add logging block at start from LSTM_PIPELINE_FIXES.md:
   ```python
   if verbose:
       print("\n  [LSTM Configuration]")
       print(f"    Input features: {self.input_size}")
       print(f"    Sequence length: {self.sequence_length}")
       # ... (more architecture details)
   ```
4. Test: `python run_lstm_pipeline.py --verbose True`
5. Verify: Architecture printed at start of training

**Why**: Hard to reproduce results without knowing exact architecture used.

---

### ISSUE #13: Return Value Documentation
**Severity**: MEDIUM | **Effort**: 15 min | **Impact**: Low (better code clarity)

**What to do**:
1. Open `run_lstm_pipeline.py`
2. Update `fit_lstm()` docstring with return value specification (from LSTM_PIPELINE_FIXES.md)
3. Update return statement to return dict with clear structure
4. Update `if __name__ == "__main__"` block to use return value
5. Test: `python run_lstm_pipeline.py`
6. Verify: Console shows final summary with file paths

**Why**: Return values were dead code, now actually used.

---

### ISSUE #14: Checkpoint Support (Optional Enhancement)
**Severity**: MEDIUM | **Effort**: 30 min | **Impact**: Low (nice to have)

**What to do**:
1. Add checkpoint saving in `LSTMModel.fit()` when new best model found
2. Code in LSTM_PIPELINE_FIXES.md shows how to save checkpoints
3. Allow resuming training from checkpoint
4. Test: Run training, interrupt, then resume
5. Verify: Model continues from best checkpoint

**Why**: No recovery from interrupted training currently.

---

## TESTING CHECKLIST

After implementing fixes, run these tests:

```bash
# Test 1: Basic run with default parameters
python run_lstm_pipeline.py --verbose True

# Test 2: Return mode (training on log-returns)
python run_lstm_pipeline.py --target_mode return --verbose True

# Test 3: Level mode (training on price levels)
python run_lstm_pipeline.py --target_mode level --verbose True

# Test 4: Custom hyperparameters
python run_lstm_pipeline.py --sequence_length 60 --epochs 100 --hidden_size1 128 --hidden_size2 64 --verbose True

# Test 5: Model saving
python run_lstm_pipeline.py --save_model --verbose True

# Test 6: Invalid parameters (should fail with clear error)
python run_lstm_pipeline.py --sequence_length 0 --verbose True  # Should error
python run_lstm_pipeline.py --patience 100 --epochs 50 --verbose True  # Should error
```

---

## IMPLEMENTATION ORDER (Recommended)

**Phase 1 (CRITICAL - 1.5 hours)**:
1. Issue #1: Data Leakage (30 min)
2. Issue #2: Sequence Alignment (45 min)  
3. Issue #3: Feature Validation (20 min)

**Phase 2 (HIGH - 2 hours)**:
4. Issue #4: Target Mode (40 min)
5. Issue #5: Scaling Consistency (30 min)
6. Issue #6: NaN Tracking (25 min)
7. Issue #7: LSTMModel Validation (20 min)
8. Issue #8: CLI Validation (15 min)

**Phase 3 (MEDIUM - 1 hour)**:
9. Issue #9: Early Stopping Logging (10 min)
10. Issue #10: Device Setup (15 min)
11. Issue #11: Predictions Output (20 min)
12. Issue #12: Architecture Logging (10 min)
13. Issue #13: Return Values (15 min)

**Phase 4 (OPTIONAL - 30 min)**:
14. Issue #14: Checkpoint Support (30 min)

---

## SUCCESS CRITERIA

After all fixes, your LSTM pipeline should:

✅ **Pass all tests** without errors
✅ **Produce reasonable metrics** (DA > 50%, RMSE < 50 NGN/USD)
✅ **Align predictions** with test dates exactly
✅ **Log detailed information** about training process
✅ **Handle errors gracefully** with clear messages
✅ **Save outputs** to CSV files with complete information
✅ **Be reproducible** with same hyperparameters

---

## COMPARISON: Before vs After

| Aspect | Before (Broken) | After (Fixed) |
|--------|-----------------|---------------|
| Data Leakage | ❌ Features shifted before split | ✅ Features only shifted within split |
| Sequence Alignment | ❌ Predictions misaligned | ✅ Predictions aligned with dates |
| Feature Validation | ❌ Silent failures | ✅ Explicit checks with errors |
| Scaling | ❌ Extrapolation risk | ✅ Fit on train+val |
| NaN Handling | ❌ Silent replacement | ✅ Detailed tracking & logging |
| Error Messages | ❌ Cryptic | ✅ Clear & actionable |
| Reproducibility | ❌ Hard to track architecture | ✅ Full logging of config |
| Predictions CSV | ❌ Incomplete | ✅ Full analysis columns |

---

## ESTIMATED TIMELINE

| Phase | Time | Complexity |
|-------|------|-----------|
| Phase 1 (CRITICAL) | 1.5 hours | High |
| Phase 2 (HIGH) | 2 hours | Medium-High |
| Phase 3 (MEDIUM) | 1 hour | Low-Medium |
| Phase 4 (OPTIONAL) | 0.5 hours | Low |
| Testing & Verification | 0.5 hours | Low |
| **TOTAL** | **~5 hours** | Medium |

---

## SUPPORT REFERENCES

- **Detailed Audit**: Read LSTM_PIPELINE_AUDIT.md for complete issue descriptions
- **Code Examples**: Reference LSTM_PIPELINE_FIXES.md for corrected code
- **Current State**: Review original files in src/ and run_lstm_pipeline.py
- **Hybrid Model**: DO NOT TOUCH - not included in this audit

---

**Last Updated**: May 15, 2026
**Status**: Ready for implementation
