#!/usr/bin/env python3
"""Minimal LSTM test to identify the NaN issue."""

import sys
import numpy as np
import pandas as pd

print("=" * 70)
print("MINIMAL LSTM PREDICTION TEST")
print("=" * 70)

# Load data
train_data = pd.read_csv('data/train_data.csv', index_col=0)
val_data = pd.read_csv('data/val_data.csv', index_col=0)
test_data = pd.read_csv('data/test_data.csv', index_col=0)

feature_cols = ['brent_oil', 'mpr', 'cpi', 'oil_return', 'usdngn_volatility',
                'usdngn_ma5', 'usdngn_ma20', 'rate_oil_ratio', 'mpr_change']

X_train = np.nan_to_num(train_data[feature_cols].values, nan=0)
y_train = train_data['usdngn'].values
X_val = np.nan_to_num(val_data[feature_cols].values, nan=0)
y_val = val_data['usdngn'].values
X_test = np.nan_to_num(test_data[feature_cols].values, nan=0)
y_test = test_data['usdngn'].values

print(f"\nX_train: {X_train.shape}")
print(f"X_val: {X_val.shape}")
print(f"X_test: {X_test.shape}")

sequence_length = 15

# Minimal test: do NOT use LSTM, just test the predict logic with numpy
print("\n[Testing predict logic WITHOUT LSTM]")

from sklearn.preprocessing import MinMaxScaler

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_combined = np.vstack([X_train, X_val, X_test])
print(f"X_combined shape: {X_combined.shape}")

# Fit scaler on combined data
scaler_X.fit(X_combined)
scaler_y.fit(np.concatenate([y_train, y_val, y_test]).reshape(-1, 1))

# Now test the rolling prediction logic
X_test_scaled = scaler_X.transform(X_test)
X_context_scaled = scaler_X.transform(np.vstack([X_train, X_val]))
X_full = np.vstack([X_context_scaled, X_test_scaled])

print(f"\nX_test_scaled: {X_test_scaled.shape}")
print(f"X_context_scaled: {X_context_scaled.shape}")
print(f"X_full: {X_full.shape}")

start_idx = len(X_context_scaled)
print(f"start_idx: {start_idx}")
print(f"sequence_length: {sequence_length}")

predictions = []
for i in range(len(X_test)):
    global_idx = start_idx + i
    
    if global_idx < sequence_length:
        predictions.append(np.nan)
        if i < 5:
            print(f"  i={i}, global_idx={global_idx} < {sequence_length}, -> NaN")
    else:
        window_start = global_idx - sequence_length
        window_end = global_idx
        window = X_full[window_start:window_end]
        
        # Dummy prediction: use window mean as a test
        window_mean = window.mean()
        pred_scaled = window_mean
        
        # Inverse scale - BUT need a proper 2D array for inverse_transform
        pred_2d = np.array([[pred_scaled]])
        pred_original = scaler_y.inverse_transform(pred_2d)[0, 0]
        
        predictions.append(pred_original)
        
        if i < 5:
            print(f"  i={i}, global_idx={global_idx}, window {window_start}:{window_end}, pred={pred_original:.2f}")

predictions_arr = np.array(predictions)
print(f"\nPredictions shape: {predictions_arr.shape}")
print(f"Valid: {np.sum(~np.isnan(predictions_arr))}, NaN: {np.sum(np.isnan(predictions_arr))}")
print(f"First 20: {predictions_arr[:20]}")
print(f"Actual first 20 y_test: {y_test[:20]}")
