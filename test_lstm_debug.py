#!/usr/bin/env python3
"""Quick test of LSTM predict method with debug output."""

import numpy as np
import pandas as pd
from src.models import LSTMModel

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

print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"X_test shape: {X_test.shape}")

# Create and train LSTM
print("\n[Training LSTM...]")
lstm_model = LSTMModel(
    input_size=X_train.shape[1],
    sequence_length=15,
    batch_size=32,
    epochs=75,
    patience=12,
    learning_rate=0.0005,
)
lstm_model.fit(X_train, y_train, X_val, y_val, verbose=True)

print("\n[Predicting on test set with context...]")
X_context = np.vstack([X_train, X_val])
print(f"X_context shape: {X_context.shape}")

predictions = lstm_model.predict(X_test, X_context=X_context, verbose_debug=True)

print(f"\nPredictions shape: {predictions.shape}")
print(f"Valid predictions: {np.sum(~np.isnan(predictions))}")
print(f"NaN predictions: {np.sum(np.isnan(predictions))}")
print(f"First 20 predictions: {predictions[:20]}")
