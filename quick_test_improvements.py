"""Quick test of hybrid model improvements"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/mnt/c/Users/HP/Desktop/Masters/Thesis/Code/usdngn_forecasting')

from part4_models import HybridARIMALSTM, RandomWalkModel, ARIMAModel
from part5_evaluation import ModelEvaluator

# Load data
train_data = pd.read_csv('data/train_data.csv')
val_data = pd.read_csv('data/val_data.csv')
test_data = pd.read_csv('data/test_data.csv')

# Features
features = [col for col in train_data.columns if col in [
    'brent_oil', 'mpr', 'cpi', 'oil_return', 'usdngn_volatility',
    'usdngn_ma5', 'usdngn_ma20', 'rate_oil_ratio', 'mpr_change'
]]

X_train = train_data[features].values
y_train = train_data['usdngn'].values
X_test = test_data[features].values
y_test = test_data['usdngn'].values
X_val = val_data[features].values
y_val = val_data['usdngn'].values

# Handle NaN
X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
X_val = np.nan_to_num(X_val, nan=0, posinf=0, neginf=0)
X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

print("\nQuick Model Comparison")
print("=" * 60)

# Random Walk
rw = RandomWalkModel().fit(y_train)
rw_pred = np.concatenate([[y_train[-1]], y_test[:-1]])
rw_rmse = ModelEvaluator.rmse(y_test, rw_pred)
rw_da = ModelEvaluator.directional_accuracy(y_test, rw_pred)
print(f"\nRandom Walk (Baseline):")
print(f"  RMSE: {rw_rmse:.4f}")
print(f"  DA:   {rw_da:.1%}")

# ARIMA
arima = ARIMAModel()
arima.fit(y_train, verbose=False)
arima_pred = arima.predict(len(y_test))
arima_rmse = ModelEvaluator.rmse(y_test, arima_pred)
arima_da = ModelEvaluator.directional_accuracy(y_test, arima_pred)
print(f"\nARIMA (Baseline):")
print(f"  RMSE: {arima_rmse:.4f}")
print(f"  DA:   {arima_da:.1%}")

# Enhanced Hybrid
print(f"\nEnhanced Hybrid (Optimized):")
print("  Training...")
hybrid = HybridARIMALSTM(
    arima_order=arima.best_order,
    sequence_length=40,
    arima_weight=0.25,  # ENHANCED: More LSTM weight
    use_directional_ensemble=True
)
hybrid.fit(X_train, y_train, X_val, y_val, verbose=False)
hybrid_pred = hybrid.predict(X_test)

# Align lengths
min_len = min(len(y_test), len(hybrid_pred))
y_test_aligned = y_test[-min_len:]
hybrid_pred_aligned = hybrid_pred[-min_len:]

hybrid_rmse = ModelEvaluator.rmse(y_test_aligned, hybrid_pred_aligned)
hybrid_da = ModelEvaluator.directional_accuracy(y_test_aligned, hybrid_pred_aligned)

print(f"  RMSE: {hybrid_rmse:.4f} (vs RW: {rw_rmse:.4f}) - {'✓ BETTER' if hybrid_rmse < rw_rmse else '✗ WORSE'}")
print(f"  DA:   {hybrid_da:.1%} (target: 70%+) - {'✓ TARGET' if hybrid_da >= 0.70 else '✗ BELOW'}")

print("\n" + "=" * 60)
print("SUMMARY:")
print(f"  RW RMSE: {rw_rmse:.4f} | RW DA: {rw_da:.1%}")
print(f"  HY RMSE: {hybrid_rmse:.4f} | HY DA: {hybrid_da:.1%}")

if hybrid_rmse < rw_rmse and hybrid_da >= 0.70:
    print("\n✅ BOTH TARGETS ACHIEVED!")
else:
    print("\n⚠️ Targets:")
    print(f"   - RMSE < RW: {'✓' if hybrid_rmse < rw_rmse else '✗'}")
    print(f"   - DA >= 70%: {'✓' if hybrid_da >= 0.70 else '✗'}")
