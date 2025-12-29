"""
Quick test of the pipeline with reduced data
"""
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from part1_data_collection import DataCollector
from part2_preprocessing import DataPreprocessor, DataSplitter
from part4_models import ARIMAModel, LSTMModel, HybridARIMALSTM, RandomWalkModel
from part5_evaluation import ModelEvaluator

print("\n" + "=" * 70)
print("QUICK PIPELINE TEST (Reduced Data)")
print("=" * 70)

start_time = datetime.now()

# Stage 1: Data Collection (1 year only)
print("\n[STAGE 1] Data Collection (2024-2025)")
collector = DataCollector(start_date='2024-01-01', end_date='2025-12-31')
raw_data = collector.collect_all_data()
print(f"Raw data shape: {raw_data.shape}")

# Stage 2: Preprocessing
print("\n[STAGE 2] Preprocessing")
preprocessor = DataPreprocessor(raw_data)
processed_data, stationarity = preprocessor.preprocess()
print(f"Processed data shape: {processed_data.shape}")
print("\nStationarity Tests:")
print(stationarity)

# Stage 3: Data Splitting
print("\n[STAGE 3] Data Splitting")
splitter = DataSplitter()
train_data, val_data, test_data = splitter.split(processed_data)
print(f"Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")

# Stage 4: Model Training (simplified)
print("\n[STAGE 4] Model Training")
feature_cols = ['brent_oil', 'mpr', 'cpi', 'oil_return']
available_features = [f for f in feature_cols if f in train_data.columns]
print(f"Using features: {available_features}")

X_train = train_data[available_features].values
y_train = train_data['usdngn'].values
X_test = test_data[available_features].values
y_test = test_data['usdngn'].values

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

models = {}
predictions = {}

# Random Walk
print("  Training Random Walk...")
rw = RandomWalkModel().fit(y_train)
models['Random Walk'] = rw
rw_pred = np.roll(y_test, 1)
rw_pred[0] = y_train[-1]
predictions['Random Walk'] = rw_pred

# ARIMA
print("  Training ARIMA...")
arima = ARIMAModel()
arima.fit(y_train, verbose=False)
models['ARIMA'] = arima
predictions['ARIMA'] = arima.predict(len(y_test))

# Stage 5: Evaluation
print("\n[STAGE 5] Evaluation")
metrics_all = {}
for name, pred in predictions.items():
    min_len = min(len(y_test), len(pred))
    metrics = ModelEvaluator.compute_all_metrics(y_test[:min_len], pred[:min_len])
    metrics_all[name] = metrics
    print(f"  {name}: RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}, DA={metrics['DA']:.1f}%")

# Summary
duration = (datetime.now() - start_time).total_seconds()
print("\n" + "=" * 70)
print(f"✓ Pipeline completed in {duration:.1f} seconds")
print("=" * 70)
print(f"\nResults Summary:")
for name, metrics in metrics_all.items():
    print(f"  {name}: RMSE={metrics['RMSE']:.2f}")
