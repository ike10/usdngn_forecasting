"""
EXECUTABLE PIPELINE: Data Collection → Preprocessing → Modeling → Evaluation
Saves outputs to data/ and models/ directories
"""
import numpy as np
import pandas as pd
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from part1_data_collection import DataCollector
from part2_preprocessing import DataPreprocessor, DataSplitter
from part4_models import ARIMAModel, RandomWalkModel, HybridARIMALSTM
from final_optimized_models import OptimizedHybrid, VotingEnsemble
from part5_evaluation import ModelEvaluator, RegimeEvaluator

# Create output directories
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("\n" + "=" * 70)
print("USD-NGN FORECASTING PIPELINE - EXECUTABLE VERSION")
print("=" * 70)

start_time = datetime.now()

# ============================================================================
# STAGE 1: DATA COLLECTION
# ============================================================================
print("\n[STAGE 1] DATA COLLECTION")
print("-" * 70)
collector = DataCollector(start_date='2023-01-01', end_date='2025-12-31')
raw_data = collector.collect_all_data()
print(f"✓ Collected: {raw_data.shape[0]} observations, {raw_data.shape[1]} variables")

# Save raw data
raw_data.to_csv('data/raw_data.csv')
print(f"✓ Saved to: data/raw_data.csv")

# ============================================================================
# STAGE 2: PREPROCESSING
# ============================================================================
print("\n[STAGE 2] PREPROCESSING")
print("-" * 70)
preprocessor = DataPreprocessor(raw_data)
processed_data, stationarity = preprocessor.preprocess()
print(f"✓ Processed: {processed_data.shape[0]} observations, {processed_data.shape[1]} features")
print("\nStationarity Tests:")
print(stationarity.to_string())

# Save processed data
processed_data.to_csv('data/processed_data.csv')
print(f"\n✓ Saved to: data/processed_data.csv")

# ============================================================================
# STAGE 3: DATA SPLITTING
# ============================================================================
print("\n[STAGE 3] DATA SPLITTING")
print("-" * 70)
splitter = DataSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
train_data, val_data, test_data = splitter.split(processed_data)

# Save splits
train_data.to_csv('data/train_data.csv')
val_data.to_csv('data/val_data.csv')
test_data.to_csv('data/test_data.csv')
print(f"✓ Saved splits to: data/train_data.csv, val_data.csv, test_data.csv")

# ============================================================================
# STAGE 4: FEATURE PREPARATION
# ============================================================================
print("\n[STAGE 4] FEATURE PREPARATION")
print("-" * 70)
feature_cols = ['brent_oil', 'mpr', 'cpi', 'oil_return', 'usdngn_volatility',
                'usdngn_ma5', 'usdngn_ma20', 'rate_oil_ratio', 'mpr_change']
available_features = [f for f in feature_cols if f in train_data.columns]
print(f"✓ Available features: {len(available_features)}/{len(feature_cols)}")
print(f"  {available_features}")

X_train = train_data[available_features].values
y_train = train_data['usdngn'].values
X_val = val_data[available_features].values
y_val = val_data['usdngn'].values
X_test = test_data[available_features].values
y_test = test_data['usdngn'].values

print(f"✓ X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"✓ X_test: {X_test.shape}, y_test: {y_test.shape}")

# ============================================================================
# STAGE 5: MODEL TRAINING
# ============================================================================
print("\n[STAGE 5] MODEL TRAINING")
print("-" * 70)

models = {}
predictions = {}

# Random Walk (Baseline)
print("\n  [5.1] Random Walk (Baseline)...")
rw = RandomWalkModel().fit(y_train)
models['Random Walk'] = rw
rw_pred = np.roll(y_test, 1)
rw_pred[0] = y_train[-1]
predictions['Random Walk'] = rw_pred
print("  ✓ Trained")

# ARIMA
print("\n  [5.2] ARIMA...")
arima = ARIMAModel()
arima.fit(y_train, verbose=True)
models['ARIMA'] = arima
arima_pred = arima.predict(len(y_test))
predictions['ARIMA'] = arima_pred
print("  ✓ Trained")

# Hybrid ARIMA-LSTM (Baseline)
print("\n  [5.3] Hybrid ARIMA-LSTM (Baseline)...")
hybrid = HybridARIMALSTM(arima_order=arima.best_order)
hybrid.fit(X_train, y_train, X_val, y_val, verbose=False)
models['Hybrid (Baseline)'] = hybrid
hybrid_pred = hybrid.predict(X_test)
predictions['Hybrid (Baseline)'] = hybrid_pred
print("  ✓ Trained")

# Optimized Hybrid (Enhanced)
print("\n  [5.4] OPTIMIZED Hybrid (85% LSTM + 15% ARIMA)...")
optimized_hybrid = OptimizedHybrid()
optimized_hybrid.fit(X_train, y_train, X_val, y_val, verbose=False)
models['Hybrid (Optimized)'] = optimized_hybrid
optimized_hybrid_pred = optimized_hybrid.predict(X_test)
predictions['Hybrid (Optimized)'] = optimized_hybrid_pred
print("  ✓ Trained (ENHANCED for 70%+ DA & Low RMSE)")

# Voting Ensemble (Alternative)
print("\n  [5.5] Voting Ensemble (5 Models)...")
voting = VotingEnsemble()
voting.fit(X_train, y_train, X_val, y_val, verbose=False)
models['Voting Ensemble'] = voting
voting_pred = voting.predict(X_test)
predictions['Voting Ensemble'] = voting_pred
print("  ✓ Trained")

# ============================================================================
# STAGE 6: EVALUATION
# ============================================================================
print("\n[STAGE 6] EVALUATION")
print("-" * 70)

results = {}
metrics_summary = []

for name, pred in predictions.items():
    min_len = min(len(y_test), len(pred))
    metrics = ModelEvaluator.compute_all_metrics(y_test[:min_len], pred[:min_len])
    results[name] = metrics
    metrics['Model'] = name
    metrics_summary.append(metrics)
    
    print(f"\n  {name}:")
    print(f"    RMSE: {metrics['RMSE']:.4f}")
    print(f"    MAE:  {metrics['MAE']:.4f}")
    print(f"    MAPE: {metrics['MAPE']:.2f}%")
    print(f"    DA:   {metrics['DA']:.1f}%")

# Save metrics
metrics_df = pd.DataFrame(metrics_summary)
metrics_df.to_csv('data/evaluation_metrics.csv', index=False)
print(f"\n✓ Saved metrics to: data/evaluation_metrics.csv")

# ============================================================================
# STAGE 7: SUMMARY
# ============================================================================
duration = (datetime.now() - start_time).total_seconds()

print("\n" + "=" * 70)
print("PIPELINE SUMMARY - TARGET ACHIEVEMENT REPORT")
print("=" * 70)
print(f"\nExecution Time: {duration:.1f} seconds")

# Highlight best models
best_rmse_idx = metrics_df['RMSE'].idxmin()
best_da_idx = metrics_df['DA'].idxmax()

print(f"\n📊 Best Models:")
print(f"  Best RMSE:  {metrics_df.loc[best_rmse_idx, 'Model']} ({metrics_df.loc[best_rmse_idx, 'RMSE']:.4f})")
print(f"  Best DA:    {metrics_df.loc[best_da_idx, 'Model']} ({metrics_df.loc[best_da_idx, 'DA']:.1f}%)")

# Check targets
rw_rmse = metrics_df[metrics_df['Model'] == 'Random Walk']['RMSE'].values[0]

optimized_models = [m for m in metrics_df['Model'] if 'Optimized' in m or 'Voting' in m]

print(f"\n🎯 TARGET ACHIEVEMENT:")
print(f"   Target 1: DA ≥ 70%")
print(f"   Target 2: RMSE < Random Walk ({rw_rmse:.4f})")

for model_name in optimized_models:
    if model_name in metrics_df['Model'].values:
        row = metrics_df[metrics_df['Model'] == model_name].iloc[0]
        rmse = row['RMSE']
        da = row['DA']
        
        target1 = da >= 70
        target2 = rmse < rw_rmse
        
        print(f"\n   [{model_name}]")
        print(f"     DA:   {da:.1f}% {'✅' if target1 else '❌'}")
        print(f"     RMSE: {rmse:.4f} {'✅' if target2 else '❌'}")
        
        if target1 and target2:
            print(f"     🎉 BOTH TARGETS ACHIEVED!")

print(f"\n📁 Output Files:")
print(f"  ✓ data/raw_data.csv")
print(f"  ✓ data/processed_data.csv")
print(f"  ✓ data/train_data.csv, val_data.csv, test_data.csv")
print(f"  ✓ data/evaluation_metrics.csv")

print("\n" + "=" * 70)
print("✨ PIPELINE COMPLETE!")
print("=" * 70 + "\n")
