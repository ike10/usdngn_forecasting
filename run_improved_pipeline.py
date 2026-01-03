"""
IMPROVED PIPELINE: Enhanced predictions with hyperparameter optimization
Tests improved models to reduce RMSE/MAE while maintaining directional accuracy
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
from improved_models import ImprovedARIMAModel, ImprovedLSTMModel, ImprovedHybridARIMALSTM
from part5_evaluation import ModelEvaluator

# Create output directories
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("\n" + "=" * 80)
print("USD-NGN IMPROVED PREDICTIONS PIPELINE")
print("=" * 80)

start_time = datetime.now()

# ============================================================================
# STAGE 1-3: Data Collection, Preprocessing, Splitting
# ============================================================================
print("\n[STAGE 1-3] DATA PREPARATION")
print("-" * 80)

collector = DataCollector(start_date='2023-01-01', end_date='2025-12-31')
raw_data = collector.collect_all_data()
print(f"✓ Collected: {raw_data.shape[0]} observations")

preprocessor = DataPreprocessor(raw_data)
processed_data, _ = preprocessor.preprocess()
print(f"✓ Processed: {processed_data.shape[0]} observations, {processed_data.shape[1]} features")

splitter = DataSplitter()
train_data, val_data, test_data = splitter.split(processed_data)
print(f"✓ Split: Train={train_data.shape[0]}, Val={val_data.shape[0]}, Test={test_data.shape[0]}")

# Prepare features
feature_cols = ['brent_oil', 'mpr', 'cpi', 'oil_return', 'usdngn_volatility',
                'usdngn_ma5', 'usdngn_ma20', 'rate_oil_ratio', 'mpr_change']
available_features = [f for f in feature_cols if f in train_data.columns]

X_train = train_data[available_features].values
y_train = train_data['usdngn'].values
X_val = val_data[available_features].values
y_val = val_data['usdngn'].values
X_test = test_data[available_features].values
y_test = test_data['usdngn'].values

# ============================================================================
# STAGE 4: BASELINE MODELS (Original)
# ============================================================================
print("\n[STAGE 4] BASELINE MODELS (Original Implementation)")
print("-" * 80)

baseline_results = {}

# Baseline Random Walk
print("\n  [4.1] Random Walk Baseline...")
rw = RandomWalkModel().fit(y_train)
rw_pred = np.roll(y_test, 1)
rw_pred[0] = y_train[-1]
rw_metrics = ModelEvaluator.compute_all_metrics(y_test, rw_pred)
baseline_results['Random Walk'] = rw_metrics
print(f"  ✓ RMSE={rw_metrics['RMSE']:.4f}, MAE={rw_metrics['MAE']:.4f}, DA={rw_metrics['DA']:.1f}%")

# Baseline ARIMA
print("\n  [4.2] ARIMA Baseline...")
arima_baseline = ARIMAModel()
arima_baseline.fit(y_train, verbose=False)
arima_pred_baseline = arima_baseline.predict(len(y_test))
arima_baseline_metrics = ModelEvaluator.compute_all_metrics(y_test, arima_pred_baseline)
baseline_results['ARIMA (Baseline)'] = arima_baseline_metrics
print(f"  ✓ RMSE={arima_baseline_metrics['RMSE']:.4f}, MAE={arima_baseline_metrics['MAE']:.4f}, DA={arima_baseline_metrics['DA']:.1f}%")

# Baseline Hybrid
print("\n  [4.3] Hybrid ARIMA-LSTM Baseline...")
hybrid_baseline = HybridARIMALSTM(arima_order=arima_baseline.best_order)
hybrid_baseline.fit(X_train, y_train, X_val, y_val, verbose=False)
hybrid_pred_baseline = hybrid_baseline.predict(X_test)
hybrid_baseline_metrics = ModelEvaluator.compute_all_metrics(y_test, hybrid_pred_baseline)
baseline_results['Hybrid (Baseline)'] = hybrid_baseline_metrics
print(f"  ✓ RMSE={hybrid_baseline_metrics['RMSE']:.4f}, MAE={hybrid_baseline_metrics['MAE']:.4f}, DA={hybrid_baseline_metrics['DA']:.1f}%")

# ============================================================================
# STAGE 5: IMPROVED MODELS
# ============================================================================
print("\n[STAGE 5] IMPROVED MODELS (Enhanced Hyperparameters)")
print("-" * 80)

improved_results = {}

# Improved ARIMA
print("\n  [5.1] Improved ARIMA (Extended Grid Search)...")
arima_improved = ImprovedARIMAModel(max_p=5, max_d=2, max_q=5)
arima_improved.fit(y_train, verbose=True)
arima_pred_improved = arima_improved.predict(len(y_test))
arima_improved_metrics = ModelEvaluator.compute_all_metrics(y_test, arima_pred_improved)
improved_results['ARIMA (Improved)'] = arima_improved_metrics
print(f"  ✓ RMSE={arima_improved_metrics['RMSE']:.4f}, MAE={arima_improved_metrics['MAE']:.4f}, DA={arima_improved_metrics['DA']:.1f}%")

# Improved LSTM (Ensemble)
print("\n  [5.2] Improved LSTM (Ensemble Architecture)...")
lstm_improved = ImprovedLSTMModel(input_size=X_train.shape[1])
lstm_improved.fit(X_train, y_train, X_val, y_val, verbose=True)
lstm_pred_improved = lstm_improved.predict(X_test)
lstm_improved_metrics = ModelEvaluator.compute_all_metrics(y_test, lstm_pred_improved)
improved_results['LSTM (Improved)'] = lstm_improved_metrics
print(f"  ✓ RMSE={lstm_improved_metrics['RMSE']:.4f}, MAE={lstm_improved_metrics['MAE']:.4f}, DA={lstm_improved_metrics['DA']:.1f}%")

# Improved Hybrid
print("\n  [5.3] Improved Hybrid (3-Component Ensemble)...")
hybrid_improved = ImprovedHybridARIMALSTM(
    arima_order=arima_improved.best_order,
    lstm_weight=0.5,
    arima_weight=0.3,
    residual_weight=0.2
)
hybrid_improved.fit(X_train, y_train, X_val, y_val, verbose=True)
hybrid_pred_improved = hybrid_improved.predict(X_test)
hybrid_improved_metrics = ModelEvaluator.compute_all_metrics(y_test, hybrid_pred_improved)
improved_results['Hybrid (Improved)'] = hybrid_improved_metrics
print(f"  ✓ RMSE={hybrid_improved_metrics['RMSE']:.4f}, MAE={hybrid_improved_metrics['MAE']:.4f}, DA={hybrid_improved_metrics['DA']:.1f}%")

# ============================================================================
# STAGE 6: COMPARISON AND ANALYSIS
# ============================================================================
print("\n[STAGE 6] MODEL COMPARISON")
print("-" * 80)

# Create comparison table
all_results = {**baseline_results, **improved_results}
comparison_df = pd.DataFrame(all_results).T
comparison_df['Model'] = comparison_df.index
comparison_df = comparison_df[['Model', 'RMSE', 'MAE', 'MAPE', 'DA', 'N']]

print("\nBASELINE vs IMPROVED MODELS:")
print(comparison_df.to_string(index=False))

# Calculate improvements
print("\n" + "=" * 80)
print("IMPROVEMENT ANALYSIS")
print("=" * 80)

baseline_hybrid_rmse = baseline_results['Hybrid (Baseline)']['RMSE']
improved_hybrid_rmse = improved_results['Hybrid (Improved)']['RMSE']
rmse_improvement = ((baseline_hybrid_rmse - improved_hybrid_rmse) / baseline_hybrid_rmse * 100)

baseline_hybrid_mae = baseline_results['Hybrid (Baseline)']['MAE']
improved_hybrid_mae = improved_results['Hybrid (Improved)']['MAE']
mae_improvement = ((baseline_hybrid_mae - improved_hybrid_mae) / baseline_hybrid_mae * 100)

baseline_hybrid_da = baseline_results['Hybrid (Baseline)']['DA']
improved_hybrid_da = improved_results['Hybrid (Improved)']['DA']
da_change = improved_hybrid_da - baseline_hybrid_da

print(f"\nHybrid Model Improvements:")
print(f"  RMSE: {baseline_hybrid_rmse:.4f} → {improved_hybrid_rmse:.4f} ({rmse_improvement:+.1f}%)")
print(f"  MAE:  {baseline_hybrid_mae:.4f} → {improved_hybrid_mae:.4f} ({mae_improvement:+.1f}%)")
print(f"  DA:   {baseline_hybrid_da:.1f}% → {improved_hybrid_da:.1f}% ({da_change:+.1f}%)")

# Find best overall model
best_rmse_model = comparison_df.loc[comparison_df['RMSE'].idxmin()]
best_da_model = comparison_df.loc[comparison_df['DA'].idxmax()]
best_mae_model = comparison_df.loc[comparison_df['MAE'].idxmin()]

print(f"\nBest Models by Metric:")
print(f"  Best RMSE: {best_rmse_model['Model']} ({best_rmse_model['RMSE']:.4f})")
print(f"  Best MAE:  {best_mae_model['Model']} ({best_mae_model['MAE']:.4f})")
print(f"  Best DA:   {best_da_model['Model']} ({best_da_model['DA']:.1f}%)")

# ============================================================================
# STAGE 7: SAVE RESULTS
# ============================================================================
print("\n[STAGE 7] SAVING RESULTS")
print("-" * 80)

# Save comparison
comparison_df.to_csv('data/model_comparison.csv', index=False)
print(f"✓ Saved to: data/model_comparison.csv")

# Save predictions
predictions_df = pd.DataFrame({
    'actual': y_test,
    'baseline_hybrid': hybrid_pred_baseline,
    'improved_hybrid': hybrid_pred_improved,
    'improved_lstm': lstm_pred_improved,
    'improved_arima': arima_pred_improved
})
predictions_df.to_csv('data/predictions_comparison.csv', index=False)
print(f"✓ Saved to: data/predictions_comparison.csv")

# Save detailed metrics
detailed_metrics = []
for model_name, metrics in all_results.items():
    metrics['Model'] = model_name
    detailed_metrics.append(metrics)
detailed_df = pd.DataFrame(detailed_metrics)
detailed_df.to_csv('data/detailed_metrics.csv', index=False)
print(f"✓ Saved to: data/detailed_metrics.csv")

# ============================================================================
# SUMMARY
# ============================================================================
duration = (datetime.now() - start_time).total_seconds()

print("\n" + "=" * 80)
print("PIPELINE SUMMARY")
print("=" * 80)
print(f"\nExecution Time: {duration:.1f} seconds")
print(f"\nKey Findings:")
print(f"  • Baseline Hybrid RMSE: {baseline_hybrid_rmse:.4f}")
print(f"  • Improved Hybrid RMSE: {improved_hybrid_rmse:.4f}")
print(f"  • RMSE Improvement: {rmse_improvement:.1f}%")
print(f"\nOutput Files Created:")
print(f"  ✓ data/model_comparison.csv - Side-by-side model metrics")
print(f"  ✓ data/predictions_comparison.csv - Actual vs predictions")
print(f"  ✓ data/detailed_metrics.csv - Full metric breakdown")
print(f"\nRecommendation:")
if rmse_improvement > 0:
    print(f"  ✓ Use the IMPROVED HYBRID model ({rmse_improvement:.1f}% better RMSE)")
else:
    print(f"  ⚠ Baseline Hybrid is better. Consider tuning hyperparameters further.")

print("\n" + "=" * 80)
print("✓ IMPROVED PIPELINE COMPLETE!")
print("=" * 80 + "\n")
