"""
ENHANCED PIPELINE: Optimized for 70%+ Directional Accuracy and Low RMSE
Uses the Enhanced Hybrid Model with Residual Correction
"""
import numpy as np
import pandas as pd
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from part1_data_collection import DataCollector
from part2_preprocessing import DataPreprocessor, DataSplitter
from part4_models import ARIMAModel, RandomWalkModel
from enhanced_hybrid_model import EnhancedHybridModel
from part5_evaluation import ModelEvaluator

# Create output directories
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("\n" + "=" * 70)
print("USD-NGN ENHANCED FORECASTING PIPELINE")
print("Target: DA ≥ 70% | RMSE < Random Walk")
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

# Save processed data
processed_data.to_csv('data/processed_data.csv')
print(f"✓ Saved to: data/processed_data.csv")

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
print(f"✓ Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

# ============================================================================
# STAGE 4: FEATURE PREPARATION
# ============================================================================
print("\n[STAGE 4] FEATURE PREPARATION")
print("-" * 70)
feature_cols = ['brent_oil', 'mpr', 'cpi', 'oil_return', 'usdngn_volatility',
                'usdngn_ma5', 'usdngn_ma20', 'rate_oil_ratio', 'mpr_change']
available_features = [f for f in feature_cols if f in train_data.columns]
print(f"✓ Features: {len(available_features)} available")

X_train = train_data[available_features].values
y_train = train_data['usdngn'].values
X_val = val_data[available_features].values
y_val = val_data['usdngn'].values
X_test = test_data[available_features].values
y_test = test_data['usdngn'].values

# Handle NaN values
X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

print(f"✓ X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"✓ X_test: {X_test.shape}, y_test: {y_test.shape}")

# ============================================================================
# STAGE 5: MODEL TRAINING
# ============================================================================
print("\n[STAGE 5] MODEL TRAINING & EVALUATION")
print("-" * 70)

models = {}
predictions = {}

# ---- Random Walk (Baseline) ----
print("\n  [5.1] Random Walk Baseline...")
rw = RandomWalkModel().fit(y_train)
models['Random Walk'] = rw
rw_pred = np.concatenate([[y_train[-1]], y_test[:-1]])
predictions['Random Walk'] = rw_pred

rw_rmse = ModelEvaluator.rmse(y_test, rw_pred)
rw_da = ModelEvaluator.directional_accuracy(y_test, rw_pred)
print(f"    RMSE: {rw_rmse:.4f}")
print(f"    DA:   {rw_da:.1f}%")

# ---- ARIMA (Baseline) ----
print("\n  [5.2] ARIMA Baseline...")
arima = ARIMAModel()
arima.fit(y_train, verbose=False)
models['ARIMA'] = arima
arima_pred = arima.predict(len(y_test))
predictions['ARIMA'] = arima_pred

arima_rmse = ModelEvaluator.rmse(y_test, arima_pred)
arima_da = ModelEvaluator.directional_accuracy(y_test, arima_pred)
print(f"    RMSE: {arima_rmse:.4f}")
print(f"    DA:   {arima_da:.1f}%")

# ---- Enhanced Hybrid Model (NEW) ----
print("\n  [5.3] Enhanced Hybrid Model (Target: 70%+ DA, RMSE < RW)...")
try:
    enhanced = EnhancedHybridModel(sequence_length=30)
    enhanced.fit(X_train, y_train, X_val, y_val, verbose=False)
    enhanced_pred = enhanced.predict(X_test)
    
    # Align predictions
    min_len = min(len(y_test), len(enhanced_pred))
    enhanced_pred_aligned = enhanced_pred[-min_len:]
    y_test_aligned = y_test[-min_len:]
    
    predictions['Enhanced Hybrid'] = enhanced_pred_aligned
    models['Enhanced Hybrid'] = enhanced
    
    enhanced_rmse = ModelEvaluator.rmse(y_test_aligned, enhanced_pred_aligned)
    enhanced_da = ModelEvaluator.directional_accuracy(y_test_aligned, enhanced_pred_aligned)
    print(f"    RMSE: {enhanced_rmse:.4f} (vs RW: {rw_rmse:.4f}) {'✓ BETTER' if enhanced_rmse < rw_rmse else '✗'}")
    print(f"    DA:   {enhanced_da:.1f}% (target: 70%+) {'✓ REACHED' if enhanced_da >= 70 else '✗'}")
    
    enhanced_available = True
except Exception as e:
    print(f"    ✗ Failed to train enhanced model: {str(e)[:100]}")
    enhanced_available = False

# ============================================================================
# STAGE 6: COMPREHENSIVE EVALUATION
# ============================================================================
print("\n[STAGE 6] EVALUATION & RESULTS")
print("-" * 70)

results = {}
metrics_summary = []

print("\nModel Comparison:")
print(f"{'Model':<25} {'RMSE':<12} {'MAE':<12} {'MAPE':<12} {'DA':<10}")
print("-" * 71)

for name, pred in predictions.items():
    min_len = min(len(y_test), len(pred))
    y_true_aligned = y_test[-min_len:]
    pred_aligned = pred[-min_len:] if len(pred) >= min_len else np.pad(pred, (max(0, min_len-len(pred)), 0), mode='edge')
    
    metrics = ModelEvaluator.compute_all_metrics(y_true_aligned, pred_aligned)
    results[name] = metrics
    metrics['Model'] = name
    metrics_summary.append(metrics)
    
    print(f"{name:<25} {metrics['RMSE']:<12.4f} {metrics['MAE']:<12.4f} {metrics['MAPE']:<12.2f} {metrics['DA']:<10.1f}%")

# Save metrics
metrics_df = pd.DataFrame(metrics_summary)
metrics_df = metrics_df[['Model', 'RMSE', 'MAE', 'MAPE', 'DA', 'N']]
metrics_df.to_csv('data/evaluation_metrics.csv', index=False)
print(f"\n✓ Metrics saved to: data/evaluation_metrics.csv")

# ============================================================================
# STAGE 7: TARGET ACHIEVEMENT REPORT
# ============================================================================
print("\n" + "=" * 70)
print("TARGET ACHIEVEMENT REPORT")
print("=" * 70)

if enhanced_available:
    enhanced_rmse = results['Enhanced Hybrid']['RMSE']
    enhanced_da = results['Enhanced Hybrid']['DA']
    
    print(f"\n📊 ENHANCED HYBRID MODEL RESULTS:")
    print(f"   DA:   {enhanced_da:.1f}% (Target: ≥70%) {'✅ ACHIEVED' if enhanced_da >= 70 else '⚠️ Below target'}")
    print(f"   RMSE: {enhanced_rmse:.4f} vs RW: {rw_rmse:.4f}")
    
    if enhanced_rmse < rw_rmse:
        improvement_pct = ((rw_rmse - enhanced_rmse) / rw_rmse) * 100
        print(f"   ✅ RMSE BETTER than Random Walk by {improvement_pct:.1f}%")
    else:
        worse_pct = ((enhanced_rmse - rw_rmse) / rw_rmse) * 100
        print(f"   ⚠️ RMSE {worse_pct:.1f}% worse than Random Walk")
    
    print(f"\n🎯 TARGETS:")
    target1 = enhanced_da >= 70
    target2 = enhanced_rmse < rw_rmse
    
    print(f"   1. DA ≥ 70%:                  {'✅ YES' if target1 else '❌ NO'}")
    print(f"   2. RMSE < Random Walk:        {'✅ YES' if target2 else '❌ NO'}")
    
    if target1 and target2:
        print(f"\n   ✨ ALL TARGETS ACHIEVED! ✨")
    else:
        print(f"\n   ⚠️ Some targets not yet achieved - see optimization suggestions below")
else:
    print("\n❌ Enhanced model training failed. Using baseline comparison only.")
    
    # Baseline comparison
    best_baseline = 'Random Walk' if rw_rmse < arima_rmse else 'ARIMA'
    print(f"\n📊 BASELINE COMPARISON:")
    print(f"   Random Walk DA: {rw_da:.1f}%")
    print(f"   ARIMA DA: {arima_da:.1f}%")

# ============================================================================
# STAGE 8: OPTIMIZATION SUGGESTIONS
# ============================================================================
print("\n" + "=" * 70)
print("OPTIMIZATION SUGGESTIONS")
print("=" * 70)

if enhanced_available and (results['Enhanced Hybrid']['DA'] < 70 or results['Enhanced Hybrid']['RMSE'] >= rw_rmse):
    print("\n🔧 To improve DA towards 70%:")
    print("   1. Increase DirectionalBooster weight in ensemble")
    print("   2. Expand feature set with more technical indicators")
    print("   3. Use multi-model voting for direction prediction")
    print("   4. Apply stricter confidence thresholds")
    print("   5. Implement adaptive weights based on regime changes")
    
    if results['Enhanced Hybrid']['RMSE'] >= rw_rmse:
        print("\n🔧 To reduce RMSE below Random Walk:")
        print("   1. Increase residual correction model capacity")
        print("   2. Use ensemble of base models (GB, RF, SVR)")
        print("   3. Implement adaptive weighting strategy")
        print("   4. Add stacking meta-learner on predictions")
        print("   5. Apply recursive forecasting with retraining")
else:
    print("\n✨ All targets achieved! Consider:")
    print("   1. Cross-validation on different time periods")
    print("   2. Stress testing on volatile market conditions")
    print("   3. Real-world deployment and monitoring")

# ============================================================================
# SUMMARY
# ============================================================================
duration = (datetime.now() - start_time).total_seconds()

print("\n" + "=" * 70)
print("PIPELINE SUMMARY")
print("=" * 70)
print(f"Execution Time: {duration:.1f} seconds")
print(f"\nOutput Files:")
print(f"  ✓ data/raw_data.csv")
print(f"  ✓ data/processed_data.csv")
print(f"  ✓ data/train_data.csv, val_data.csv, test_data.csv")
print(f"  ✓ data/evaluation_metrics.csv")
print("\n" + "=" * 70)
print("✨ ENHANCED PIPELINE COMPLETE! ✨")
print("=" * 70 + "\n")
