#!/usr/bin/env python3
"""
COMPREHENSIVE VALIDATION TEST
Verifies all improvements and target achievement
"""

import numpy as np
import pandas as pd
import os
import sys

print("\n" + "="*80)
print(" " * 20 + "USD-NGN FORECASTING - IMPROVEMENTS VALIDATION")
print("="*80)

# Check data availability
data_exists = os.path.exists('data/train_data.csv') and \
              os.path.exists('data/val_data.csv') and \
              os.path.exists('data/test_data.csv')

if not data_exists:
    print("\n⚠️  Data files not found. Running data collection & preprocessing...")
    try:
        from part1_data_collection import DataCollector
        from part2_preprocessing import DataPreprocessor, DataSplitter
        
        collector = DataCollector(start_date='2023-01-01', end_date='2025-12-31')
        raw_data = collector.collect_all_data()
        
        preprocessor = DataPreprocessor(raw_data)
        processed_data, _ = preprocessor.preprocess()
        
        splitter = DataSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
        train_data, val_data, test_data = splitter.split(processed_data)
        
        train_data.to_csv('data/train_data.csv', index=False)
        val_data.to_csv('data/val_data.csv', index=False)
        test_data.to_csv('data/test_data.csv', index=False)
        
        print("✅ Data generated and saved")
    except Exception as e:
        print(f"❌ Failed to generate data: {e}")
        sys.exit(1)

# Load data
print("\n[1] Loading data...")
train_data = pd.read_csv('data/train_data.csv')
val_data = pd.read_csv('data/val_data.csv')
test_data = pd.read_csv('data/test_data.csv')

features = ['brent_oil', 'mpr', 'cpi', 'oil_return', 'usdngn_volatility',
            'usdngn_ma5', 'usdngn_ma20', 'rate_oil_ratio', 'mpr_change']
features = [f for f in features if f in train_data.columns]

X_train = np.nan_to_num(train_data[features].values, nan=0, posinf=0, neginf=0)
y_train = train_data['usdngn'].values
X_test = np.nan_to_num(test_data[features].values, nan=0, posinf=0, neginf=0)
y_test = test_data['usdngn'].values
X_val = np.nan_to_num(val_data[features].values, nan=0, posinf=0, neginf=0)
y_val = val_data['usdngn'].values

print(f"✅ Loaded: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")

# Import evaluator
from part5_evaluation import ModelEvaluator

# 1. Baseline
print("\n[2] Baseline Models...")
from part4_models import RandomWalkModel, ARIMAModel

rw = RandomWalkModel().fit(y_train)
rw_pred = np.concatenate([[y_train[-1]], y_test[:-1]])
rw_metrics = ModelEvaluator.compute_all_metrics(y_test, rw_pred)
print(f"✅ Random Walk: RMSE={rw_metrics['RMSE']:.4f}, DA={rw_metrics['DA']:.1f}%")

# 2. Optimized Models
print("\n[3] Testing Optimized Models...")

results = {
    'Random Walk': rw_metrics
}

models_to_test = [
    ('Optimized Hybrid', 'final_optimized_models', 'OptimizedHybrid'),
    ('Voting Ensemble', 'final_optimized_models', 'VotingEnsemble'),
    ('Hybrid Enhanced', 'part4_models', 'HybridARIMALSTM')
]

for model_name, module_name, class_name in models_to_test:
    try:
        # Import model
        if module_name == 'final_optimized_models':
            from final_optimized_models import OptimizedHybrid, VotingEnsemble
            if class_name == 'OptimizedHybrid':
                model = OptimizedHybrid()
            else:
                model = VotingEnsemble()
        else:
            from part4_models import HybridARIMALSTM, ARIMAModel
            arima = ARIMAModel()
            arima.fit(y_train, verbose=False)
            model = HybridARIMALSTM(
                arima_order=arima.best_order,
                sequence_length=40,
                arima_weight=0.25,
                use_directional_ensemble=True
            )
        
        # Train
        print(f"  📊 {model_name}...", end=" ", flush=True)
        model.fit(X_train, y_train, X_val, y_val, verbose=False)
        
        # Predict
        pred = model.predict(X_test)
        
        # Align lengths
        min_len = min(len(y_test), len(pred))
        metrics = ModelEvaluator.compute_all_metrics(y_test[-min_len:], pred[-min_len:])
        results[model_name] = metrics
        
        # Check targets
        rmse_ok = metrics['RMSE'] < rw_metrics['RMSE']
        da_ok = metrics['DA'] >= 70
        
        status = "✅" if (rmse_ok and da_ok) else ("⚠️ " if (rmse_ok or da_ok) else "❌")
        print(f"{status} RMSE={metrics['RMSE']:.2f} DA={metrics['DA']:.1f}%")
        
    except Exception as e:
        print(f"❌ {model_name}: {str(e)[:50]}")
        results[model_name] = None

# 3. Summary
print("\n" + "="*80)
print("TARGET ACHIEVEMENT SUMMARY")
print("="*80)

print(f"\n📋 TARGETS:")
print(f"   1. Directional Accuracy ≥ 70%")
print(f"   2. RMSE < Random Walk ({rw_metrics['RMSE']:.4f})")

print(f"\n📊 MODEL COMPARISON:\n")
print(f"{'Model':<20} {'RMSE':<12} {'DA':<10} {'RMSE OK?':<10} {'DA OK?':<10} {'Status':<15}")
print("-" * 80)

target_achieved = False

for model_name, metrics in results.items():
    if metrics is None:
        print(f"{model_name:<20} {'ERROR':<12}")
        continue
    
    rmse = metrics['RMSE']
    da = metrics['DA']
    rmse_ok = rmse < rw_metrics['RMSE']
    da_ok = da >= 70
    both_ok = rmse_ok and da_ok
    
    if both_ok:
        target_achieved = True
    
    status = "✅ PASS" if both_ok else ("⚠️ PARTIAL" if (rmse_ok or da_ok) else "❌ FAIL")
    
    print(f"{model_name:<20} {rmse:<12.4f} {da:<10.1f} {str(rmse_ok):<10} {str(da_ok):<10} {status:<15}")

# Final verdict
print("\n" + "="*80)
print("FINAL VERDICT")
print("="*80)

if target_achieved:
    print("\n✨ SUCCESS! At least one model achieves both targets:")
    print("   ✅ Directional Accuracy ≥ 70%")
    print("   ✅ RMSE < Random Walk")
    print("\n🎉 IMPROVEMENTS COMPLETE AND VALIDATED!")
else:
    print("\n⚠️  Targets not yet fully achieved by all models")
    print("   Review optimization strategies in IMPROVEMENTS_SUMMARY.md")

# Save results
results_df = pd.DataFrame([
    {
        'Model': name,
        'RMSE': metrics['RMSE'] if metrics else np.nan,
        'MAE': metrics['MAE'] if metrics else np.nan,
        'MAPE': metrics['MAPE'] if metrics else np.nan,
        'DA': metrics['DA'] if metrics else np.nan,
        'N': metrics['N'] if metrics else np.nan
    }
    for name, metrics in results.items()
])

results_df.to_csv('data/validation_results.csv', index=False)
print(f"\n📁 Results saved to: data/validation_results.csv")

print("\n" + "="*80)
print("✓ VALIDATION COMPLETE")
print("="*80 + "\n")
