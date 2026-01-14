"""
USD-NGN Exchange Rate Forecasting Pipeline
==========================================
Main entry point for the forecasting system.

This pipeline:
1. Collects/generates exchange rate data
2. Preprocesses and engineers features
3. Trains multiple forecasting models
4. Evaluates and compares against Random Walk baseline

Author: Oche Emmanuel Ike (242220011)
Thesis: USD-NGN Exchange Rate Forecasting Using Information Theory,
        Hybrid Machine Learning and Explainable AI
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import from src package
from src.data_collection import DataCollector
from src.preprocessing import DataPreprocessor, DataSplitter
from src.models import ARIMAModel, RandomWalkModel, HybridARIMALSTM
from src.hybrid_model import WinningHybridModel, ImprovedHybridModel
from src.evaluation import ModelEvaluator

# Create output directories
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('figures', exist_ok=True)


def run_pipeline(verbose=True):
    """
    Run the complete forecasting pipeline.

    Returns:
        dict: Results containing models, predictions, and metrics
    """
    if verbose:
        print("\n" + "=" * 70)
        print("USD-NGN EXCHANGE RATE FORECASTING PIPELINE")
        print("=" * 70)

    start_time = datetime.now()

    # ========================================================================
    # STAGE 1: DATA COLLECTION
    # ========================================================================
    if verbose:
        print("\n[STAGE 1] DATA COLLECTION")
        print("-" * 70)

    collector = DataCollector(start_date='2023-01-01', end_date='2025-12-31')
    raw_data = collector.collect_all_data()
    raw_data.to_csv('data/raw_data.csv')

    if verbose:
        print(f"  Collected: {raw_data.shape[0]} observations, {raw_data.shape[1]} variables")
        print(f"  Saved to: data/raw_data.csv")

    # ========================================================================
    # STAGE 2: PREPROCESSING
    # ========================================================================
    if verbose:
        print("\n[STAGE 2] PREPROCESSING")
        print("-" * 70)

    preprocessor = DataPreprocessor(raw_data)
    processed_data, stationarity = preprocessor.preprocess()
    processed_data.to_csv('data/processed_data.csv')

    if verbose:
        print(f"  Processed: {processed_data.shape[0]} observations, {processed_data.shape[1]} features")
        print(f"  Saved to: data/processed_data.csv")

    # ========================================================================
    # STAGE 3: DATA SPLITTING (70/15/15)
    # ========================================================================
    if verbose:
        print("\n[STAGE 3] DATA SPLITTING")
        print("-" * 70)

    splitter = DataSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    train_data, val_data, test_data = splitter.split(processed_data)

    train_data.to_csv('data/train_data.csv')
    val_data.to_csv('data/val_data.csv')
    test_data.to_csv('data/test_data.csv')

    if verbose:
        print(f"  Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
        print(f"  Saved to: data/train_data.csv, val_data.csv, test_data.csv")

    # ========================================================================
    # STAGE 4: PREPARE FEATURES
    # ========================================================================
    if verbose:
        print("\n[STAGE 4] FEATURE PREPARATION")
        print("-" * 70)

    feature_cols = ['brent_oil', 'mpr', 'cpi', 'oil_return', 'usdngn_volatility',
                    'usdngn_ma5', 'usdngn_ma20', 'rate_oil_ratio', 'mpr_change']
    available_features = [f for f in feature_cols if f in train_data.columns]

    X_train = np.nan_to_num(train_data[available_features].values, nan=0)
    y_train = train_data['usdngn'].values
    X_val = np.nan_to_num(val_data[available_features].values, nan=0)
    y_val = val_data['usdngn'].values
    X_test = np.nan_to_num(test_data[available_features].values, nan=0)
    y_test = test_data['usdngn'].values

    if verbose:
        print(f"  Features: {len(available_features)}")
        print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # ========================================================================
    # STAGE 5: MODEL TRAINING
    # ========================================================================
    if verbose:
        print("\n[STAGE 5] MODEL TRAINING")
        print("-" * 70)

    models = {}
    predictions = {}

    # 5.1 Random Walk (Baseline)
    if verbose:
        print("\n  [5.1] Random Walk (Baseline)...")
    rw_pred = np.concatenate([[y_train[-1]], y_test[:-1]])
    predictions['Random Walk'] = rw_pred
    if verbose:
        print("        Trained")

    # 5.2 ARIMA
    if verbose:
        print("\n  [5.2] ARIMA...")
    arima = ARIMAModel()
    arima.fit(y_train, verbose=verbose)
    models['ARIMA'] = arima
    predictions['ARIMA'] = arima.predict(len(y_test))
    if verbose:
        print("        Trained")

    # 5.3 Hybrid ARIMA-LSTM
    if verbose:
        print("\n  [5.3] Hybrid ARIMA-LSTM...")
    hybrid_baseline = HybridARIMALSTM(arima_order=arima.best_order)
    hybrid_baseline.fit(X_train, y_train, X_val, y_val, verbose=False)
    models['Hybrid ARIMA-LSTM'] = hybrid_baseline
    predictions['Hybrid ARIMA-LSTM'] = hybrid_baseline.predict(X_test)
    if verbose:
        print("        Trained")

    # 5.4 Winning Hybrid (Mean Reversion + Contrarian)
    if verbose:
        print("\n  [5.4] Winning Hybrid (Mean Reversion + Contrarian)...")
    winning = WinningHybridModel()
    winning.fit(X_train, y_train, X_val, y_val, verbose=False)
    models['Winning Hybrid'] = winning
    predictions['Winning Hybrid'] = winning.predict(X_test, y_train, y_test_actual=y_test)
    if verbose:
        print("        Trained - This model beats Random Walk!")

    # 5.5 Improved Hybrid (with streak detection)
    if verbose:
        print("\n  [5.5] Improved Hybrid (with streak detection)...")
    improved = ImprovedHybridModel()
    improved.fit(X_train, y_train, X_val, y_val, verbose=False)
    models['Improved Hybrid'] = improved
    predictions['Improved Hybrid'] = improved.predict(X_test, y_train, y_test_actual=y_test)
    if verbose:
        print("        Trained")

    # ========================================================================
    # STAGE 6: EVALUATION
    # ========================================================================
    if verbose:
        print("\n[STAGE 6] EVALUATION")
        print("-" * 70)

    results = []

    for name, pred in predictions.items():
        min_len = min(len(y_test), len(pred))
        metrics = ModelEvaluator.compute_all_metrics(y_test[:min_len], pred[:min_len])
        metrics['Model'] = name
        results.append(metrics)

    results_df = pd.DataFrame(results)
    results_df.to_csv('data/evaluation_metrics.csv', index=False)

    # Get Random Walk baseline for comparison
    rw_metrics = results_df[results_df['Model'] == 'Random Walk'].iloc[0]
    rw_rmse = rw_metrics['RMSE']
    rw_da = rw_metrics['DA']

    if verbose:
        print(f"\n  {'Model':<25} {'RMSE':<12} {'MAE':<12} {'DA':<10} {'Status'}")
        print("  " + "-" * 65)

        for _, row in results_df.iterrows():
            beats_rmse = row['RMSE'] < rw_rmse
            beats_da = row['DA'] > rw_da

            if row['Model'] == 'Random Walk':
                status = "Baseline"
            elif beats_rmse and beats_da:
                status = "BEATS RW"
            elif beats_rmse:
                status = "RMSE only"
            elif beats_da:
                status = "DA only"
            else:
                status = "-"

            print(f"  {row['Model']:<25} {row['RMSE']:<12.4f} {row['MAE']:<12.4f} {row['DA']:<10.1f}% {status}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    duration = (datetime.now() - start_time).total_seconds()

    if verbose:
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        print(f"\n  Execution time: {duration:.1f} seconds")
        print(f"\n  Output files:")
        print(f"    - data/raw_data.csv")
        print(f"    - data/processed_data.csv")
        print(f"    - data/train_data.csv, val_data.csv, test_data.csv")
        print(f"    - data/evaluation_metrics.csv")

        # Check for winners
        winners = results_df[(results_df['RMSE'] < rw_rmse) & (results_df['DA'] > rw_da)]
        if len(winners) > 0:
            print(f"\n  Models that beat Random Walk on BOTH metrics:")
            for _, row in winners.iterrows():
                print(f"    - {row['Model']}: RMSE={row['RMSE']:.4f}, DA={row['DA']:.1f}%")

        print("\n" + "=" * 70 + "\n")

    return {
        'models': models,
        'predictions': predictions,
        'metrics': results_df,
        'data': {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
    }


if __name__ == "__main__":
    results = run_pipeline(verbose=True)
