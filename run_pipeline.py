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
from src.information_analysis import run_information_analysis
from src.preprocessing import DataPreprocessor, DataSplitter
from src.models import ARIMAModel, RandomWalkModel, MovingAverageModel, HybridARIMALSTM
from src.hybrid_model import MeanReversionModel, MeanReversionStreakModel
from src.evaluation import ModelEvaluator, EnsembleUtils, WalkForwardValidator, RegimeEvaluator

# Create output directories
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('figures', exist_ok=True)


def run_pipeline(
    verbose=True,
    ensemble_mode='validation',
    run_walk_forward=False,
    runtime_profile='fast',
):
    """
    Run the complete forecasting pipeline.

    Returns:
        dict: Results containing models, predictions, and metrics
    """
    if verbose:
        print("\n" + "=" * 70)
        print("USD-NGN EXCHANGE RATE FORECASTING PIPELINE")
        print("=" * 70)
        print(f"Runtime profile: {runtime_profile}")

    start_time = datetime.now()
    te_bootstrap = 40 if runtime_profile == 'fast' else 100

    # ========================================================================
    # STAGE 1: DATA COLLECTION
    # ========================================================================
    if verbose:
        print("\n[STAGE 1] DATA COLLECTION")
        print("-" * 70)

    collector = DataCollector(start_date='1995-01-01', end_date='2024-12-31')
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

    # Shift features to avoid look-ahead bias.
    # Exclude lag columns — they are already shifted by definition.
    lag_cols = [c for c in processed_data.columns if '_lag' in c]
    feature_cols_to_shift = [
        c for c in processed_data.columns
        if c != 'usdngn' and c not in lag_cols
    ]
    processed_data[feature_cols_to_shift] = processed_data[feature_cols_to_shift].shift(1)
    processed_data = processed_data.dropna()
    processed_data.to_csv('data/processed_data.csv')

    if verbose:
        print(f"  Processed (lagged features): {processed_data.shape[0]} observations, {processed_data.shape[1]} features")
        print(f"  Feature shift applied to {len(feature_cols_to_shift)} columns (excluded {len(lag_cols)} lag columns)")
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

    info_results = run_information_analysis(
        train_data,
        target='usdngn',
        source_vars=available_features,
        lag=1,
        alpha=0.65,
        n_bootstrap=te_bootstrap,
    )
    feature_weights_df = info_results['feature_weights']
    feature_weight_map = info_results['weight_map']
    feature_weight_vector = np.array([feature_weight_map.get(f, 1.0) for f in available_features], dtype=float)

    feature_weights_df.to_csv('data/feature_weights.csv', index=False)
    info_results['te_results'].to_csv('data/transfer_entropy_scores.csv', index=False)
    info_results['mi_results'].to_csv('data/mutual_information_scores.csv', index=False)

    X_train = np.nan_to_num(train_data[available_features].values, nan=0)
    y_train = train_data['usdngn'].values
    X_val = np.nan_to_num(val_data[available_features].values, nan=0)
    y_val = val_data['usdngn'].values
    X_test = np.nan_to_num(test_data[available_features].values, nan=0)
    y_test = test_data['usdngn'].values

    if verbose:
        print(f"  Features: {len(available_features)}")
        print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        if len(feature_weights_df) > 0:
            top_weights = feature_weights_df.head(min(5, len(feature_weights_df)))
            top_str = ", ".join(
                f"{row['feature']}={row['weight']:.2f}" for _, row in top_weights.iterrows()
            )
            print(f"  Top information-weighted features: {top_str}")
        print("  Saved to: data/feature_weights.csv, transfer_entropy_scores.csv, mutual_information_scores.csv")

    # ========================================================================
    # STAGE 5: MODEL TRAINING
    # ========================================================================
    if verbose:
        print("\n[STAGE 5] MODEL TRAINING")
        print("-" * 70)

    models = {}
    rolling_predictions = {}
    val_rolling_predictions = {}

    # 5.1 Random Walk (Baseline)
    if verbose:
        print("\n  [5.1] Random Walk (Baseline)...")
    history = list(y_train)
    rw_roll = np.zeros(len(y_test))
    for i in range(len(y_test)):
        rw_roll[i] = history[-1]
        history.append(y_test[i])
    rolling_predictions['Random Walk'] = rw_roll
    # Validation rolling predictions
    history_val = list(y_train)
    rw_val_roll = np.zeros(len(y_val))
    for i in range(len(y_val)):
        rw_val_roll[i] = history_val[-1]
        history_val.append(y_val[i])
    val_rolling_predictions['Random Walk'] = rw_val_roll
    if verbose:
        print("        Trained (rolling one-step-ahead)")

    # 5.2 Moving Average (Baseline)
    if verbose:
        print("\n  [5.2] Moving Average (Baseline)...")
    ma_model = MovingAverageModel(window=20)
    ma_model.fit(y_train)
    models['Moving Average'] = ma_model
    history = list(y_train)
    ma_roll = np.zeros(len(y_test))
    window = ma_model.window
    for i in range(len(y_test)):
        window_vals = history[-window:] if len(history) >= window else history
        ma_roll[i] = np.mean(window_vals)
        history.append(y_test[i])
    rolling_predictions['Moving Average'] = ma_roll
    # Validation rolling predictions
    history_val = list(y_train)
    ma_val_roll = np.zeros(len(y_val))
    for i in range(len(y_val)):
        window_vals = history_val[-window:] if len(history_val) >= window else history_val
        ma_val_roll[i] = np.mean(window_vals)
        history_val.append(y_val[i])
    val_rolling_predictions['Moving Average'] = ma_val_roll
    if verbose:
        print("        Trained (rolling one-step-ahead)")

    # 5.3 ARIMA
    if verbose:
        print("\n  [5.3] ARIMA...")
    arima = ARIMAModel()
    arima.fit(y_train, verbose=verbose)
    models['ARIMA'] = arima
    rolling_predictions['ARIMA'] = arima.predict_rolling(y_test, verbose=False)
    # Validation rolling predictions
    val_rolling_predictions['ARIMA'] = arima.predict_rolling(y_val, verbose=False)
    if verbose:
        print("        Trained (rolling one-step-ahead)")

    # 5.4 Hybrid ARIMA-LSTM
    if verbose:
        print("\n  [5.4] Hybrid ARIMA-LSTM...")
    hybrid_baseline = HybridARIMALSTM(
        arima_order=arima.best_order,
        feature_weights=feature_weight_vector,
    )
    hybrid_baseline.fit(X_train, y_train, X_val, y_val, verbose=False)
    models['Hybrid ARIMA-LSTM'] = hybrid_baseline
    rolling_predictions['Hybrid ARIMA-LSTM'] = hybrid_baseline.predict_rolling(X_test, y_test)
    # Validation rolling predictions
    val_rolling_predictions['Hybrid ARIMA-LSTM'] = hybrid_baseline.predict_rolling(X_val, y_val)

    # Also get directional ensemble predictions if available
    direction_results = None
    if hybrid_baseline.directional_ensemble is not None and hybrid_baseline.directional_ensemble.is_fitted:
        try:
            dir_preds = hybrid_baseline.predict_with_direction(
                X_test,
                y_context=y_train,
                y_actual=y_test,
                use_rolling_levels=True
            )
            direction_results = hybrid_baseline.compute_directional_accuracy(y_test, dir_preds)
        except Exception as e:
            if verbose:
                print(f"        Directional ensemble evaluation failed: {e}")

    if verbose:
        print("        Trained (rolling one-step-ahead + directional ensemble)")
        if direction_results is not None:
            print(f"        Directional Ensemble DA: {direction_results['overall_da']:.1%}"
                  f" (p={direction_results['p_value']:.4f},"
                  f" {'significant' if direction_results['significant'] else 'not significant'})")

    # 5.5 Mean Reversion (rule-based baseline)
    if verbose:
        print("\n  [5.5] Mean Reversion (rule-based baseline)...")
    mr_model = MeanReversionModel()
    mr_model.fit(X_train, y_train, X_val, y_val, verbose=False)
    models['Mean Reversion'] = mr_model
    rolling_predictions['Mean Reversion'] = mr_model.predict(X_test, y_train, y_test_actual=y_test)
    # Validation rolling predictions
    val_rolling_predictions['Mean Reversion'] = mr_model.predict(X_val, y_train, y_test_actual=y_val)
    if verbose:
        print("        Trained (rolling one-step-ahead)")

    # 5.6 Mean Reversion + Streak (rule-based baseline)
    if verbose:
        print("\n  [5.6] Mean Reversion + Streak (rule-based baseline)...")
    mrs_model = MeanReversionStreakModel()
    mrs_model.fit(X_train, y_train, X_val, y_val, verbose=False)
    models['Mean Reversion + Streak'] = mrs_model
    rolling_predictions['Mean Reversion + Streak'] = mrs_model.predict(X_test, y_train, y_test_actual=y_test)
    # Validation rolling predictions
    val_rolling_predictions['Mean Reversion + Streak'] = mrs_model.predict(X_val, y_train, y_test_actual=y_val)
    if verbose:
        print("        Trained (rolling one-step-ahead)")

    # 5.7 Ensemble (validation-weighted by default)
    ensemble_members = {
        k: v for k, v in rolling_predictions.items()
        if k not in {'Random Walk', 'Moving Average'}
    }
    ensemble_val_members = {
        k: v for k, v in val_rolling_predictions.items()
        if k in ensemble_members
    }

    ensemble_weights = None
    if ensemble_mode == 'validation' and ensemble_val_members:
        val_rows = []
        for name, pred in ensemble_val_members.items():
            m = ModelEvaluator.compute_all_metrics(y_val, pred)
            val_rows.append({'Model': name, **m})
        val_metrics_df = pd.DataFrame(val_rows)
        # Weight by inverse RMSE (lower RMSE -> higher weight).
        inv_rmse = 1.0 / (val_metrics_df['RMSE'].values + 1e-8)
        raw_weights = dict(zip(val_metrics_df['Model'], inv_rmse))
        ensemble_weights = EnsembleUtils.normalize_weights(raw_weights, list(ensemble_members.keys()))
        weights_df = pd.DataFrame(
            [{'Model': k, 'Weight': v} for k, v in ensemble_weights.items()]
        ).sort_values('Weight', ascending=False)
        weights_df.to_csv('data/ensemble_weights.csv', index=False)

    prev_values_val = np.concatenate([[y_train[-1]], y_val[:-1]])
    optimized_ensemble_weights = EnsembleUtils.optimize_weights(
        ensemble_val_members,
        y_val,
        prev_values=prev_values_val,
        objective='balanced',
        n_samples=300,
        random_state=42,
    ) if ensemble_val_members else None

    ensemble_pred_simple = EnsembleUtils.weighted_average(ensemble_members, weights=None)
    if len(ensemble_pred_simple) > 0:
        rolling_predictions['Ensemble (Simple Avg)'] = ensemble_pred_simple

    if ensemble_weights is not None:
        ensemble_pred_weighted = EnsembleUtils.weighted_average(ensemble_members, weights=ensemble_weights)
        if len(ensemble_pred_weighted) > 0:
            rolling_predictions['Ensemble (Val-Weighted)'] = ensemble_pred_weighted

    if optimized_ensemble_weights is not None:
        ensemble_pred_optimized = EnsembleUtils.weighted_average(
            ensemble_members,
            weights=optimized_ensemble_weights
        )
        if len(ensemble_pred_optimized) > 0:
            rolling_predictions['Ensemble (Optimized)'] = ensemble_pred_optimized
        optimized_weights_df = pd.DataFrame(
            [{'Model': k, 'Weight': v} for k, v in optimized_ensemble_weights.items()]
        ).sort_values('Weight', ascending=False)
        optimized_weights_df.to_csv('data/ensemble_weights_optimized.csv', index=False)

    if verbose:
        print("\n  [5.7] Ensembles...")
        print(f"        Members: {', '.join(ensemble_members.keys())}")
        if ensemble_weights is not None:
            top = sorted(ensemble_weights.items(), key=lambda kv: kv[1], reverse=True)[:3]
            top_str = ", ".join([f"{k}={v:.2f}" for k, v in top])
            print(f"        Validation-weighted ensemble enabled ({top_str})")
            print("        Weights saved to: data/ensemble_weights.csv")
        else:
            print("        Simple average ensemble enabled")
        if optimized_ensemble_weights is not None:
            top_opt = sorted(optimized_ensemble_weights.items(), key=lambda kv: kv[1], reverse=True)[:3]
            top_opt_str = ", ".join([f"{k}={v:.2f}" for k, v in top_opt])
            print(f"        Optimized ensemble enabled ({top_opt_str})")
            print("        Weights saved to: data/ensemble_weights_optimized.csv")

    # Optional: lightweight walk-forward validation hook
    walk_forward_results = None
    if run_walk_forward:
        try:
            combined = np.concatenate([y_train, y_val])
            wf = WalkForwardValidator.validate_univariate(
                y=combined,
                model_factory=lambda: RandomWalkModel(),
                initial_train_size=max(200, int(0.5 * len(combined))),
                horizon=1,
                step_size=1,
            )
            walk_forward_results = wf
            wf_df = pd.DataFrame({
                'anchor': wf['anchors'],
                'y_true': wf['actuals'],
                'y_pred': wf['predictions'],
            })
            wf_df.to_csv('data/walk_forward_random_walk.csv', index=False)
            if verbose:
                print("\n  [WF] Random Walk walk-forward results saved to: data/walk_forward_random_walk.csv")
        except Exception as exc:
            if verbose:
                print(f"\n  [WF] Walk-forward validation failed: {exc}")

    # ========================================================================
    # STAGE 6: EVALUATION (all metrics on rolling one-step-ahead predictions)
    # ========================================================================
    if verbose:
        print("\n[STAGE 6] EVALUATION")
        print("-" * 70)
        print("  All metrics computed on rolling one-step-ahead predictions")
        print("  DA uses proper one-step-ahead formula (pred vs prev_actual)")

    # For proper one-step-ahead DA: prev_values[i] = value known when predicting y_test[i]
    # At i=0, we know y_train[-1]. At i>=1, we know y_test[i-1].
    prev_values = np.concatenate([[y_train[-1]], y_test[:-1]])

    results = []

    for name, roll_pred in rolling_predictions.items():
        roll_len = min(len(y_test), len(roll_pred))
        metrics = ModelEvaluator.compute_all_metrics(
            y_test[:roll_len],
            roll_pred[:roll_len],
            prev_values=prev_values[:roll_len]
        )
        metrics['Model'] = name
        results.append(metrics)

    results_df = pd.DataFrame(results)

    # Add directional ensemble DA as a separate column if available
    results_df['DA_Ensemble'] = np.nan
    if direction_results is not None:
        mask = results_df['Model'] == 'Hybrid ARIMA-LSTM'
        results_df.loc[mask, 'DA_Ensemble'] = direction_results['overall_da'] * 100

    regime_frames = []
    test_dates = test_data.index if isinstance(test_data.index, pd.DatetimeIndex) else pd.to_datetime(test_data.index)
    for name, roll_pred in rolling_predictions.items():
        regime_df = RegimeEvaluator.evaluate_by_regime(y_test, roll_pred, test_dates)
        if len(regime_df) > 0:
            regime_df = regime_df.reset_index().rename(columns={'index': 'Regime'})
            regime_df.insert(0, 'Model', name)
            regime_frames.append(regime_df)

    regime_results_df = pd.concat(regime_frames, ignore_index=True) if regime_frames else pd.DataFrame()
    if len(regime_results_df) > 0:
        regime_results_df.to_csv('data/regime_evaluation.csv', index=False)

    results_df.to_csv('data/evaluation_metrics.csv', index=False)

    # Get Random Walk baseline for comparison
    rw_metrics = results_df[results_df['Model'] == 'Random Walk'].iloc[0]
    rw_rmse = rw_metrics['RMSE']
    rw_da = rw_metrics['DA']

    if verbose:
        print(f"\n  {'Model':<25} {'RMSE':<12} {'MAE':<12} {'DA':<12} {'Status'}")
        print("  " + "-" * 72)

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

            print(f"  {row['Model']:<25} {row['RMSE']:<12.4f} {row['MAE']:<12.4f} {row['DA']:<12.1f}% {status}")

        if direction_results is not None:
            print(f"\n  Hybrid ARIMA-LSTM Directional Ensemble:")
            print(f"    Overall DA:    {direction_results['overall_da']:.1%}")
            print(f"    Hi-Conf DA:    {direction_results['high_conf_da']:.1%}"
                  f" (coverage: {direction_results['high_conf_coverage']:.0%})")
            print(f"    p-value:       {direction_results['p_value']:.4f}"
                  f" ({'significant' if direction_results['significant'] else 'not significant'})")

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
        print(f"    - data/feature_weights.csv, transfer_entropy_scores.csv, mutual_information_scores.csv")
        print(f"    - data/evaluation_metrics.csv")
        if len(regime_results_df) > 0:
            print(f"    - data/regime_evaluation.csv")

        # Check for winners
        winners = results_df[(results_df['RMSE'] < rw_rmse) & (results_df['DA'] > rw_da)]
        if len(winners) > 0:
            print(f"\n  Models that beat Random Walk on BOTH metrics:")
            for _, row in winners.iterrows():
                print(f"    - {row['Model']}: RMSE={row['RMSE']:.4f}, DA={row['DA']:.1f}%")

        print("\n" + "=" * 70 + "\n")

    return {
        'models': models,
        'predictions': rolling_predictions,
        'rolling_predictions': rolling_predictions,
        'metrics': results_df,
        'analysis': info_results,
        'direction_results': direction_results,
        'ensemble_weights': ensemble_weights,
        'optimized_ensemble_weights': optimized_ensemble_weights,
        'walk_forward': walk_forward_results,
        'regime_analysis': regime_results_df,
        'data': {
            'raw': raw_data,
            'processed': processed_data,
            'stationarity': stationarity,
            'train_df': train_data,
            'val_df': val_data,
            'test_df': test_data,
            'feature_names': available_features,
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
    }


if __name__ == "__main__":
    results = run_pipeline(verbose=True)
