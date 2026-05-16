"""
Standalone LSTM pipeline for USD-NGN forecasting.

This script intentionally keeps LSTM out of the main benchmark pipeline while
making it easy to test recurrent modelling by itself.
"""

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

from src.data_collection import DataCollector
from src.preprocessing import DataPreprocessor, DataSplitter
from src.models import LSTMModel
from src.evaluation import ModelEvaluator


FEATURE_COLS = [
    'brent_oil', 'mpr', 'cpi', 'oil_return', 'usdngn_volatility',
    'usdngn_ma5', 'usdngn_ma20', 'rate_oil_ratio', 'mpr_change',
    'usdngn_lag1', 'usdngn_lag5', 'usdngn_lag10',
    'usdngn_return', 'usdngn_trend', 'usdngn_roc5',
    'time_idx', 'sin_doy', 'cos_doy',
]


def prepare_data_with_validation(seed=42, verbose=True):
    """Prepare data with comprehensive validation and NaN tracking."""
    np.random.seed(seed)
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    if verbose:
        print("\n[STAGE 0] DATA COLLECTION & PREPROCESSING")
        print("-" * 70)

    collector = DataCollector(start_date='1995-01-01', end_date='2025-12-31')
    raw_data = collector.collect_all_data(verbose=verbose)
    raw_data.to_csv('data/raw_data.csv')

    preprocessor = DataPreprocessor(raw_data)
    processed_data, _ = preprocessor.preprocess()
    processed_data.to_csv('data/processed_data.csv')

    splitter = DataSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    train_data, val_data, test_data = splitter.split(processed_data)

    train_data.to_csv('data/train_data.csv')
    val_data.to_csv('data/val_data.csv')
    test_data.to_csv('data/test_data.csv')

    available_features = [f for f in FEATURE_COLS if f in train_data.columns]
    
    missing_in_val = set(available_features) - set(val_data.columns)
    missing_in_test = set(available_features) - set(test_data.columns)
    
    if missing_in_val:
        raise ValueError(f"Features in train but missing in val: {missing_in_val}")
    if missing_in_test:
        raise ValueError(f"Features in train but missing in test: {missing_in_test}")
    
    if verbose:
        print(f"OK All {len(available_features)} features present in train/val/test")

    X_train_raw = train_data[available_features].values
    X_val_raw = val_data[available_features].values
    X_test_raw = test_data[available_features].values

    def log_nan_stats(X, split_name, feature_names):
        nan_mask = np.isnan(X)
        nan_count = nan_mask.sum()
        if nan_count > 0:
            print(f"\n[NaN Analysis] {split_name}: {nan_count} total NaN values")
            for i, fname in enumerate(feature_names):
                col_nans = nan_mask[:, i].sum()
                if col_nans > 0:
                    pct = 100 * col_nans / len(X)
                    print(f"  {fname}: {col_nans} ({pct:.2f}%)")
        else:
            print(f"OK {split_name}: No NaN values")

    log_nan_stats(X_train_raw, "Train", available_features)
    log_nan_stats(X_val_raw, "Validation", available_features)
    log_nan_stats(X_test_raw, "Test", available_features)

    X_train = np.nan_to_num(X_train_raw, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val_raw, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test_raw, nan=0.0, posinf=0.0, neginf=0.0)

    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1], \
        f"Feature dimension mismatch: train={X_train.shape[1]}, val={X_val.shape[1]}, test={X_test.shape[1]}"
    
    if verbose:
        print(f"OK Shape consistency: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    return train_data, val_data, test_data, available_features, X_train, X_val, X_test


def prepare_data(seed=42, verbose=True):
    """Backward compatibility wrapper."""
    return prepare_data_with_validation(seed, verbose)


def prepare_targets(train_data, val_data, test_data, target_mode='level'):
    """Prepare target variables with clear semantics."""
    y_test_level = test_data['usdngn'].values
    y_train_level = train_data['usdngn'].values
    y_val_level = val_data['usdngn'].values
    
    if target_mode == 'return':
        y_train = train_data['usdngn_return'].values
        y_val = val_data['usdngn_return'].values
        y_test = test_data['usdngn_return'].values
        
        y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
        y_val = np.nan_to_num(y_val, nan=0.0, posinf=0.0, neginf=0.0)
        y_test = np.nan_to_num(y_test, nan=0.0, posinf=0.0, neginf=0.0)
        
        return y_train, y_val, y_test, y_test_level
    
    elif target_mode == 'level':
        return y_train_level, y_val_level, y_test_level, y_test_level
    
    elif target_mode == 'log_price':
        y_train = np.log(np.maximum(y_train_level, 1e-8))
        y_val = np.log(np.maximum(y_val_level, 1e-8))
        y_test = np.log(np.maximum(y_test_level, 1e-8))
        
        y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
        y_val = np.nan_to_num(y_val, nan=0.0, posinf=0.0, neginf=0.0)
        y_test = np.nan_to_num(y_test, nan=0.0, posinf=0.0, neginf=0.0)
        
        return y_train, y_val, y_test, y_test_level
    
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}. Choose 'level', 'return', or 'log_price'.")


def reconstruct_levels_from_returns(return_pred, prev_levels):
    """Reconstruct price levels from log-return predictions."""
    return_pred = np.nan_to_num(return_pred, nan=0.0, posinf=0.0, neginf=0.0)
    return prev_levels * np.exp(return_pred)


def setup_device():
    """Setup PyTorch device with validation."""
    if not TORCH_AVAILABLE:
        print("\n[WARN] PyTorch not available - using CPU/numpy mode")
        return None

    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = torch.cuda.get_device_name(0)
        device_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n[INFO] GPU available: {device_name} ({device_mem:.1f} GB)")
        return device
    else:
        print("\n[INFO] GPU not available - using CPU")
        return torch.device('cpu')


def fit_lstm(args):
    start_time = datetime.now()

    print("\n" + "=" * 70)
    print("USD-NGN STANDALONE LSTM PIPELINE (CORRECTED)")
    print("=" * 70)
    
    device = setup_device()

    train_data, val_data, test_data, features, X_train, X_val, X_test = prepare_data_with_validation(
        seed=args.seed,
        verbose=args.verbose,
    )

    # Prepare targets with clear semantics
    y_train, y_val, y_test, y_test_level = prepare_targets(
        train_data, val_data, test_data, 
        target_mode=args.target_mode
    )
    
    y_val_level = val_data['usdngn'].values
    y_train_level = train_data['usdngn'].values

    print("\n[STAGE 1] LSTM TRAINING")
    print("-" * 70)
    print(f"Features: {len(features)}")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Target mode: {args.target_mode}")
    print(f"Sequence length: {args.sequence_length}, epochs: {args.epochs}, patience: {args.patience}")
    print(f"Architecture: LSTM({args.hidden_size1}) -> LSTM({args.hidden_size2}) -> FC(64)->FC(32)->FC(1)")

    model = LSTMModel(
        input_size=X_train.shape[1],
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        hidden_size1=args.hidden_size1,
        hidden_size2=args.hidden_size2,
        dropout=args.dropout,
    )
    model.fit(X_train, y_train, X_val, y_val, verbose=args.verbose)

    print("\n[STAGE 2] LSTM PREDICTION")
    print("-" * 70)
    
    # Make predictions with proper context alignment
    val_context = X_train
    test_context = np.vstack([X_train, X_val])
    
    val_pred_raw = model.predict(X_val, X_context=val_context, verbose_debug=False)
    test_pred_raw = model.predict(X_test, X_context=test_context, verbose_debug=False)

    # Reconstruct predictions to levels based on target mode
    if args.target_mode == 'return':
        prev_val = np.concatenate((np.array([y_train_level[-1]]), y_val_level[:-1]))
        prev_test = np.concatenate((np.array([y_val_level[-1]]), y_test_level[:-1]))

        val_pred = reconstruct_levels_from_returns(val_pred_raw, prev_val)
        test_pred = reconstruct_levels_from_returns(test_pred_raw, prev_test)
    elif args.target_mode == 'log_price':
        prev_val = np.concatenate((np.array([y_train_level[-1]]), y_val_level[:-1]))
        prev_test = np.concatenate((np.array([y_val_level[-1]]), y_test_level[:-1]))

        val_pred = np.exp(val_pred_raw)
        test_pred = np.exp(test_pred_raw)
    else:
        val_pred = val_pred_raw
        test_pred = test_pred_raw
        
        prev_val = np.concatenate((np.array([y_train_level[-1]]), y_val_level[:-1]))
        prev_test = np.concatenate((np.array([y_val_level[-1]]), y_test_level[:-1]))

    val_pred = np.where(np.isnan(val_pred), prev_val, val_pred)
    test_pred = np.where(np.isnan(test_pred), prev_test, test_pred)

    print("\n[STAGE 3] EVALUATION")
    print("-" * 70)
    
    val_metrics = ModelEvaluator.compute_all_metrics(y_val_level, val_pred, prev_values=prev_val)
    test_metrics = ModelEvaluator.compute_all_metrics(y_test_level, test_pred, prev_values=prev_test)

    print(f"Validation: RMSE={val_metrics['RMSE']:.4f}, MAE={val_metrics['MAE']:.4f}, "
          f"MAPE={val_metrics['MAPE']:.4f}, DA={val_metrics['DA']:.2f}%")
    print(f"Test:       RMSE={test_metrics['RMSE']:.4f}, MAE={test_metrics['MAE']:.4f}, "
          f"MAPE={test_metrics['MAPE']:.4f}, DA={test_metrics['DA']:.2f}%")

    metrics_df = pd.DataFrame([
        {'Split': 'Validation', 'Model': f"LSTM ({args.target_mode})", **val_metrics},
        {'Split': 'Test', 'Model': f"LSTM ({args.target_mode})", **test_metrics},
    ])
    metrics_df.to_csv('data/lstm_evaluation_metrics.csv', index=False)

    # Enhanced predictions output with analysis columns
    test_actual = test_data['usdngn'].values
    predictions_df = pd.DataFrame({
        'date': test_data.index,
        'actual': test_actual,
        'previous_actual': prev_test,
        'lstm_prediction': test_pred,
        'raw_lstm_output': test_pred_raw,
        'abs_error': np.abs(test_actual - test_pred),
        'pct_error': 100 * np.abs(test_actual - test_pred) / np.abs(test_actual),
        'actual_direction': (test_actual > prev_test).astype(int),
        'predicted_direction': (test_pred > prev_test).astype(int),
        'direction_correct': ((test_actual > prev_test) == (test_pred > prev_test)).astype(int),
    })
    predictions_df.to_csv('data/lstm_predictions.csv', index=False)

    if args.save_model and TORCH_AVAILABLE and model.model is not None:
        torch.save(model.model.state_dict(), 'models/lstm_model.pt')
        print("\nOK Saved model weights to: models/lstm_model.pt")

    duration = (datetime.now() - start_time).total_seconds()
    
    print("\n[STAGE 4] SUMMARY")
    print("-" * 70)
    print("Output files:")
    print("  - data/lstm_evaluation_metrics.csv")
    print("  - data/lstm_predictions.csv")
    if args.save_model:
        print("  - models/lstm_model.pt")
    print(f"\nExecution time: {duration:.1f} seconds")
    print("=" * 70 + "\n")

    return {
        'model_path': 'models/lstm_model.pt' if args.save_model else None,
        'metrics_csv': 'data/lstm_evaluation_metrics.csv',
        'predictions_csv': 'data/lstm_predictions.csv',
        'config': {
            'target_mode': args.target_mode,
            'sequence_length': args.sequence_length,
            'epochs': args.epochs,
            'patience': args.patience,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'hidden_size1': args.hidden_size1,
            'hidden_size2': args.hidden_size2,
            'dropout': args.dropout,
        }
    }


def parse_args():
    """Parse command-line arguments with validation."""
    parser = argparse.ArgumentParser(
        description='Standalone LSTM pipeline for USD-NGN forecasting'
    )
    
    parser.add_argument(
        '--target_mode', choices=['return', 'level', 'log_price'], default='return',
        help='return: train on log-returns, level: train on price levels, log_price: train on log(price)'
    )
    parser.add_argument(
        '--sequence_length', type=int, default=30,
        help='LSTM lookback window (5-100 recommended)'
    )
    parser.add_argument(
        '--epochs', type=int, default=75,
        help='maximum training epochs (10-200 typical)'
    )
    parser.add_argument(
        '--patience', type=int, default=12,
        help='epochs without improvement before stopping'
    )
    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='training batch size (16-128 typical)'
    )
    parser.add_argument(
        '--learning_rate', type=float, default=0.001,
        help='Adam optimizer learning rate (0.0001-0.01 typical)'
    )
    parser.add_argument(
        '--hidden_size1', type=int, default=96,
        help='first LSTM layer units (32-256 typical)'
    )
    parser.add_argument(
        '--hidden_size2', type=int, default=48,
        help='second LSTM layer units (16-128 typical)'
    )
    parser.add_argument(
        '--dropout', type=float, default=0.2,
        help='dropout rate (0.0-0.5 typical)'
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_model', action='store_true', help='save model weights')
    parser.add_argument(
        '--verbose', type=str, default='True', choices=['True', 'False'],
        help='verbose output'
    )
    
    args = parser.parse_args()
    args.verbose = args.verbose == 'True'
    
    # VALIDATION
    if args.sequence_length < 5 or args.sequence_length > 500:
        raise ValueError(f"sequence_length must be 5-500, got {args.sequence_length}")
    
    if args.epochs < 1 or args.epochs > 500:
        raise ValueError(f"epochs must be 1-500, got {args.epochs}")
    
    if args.patience >= args.epochs:
        raise ValueError(f"patience ({args.patience}) must be < epochs ({args.epochs})")
    
    if args.batch_size < 1 or args.batch_size > 512:
        raise ValueError(f"batch_size must be 1-512, got {args.batch_size}")
    
    if args.learning_rate < 1e-6 or args.learning_rate > 1:
        raise ValueError(f"learning_rate must be 1e-6-1, got {args.learning_rate}")
    
    if args.hidden_size1 < 8 or args.hidden_size1 > 512:
        raise ValueError(f"hidden_size1 must be 8-512, got {args.hidden_size1}")
    
    if args.hidden_size2 < 8 or args.hidden_size2 > 512:
        raise ValueError(f"hidden_size2 must be 8-512, got {args.hidden_size2}")
    
    if args.dropout < 0 or args.dropout > 0.7:
        raise ValueError(f"dropout must be 0-0.7, got {args.dropout}")
    
    return args


if __name__ == "__main__":
    try:
        args = parse_args()
        result = fit_lstm(args)
        
        if result and args.verbose:
            print("\n[SUCCESS] LSTM pipeline completed")
            print(f"Config: target_mode={result['config']['target_mode']}, "
                  f"seq_len={result['config']['sequence_length']}, "
                  f"epochs={result['config']['epochs']}")
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
