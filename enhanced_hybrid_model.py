"""
ENHANCED HYBRID MODEL: Optimized for Both Directional Accuracy (70%+) and Low RMSE
Combines multiple techniques to achieve both targets

Key Enhancements:
1. Weighted ensemble that adapts based on recent performance
2. Directional classifier with confidence weighting
3. Residual correction using machine learning
4. Adaptive ARIMA-LSTM weights
5. Direction-preserving predictions (maintain actual direction predictions)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import Ridge, LinearRegression
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


class ResidualCorrector:
    """
    Machine learning model that learns to correct predictions.
    Improves RMSE by reducing systematic errors.
    """
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, y_true, y_pred, X_features=None):
        """Learn to correct prediction errors."""
        residuals = y_true - y_pred  # What we got wrong
        
        # Use features if available
        if X_features is not None:
            X_for_correction = X_features
        else:
            # Create features from predictions
            X_for_correction = np.zeros((len(y_pred), 5))
            X_for_correction[:, 0] = y_pred
            X_for_correction[:, 1] = np.concatenate([[0], np.diff(y_pred)])
            X_for_correction[:, 2] = np.abs(np.concatenate([[0], np.diff(y_pred)]))
            
            # Recent trend
            for i in range(1, min(6, len(y_pred))):
                X_for_correction[i:, 3] = np.concatenate([np.zeros(i), np.diff(y_pred, n=1)[:-i+1]])
            
            # Momentum
            X_for_correction[:, 4] = np.concatenate([[0], np.diff(y_pred, 2)])
        
        # Handle NaN/Inf
        X_for_correction = np.nan_to_num(X_for_correction, nan=0, posinf=0, neginf=0)
        
        # Normalize
        X_scaled = self.scaler.fit_transform(X_for_correction)
        
        # Train to predict residuals
        self.model.fit(X_scaled, residuals)
        self.is_fitted = True
        return self
        
    def correct(self, y_pred, X_features=None):
        """Apply learned corrections to predictions."""
        if not self.is_fitted:
            return y_pred
            
        if X_features is not None:
            X_for_correction = X_features
        else:
            X_for_correction = np.zeros((len(y_pred), 5))
            X_for_correction[:, 0] = y_pred
            X_for_correction[:, 1] = np.concatenate([[0], np.diff(y_pred)])
            X_for_correction[:, 2] = np.abs(np.concatenate([[0], np.diff(y_pred)]))
            
            for i in range(1, min(6, len(y_pred))):
                X_for_correction[i:, 3] = np.concatenate([np.zeros(i), np.diff(y_pred, n=1)[:-i+1]])
            
            X_for_correction[:, 4] = np.concatenate([[0], np.diff(y_pred, 2)])
        
        X_for_correction = np.nan_to_num(X_for_correction, nan=0, posinf=0, neginf=0)
        X_scaled = self.scaler.transform(X_for_correction)
        
        correction = self.model.predict(X_scaled)
        return y_pred + 0.5 * correction  # Apply partial correction


class DirectionalBooster:
    """
    Ensures direction predictions are accurate and consistent.
    Lifts DA without hurting RMSE too much.
    """
    def __init__(self):
        self.direction_model = GradientBoostingClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X_features, y_true):
        """Learn to predict direction directly."""
        # Create direction labels
        direction = (np.diff(y_true) > 0).astype(int)
        X_aligned = X_features[:-1]  # Align with direction
        
        # Handle NaN/Inf
        X_aligned = np.nan_to_num(X_aligned, nan=0, posinf=0, neginf=0)
        
        # Scale
        X_scaled = self.scaler.fit_transform(X_aligned)
        
        # Train
        self.direction_model.fit(X_scaled, direction)
        self.is_fitted = True
        return self
        
    def get_direction_confidence(self, X_features):
        """Get direction predictions and confidence."""
        if not self.is_fitted or len(X_features) == 0:
            return None, None
            
        X_features = np.nan_to_num(X_features, nan=0, posinf=0, neginf=0)
        X_scaled = self.scaler.transform(X_features)
        
        proba = self.direction_model.predict_proba(X_scaled)
        direction_pred = self.direction_model.predict(X_scaled)
        
        # Confidence = distance from 0.5
        confidence = np.max(proba, axis=1)  # Highest probability
        
        return direction_pred, confidence


class EnhancedHybridModel:
    """
    Super-optimized hybrid model combining:
    - Adaptive ARIMA-LSTM weights
    - Residual correction
    - Direction boosting
    - Confidence filtering
    """
    
    def __init__(self, sequence_length=30):
        self.sequence_length = sequence_length
        
        # Core models
        self.arima = None
        self.lstm = None
        
        # Enhancement components
        self.residual_corrector = ResidualCorrector()
        self.direction_booster = DirectionalBooster()
        
        # State
        self.last_value = None
        self.train_mean = None
        self.adaptive_weight = 0.35  # Start with ARIMA weight
        
    def fit_arima(self, y_train):
        """Fit ARIMA with optimal order search."""
        if not STATSMODELS_AVAILABLE:
            return
            
        best_aic = np.inf
        best_order = (1, 1, 1)
        
        for p in range(4):
            for d in range(2):
                for q in range(4):
                    if p == 0 and q == 0:
                        continue
                    try:
                        model = ARIMA(y_train, order=(p, d, q))
                        fitted = model.fit(disp=False)
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                    except:
                        continue
        
        self.arima = ARIMA(y_train, order=best_order)
        self.arima_fitted = self.arima.fit(disp=False)
        self.arima_order = best_order
        print(f"  ✓ ARIMA{best_order} fitted (AIC={best_aic:.1f})")
        
    def fit_lstm(self, X_train, y_train, X_val=None, y_val=None):
        """Fit LSTM using a simple GradientBoosting fallback."""
        # Try to import LSTM, but don't fail if not available
        try:
            from part4_models import LSTMModel
            use_lstm = True
        except:
            use_lstm = False
        
        # Create sequences
        if len(X_train) < self.sequence_length:
            seq_len = max(10, len(X_train) // 2)
        else:
            seq_len = self.sequence_length
        
        if use_lstm:
            self.lstm = LSTMModel(
                input_size=X_train.shape[1],
                sequence_length=seq_len,
                epochs=100,
                patience=15
            )
            
            try:
                self.lstm.fit(X_train, y_train, X_val, y_val, verbose=False)
                print(f"  ✓ LSTM fitted (seq_len={seq_len})")
            except:
                # Fallback: simple gradient boosting
                self._fit_gb_fallback(X_train, y_train)
        else:
            # No LSTM available, use gradient boosting
            self._fit_gb_fallback(X_train, y_train)
    
    def _fit_gb_fallback(self, X_train, y_train):
        """Fallback to GradientBoosting when LSTM unavailable."""
        self.lstm = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42
        )
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        self.lstm.fit(X_scaled, y_train)
        self.lstm_scaler = scaler
        print(f"  ✓ LSTM (GB fallback) fitted")
            
    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        """Fit the enhanced model."""
        self.last_value = y_train[-1]
        self.train_mean = np.mean(y_train)
        
        if verbose:
            print("\n[Enhanced Hybrid] Training components...")
        
        # 1. ARIMA
        self.fit_arima(y_train)
        
        # 2. LSTM
        self.fit_lstm(X_train, y_train, X_val, y_val)
        
        # 3. Get predictions for training residuals correction
        if verbose:
            print("  [Training correction & direction models...]")
            
        arima_pred = self.arima_fitted.fittedvalues.values[-len(y_train):]
        
        # Pad to same length
        arima_pred = np.concatenate([
            np.full(len(y_train) - len(arima_pred), arima_pred[0] if len(arima_pred) > 0 else self.train_mean),
            arima_pred
        ])
        
        # Simple LSTM-like prediction (for now just use GradientBoosting fallback)
        if hasattr(self.lstm, 'predict'):
            try:
                lstm_pred = self.lstm.predict(X_train)
            except:
                lstm_pred = self.lstm.predict(self.lstm_scaler.transform(X_train))
        else:
            lstm_pred = np.full(len(y_train), self.train_mean)
        
        # Average to get base predictions
        min_len = min(len(arima_pred), len(lstm_pred), len(y_train))
        base_pred = (0.35 * arima_pred[-min_len:] + 0.65 * lstm_pred[-min_len:])
        y_aligned = y_train[-min_len:]
        
        # 4. Fit correction model
        self.residual_corrector.fit(y_aligned, base_pred, X_train[-min_len:])
        
        # 5. Fit direction booster
        self.direction_booster.fit(X_train, y_train)
        
        if verbose:
            print("  ✓ Enhanced Hybrid model ready!")
        
    def predict(self, X_test):
        """Make predictions with all enhancements."""
        n = len(X_test)
        
        # 1. ARIMA predictions
        try:
            arima_pred = self.arima_fitted.get_forecast(steps=n).predicted_mean.values
        except:
            arima_pred = np.full(n, self.last_value)
        
        # 2. LSTM predictions  
        try:
            if hasattr(self.lstm, 'predict'):
                lstm_pred = self.lstm.predict(X_test)
            else:
                lstm_pred = self.lstm.predict(self.lstm_scaler.transform(X_test))
        except:
            lstm_pred = np.full(n, self.train_mean)
        
        # Ensure same length
        min_len = min(len(arima_pred), len(lstm_pred), len(X_test))
        arima_pred = arima_pred[-min_len:] if len(arima_pred) > min_len else np.pad(arima_pred, (max(0, min_len-len(arima_pred)), 0), mode='edge')
        lstm_pred = lstm_pred[-min_len:] if len(lstm_pred) > min_len else np.pad(lstm_pred, (max(0, min_len-len(lstm_pred)), 0), mode='edge')
        X_test_aligned = X_test[-min_len:] if len(X_test) > min_len else np.pad(X_test, ((max(0, min_len-len(X_test)), 0), (0, 0)), mode='edge')
        
        # 3. Combine with adaptive weights
        hybrid_pred = self.adaptive_weight * arima_pred + (1 - self.adaptive_weight) * lstm_pred
        
        # 4. Apply residual correction
        corrected_pred = self.residual_corrector.correct(hybrid_pred, X_test_aligned)
        
        # 5. Direction enforcement (optional)
        # Get direction predictions
        direction_pred, confidence = self.direction_booster.get_direction_confidence(X_test_aligned)
        
        # Adjust predictions to match predicted direction (for high-confidence cases)
        if direction_pred is not None:
            adjusted_pred = corrected_pred.copy()
            prev_val = self.last_value
            
            for i in range(len(adjusted_pred)):
                pred_increase = adjusted_pred[i] > prev_val
                should_increase = (direction_pred[i] == 1) if i < len(direction_pred) else None
                
                if should_increase is not None and confidence[i] > 0.65:
                    conf_factor = (confidence[i] - 0.5) / 0.5  # Scale 0.5-1.0 to 0-1
                    
                    if should_increase and not pred_increase:
                        # Prediction should increase but doesn't
                        adjustment = (prev_val - adjusted_pred[i]) * conf_factor * 0.3
                        adjusted_pred[i] += adjustment
                    elif not should_increase and pred_increase:
                        # Prediction should decrease but doesn't
                        adjustment = (adjusted_pred[i] - prev_val) * conf_factor * 0.3
                        adjusted_pred[i] -= adjustment
                
                prev_val = adjusted_pred[i]
            
            return adjusted_pred
        else:
            return corrected_pred
    
    def evaluate_on_validation(self, y_true, y_pred):
        """Compute metrics."""
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        
        actual_dir = (np.diff(y_true) > 0).astype(int)
        pred_dir = (np.diff(y_pred) > 0).astype(int)
        da = np.mean(actual_dir == pred_dir)
        
        return {'rmse': rmse, 'mae': mae, 'da': da}


if __name__ == "__main__":
    print("\nEnhanced Hybrid Model Test")
    print("=" * 60)
    
    # Test data
    np.random.seed(42)
    n = 800
    y = 600 + np.cumsum(np.random.randn(n) * 2)
    X = np.column_stack([
        70 + np.cumsum(np.random.randn(n) * 0.2),  # oil
        15 + np.cumsum(np.random.randn(n) * 0.01),  # mpr
        200 + np.cumsum(np.random.randn(n) * 0.5),  # cpi
    ])
    
    # Split
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    # Random Walk baseline
    rw_pred = np.concatenate([[y_train[-1]], y_test[:-1]])
    rw_rmse = np.sqrt(np.mean((y_test - rw_pred) ** 2))
    rw_da = np.mean((np.diff(y_test) > 0) == (np.diff(rw_pred) > 0))
    
    print(f"\nRandom Walk Baseline:")
    print(f"  RMSE: {rw_rmse:.2f}")
    print(f"  DA: {rw_da:.1%}")
    
    # Enhanced Hybrid
    model = EnhancedHybridModel()
    model.fit(X_train, y_train, X_val, y_val)
    
    pred = model.predict(X_test)
    metrics = model.evaluate_on_validation(y_test, pred)
    
    print(f"\nEnhanced Hybrid Model:")
    print(f"  RMSE: {metrics['rmse']:.2f} (vs RW: {rw_rmse:.2f}) - {'✓ BETTER' if metrics['rmse'] < rw_rmse else '✗ Worse'}")
    print(f"  DA: {metrics['da']:.1%} (target: 70%+) - {'✓ REACHED' if metrics['da'] >= 0.70 else '✗ Below'}")
    
    improvement = ((rw_rmse - metrics['rmse']) / rw_rmse) * 100
    da_improvement = (metrics['da'] - rw_da) * 100
    
    print(f"\nImprovement:")
    print(f"  RMSE: {improvement:.1f}% better than RW")
    print(f"  DA: +{da_improvement:.1f}pp vs RW")
