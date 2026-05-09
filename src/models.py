"""
PART 4: MODEL DEVELOPMENT (ARIMA, LSTM, HYBRID) - ENHANCED FOR DIRECTIONAL ACCURACY
PhD Thesis: USD-NGN Exchange Rate Forecasting
Author: Oche Emmanuel Ike (242220011)

ENHANCEMENTS:
- Direct direction classification (not just regression)
- Direction-focused feature engineering
- Ensemble methods for robust predictions
- Optimized decision thresholds
- High-confidence prediction filtering

Target: Improve DA from 55% to 70%+
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.ensemble import GradientBoostingRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ============================================================================
# ENHANCEMENT 1: DIRECTION-FOCUSED FEATURE ENGINEERING
# ============================================================================

class DirectionalFeatureEngineer:
    """
    Creates features specifically designed to predict DIRECTION, not just levels.
    
    Key insight: Standard features predict price levels, but direction prediction
    requires different features - momentum, streaks, and technical indicators.
    """
    
    @staticmethod
    def create_direction_features(y, X=None):
        """
        Create direction-predictive features from price series.
        
        Parameters:
        -----------
        y : np.ndarray
            Price series (e.g., USD-NGN rates)
        X : np.ndarray, optional
            Additional features to augment
            
        Returns:
        --------
        np.ndarray : Enhanced feature matrix with direction-focused features
        """
        n = len(y)
        features = []
        
        # 1. DIRECTION LAG FEATURES (Most predictive!)
        # Yesterday's direction often predicts today's (momentum effect)
        returns = np.zeros(n)
        returns[1:] = np.diff(y) / (y[:-1] + 1e-10)
        direction = (returns > 0).astype(float)
        
        for lag in [1, 2, 3, 5]:
            lagged_dir = np.zeros(n)
            lagged_dir[lag:] = direction[:-lag]
            features.append(lagged_dir)
        
        # 2. UP RATIO (fraction of up days in recent window)
        for window in [5, 10, 20]:
            up_ratio = np.zeros(n)
            for i in range(window, n):
                up_ratio[i] = np.mean(direction[i-window:i])
            features.append(up_ratio)
        
        # 3. STREAK LENGTH (consecutive same-direction days)
        streak = np.zeros(n)
        for i in range(1, n):
            if direction[i] == direction[i-1]:
                streak[i] = streak[i-1] + 1
            else:
                streak[i] = 1
        features.append(streak)
        
        # 4. MOMENTUM FEATURES
        for window in [5, 10, 20]:
            momentum = np.zeros(n)
            momentum[window:] = (y[window:] - y[:-window]) / (y[:-window] + 1e-10)
            features.append(momentum)
            
            # Momentum z-score (normalized)
            mom_mean = np.zeros(n)
            mom_std = np.zeros(n)
            for i in range(window + 60, n):
                mom_mean[i] = np.mean(momentum[i-60:i])
                mom_std[i] = np.std(momentum[i-60:i]) + 1e-8
            momentum_zscore = np.zeros(n)
            mask = mom_std > 1e-8
            momentum_zscore[mask] = (momentum[mask] - mom_mean[mask]) / mom_std[mask]
            features.append(momentum_zscore)
        
        # 5. VOLATILITY FEATURES
        log_returns = np.zeros(n)
        log_returns[1:] = np.log(y[1:] / (y[:-1] + 1e-10) + 1e-10)
        
        for window in [5, 10, 20]:
            volatility = np.zeros(n)
            for i in range(window, n):
                volatility[i] = np.std(log_returns[i-window:i])
            features.append(volatility)
            
            # Volatility ratio (short-term vs long-term)
            vol_long = np.zeros(n)
            for i in range(window * 3, n):
                vol_long[i] = np.std(log_returns[i-window*3:i])
            vol_ratio = np.zeros(n)
            mask = vol_long > 1e-8
            vol_ratio[mask] = volatility[mask] / vol_long[mask]
            features.append(vol_ratio)
        
        # 6. RSI (Relative Strength Index)
        rsi = np.zeros(n)
        period = 14
        for i in range(period + 1, n):
            gains = np.maximum(np.diff(y[i-period-1:i]), 0)
            losses = np.maximum(-np.diff(y[i-period-1:i]), 0)
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses) + 1e-8
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))
        features.append(rsi)
        
        # RSI signals
        rsi_oversold = (rsi < 30).astype(float)
        rsi_overbought = (rsi > 70).astype(float)
        features.append(rsi_oversold)
        features.append(rsi_overbought)
        
        # 7. MOVING AVERAGE POSITION
        for window in [5, 20, 50]:
            ma = np.zeros(n)
            for i in range(window, n):
                ma[i] = np.mean(y[i-window:i])
            above_ma = (y > ma).astype(float)
            distance_ma = np.zeros(n)
            mask = ma > 1e-8
            distance_ma[mask] = (y[mask] - ma[mask]) / ma[mask]
            features.append(above_ma)
            features.append(distance_ma)
        
        # Stack all features
        direction_features = np.column_stack(features)
        
        # Combine with original features if provided
        if X is not None:
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            # Ensure same length
            min_len = min(len(direction_features), len(X))
            direction_features = np.column_stack([X[:min_len], direction_features[:min_len]])
        
        return direction_features
    
    @staticmethod
    def create_cross_asset_direction_features(y, oil=None, mpr=None, cpi=None):
        """
        Create direction features from cross-asset relationships.
        """
        n = len(y)
        features = []
        
        if oil is not None and len(oil) == n:
            # Oil momentum
            for window in [5, 10, 20]:
                oil_mom = np.zeros(n)
                oil_mom[window:] = (oil[window:] - oil[:-window]) / (oil[:-window] + 1e-10)
                features.append(oil_mom)
            
            # Oil-FX correlation (rolling)
            fx_ret = np.zeros(n)
            fx_ret[1:] = np.diff(y) / (y[:-1] + 1e-10)
            oil_ret = np.zeros(n)
            oil_ret[1:] = np.diff(oil) / (oil[:-1] + 1e-10)
            
            oil_fx_corr = np.zeros(n)
            for i in range(20, n):
                if np.std(fx_ret[i-20:i]) > 1e-8 and np.std(oil_ret[i-20:i]) > 1e-8:
                    corr = np.corrcoef(fx_ret[i-20:i], oil_ret[i-20:i])[0, 1]
                    if not np.isnan(corr):
                        oil_fx_corr[i] = corr
            features.append(oil_fx_corr)
            
            # Oil direction lags
            oil_dir = np.zeros(n)
            oil_dir[1:] = (np.diff(oil) > 0).astype(float)
            for lag in [1, 2, 3]:
                lagged = np.zeros(n)
                lagged[lag:] = oil_dir[:-lag]
                features.append(lagged)
        
        if mpr is not None and len(mpr) == n:
            mpr_up = np.zeros(n)
            mpr_up[1:] = (np.diff(mpr) > 0).astype(float)
            features.append(mpr_up)
            
            mpr_high = np.zeros(n)
            for i in range(60, n):
                mpr_high[i] = (mpr[i] > np.mean(mpr[i-60:i])).astype(float)
            features.append(mpr_high)
        
        if cpi is not None and len(cpi) == n:
            cpi_mom = np.zeros(n)
            cpi_mom[20:] = cpi[20:] - cpi[:-20]
            features.append(cpi_mom)
            
            cpi_diff = np.zeros(n)
            cpi_diff[1:] = np.diff(cpi)
            cpi_accel = np.zeros(n)
            cpi_accel[2:] = (cpi_diff[2:] > cpi_diff[1:-1]).astype(float)
            features.append(cpi_accel)
        
        if features:
            return np.column_stack(features)
        return None


# ============================================================================
# ENHANCEMENT 2: DIRECTIONAL CLASSIFIER ENSEMBLE
# ============================================================================

class DirectionalEnsemble:
    """
    Ensemble of classifiers that directly predict direction (UP/DOWN).
    
    Key insight: Predicting direction as a classification problem is more
    effective than predicting levels and deriving direction.
    """
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.scaler = StandardScaler()
        self.threshold = 0.5
        self.is_fitted = False
        
    def fit(self, X, y, X_val=None, y_val=None, verbose=True):
        """
        Fit ensemble on direction labels.
        """
        if verbose:
            print("\n  [DirectionalEnsemble] Training ensemble classifiers...")
        
        # Create direction target (1=UP, 0=DOWN)
        direction = (np.diff(y) > 0).astype(int)
        X_aligned = X[:-1]
        
        # Handle validation set
        if X_val is not None and y_val is not None and len(y_val) > 1:
            direction_val = (np.diff(y_val) > 0).astype(int)
            X_val_aligned = X_val[:-1]
        else:
            split = int(0.8 * len(X_aligned))
            X_val_aligned = X_aligned[split:]
            direction_val = direction[split:]
            X_aligned = X_aligned[:split]
            direction = direction[:split]
        
        # Handle NaN/Inf
        X_aligned = np.nan_to_num(X_aligned, nan=0.0, posinf=0.0, neginf=0.0)
        X_val_aligned = np.nan_to_num(X_val_aligned, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        self.scaler.fit(X_aligned)
        X_scaled = self.scaler.transform(X_aligned)
        X_val_scaled = self.scaler.transform(X_val_aligned)
        
        # Model 1: Gradient Boosting
        if verbose:
            print("    Training Gradient Boosting Classifier...")
        gb = GradientBoostingClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.8,
            min_samples_leaf=20,
            random_state=42
        )
        gb.fit(X_scaled, direction)
        gb_acc = np.mean(gb.predict(X_val_scaled) == direction_val)
        self.models['gb'] = gb
        self.weights['gb'] = gb_acc ** 2
        if verbose:
            print(f"      Validation DA: {gb_acc:.1%}")
        
        # Model 2: Random Forest
        if verbose:
            print("    Training Random Forest Classifier...")
        rf = RandomForestClassifier(
            n_estimators=120,
            max_depth=6,
            min_samples_leaf=15,
            class_weight='balanced',
            random_state=42,
            n_jobs=1
        )
        rf.fit(X_scaled, direction)
        rf_acc = np.mean(rf.predict(X_val_scaled) == direction_val)
        self.models['rf'] = rf
        self.weights['rf'] = rf_acc ** 2
        if verbose:
            print(f"      Validation DA: {rf_acc:.1%}")
        
        # Model 3: Logistic Regression
        if verbose:
            print("    Training Logistic Regression...")
        lr = LogisticRegression(
            C=0.1,
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
        lr.fit(X_scaled, direction)
        lr_acc = np.mean(lr.predict(X_val_scaled) == direction_val)
        self.models['lr'] = lr
        self.weights['lr'] = lr_acc ** 2
        if verbose:
            print(f"      Validation DA: {lr_acc:.1%}")
        
        # Normalize weights
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}
        else:
            self.weights = {k: 1/3 for k in self.weights}
        
        if verbose:
            print(f"    Ensemble weights: GB={self.weights['gb']:.2f}, RF={self.weights['rf']:.2f}, LR={self.weights['lr']:.2f}")
        
        # Optimize threshold
        self._optimize_threshold(X_val_scaled, direction_val, verbose)
        
        self.is_fitted = True
        return self
    
    def _optimize_threshold(self, X_val, y_val, verbose=True):
        """Find optimal probability threshold."""
        probs = self._predict_proba(X_val)
        
        best_threshold = 0.5
        best_acc = 0
        
        for t in np.arange(0.35, 0.65, 0.01):
            preds = (probs >= t).astype(int)
            acc = np.mean(preds == y_val)
            if acc > best_acc:
                best_acc = acc
                best_threshold = t
        
        self.threshold = best_threshold
        if verbose:
            print(f"    Optimal threshold: {best_threshold:.2f} (DA: {best_acc:.1%})")
    
    def _predict_proba(self, X_scaled):
        """Get weighted ensemble probability of UP direction."""
        prob = np.zeros(len(X_scaled))
        for name, model in self.models.items():
            prob += self.weights[name] * model.predict_proba(X_scaled)[:, 1]
        return prob
    
    def predict_direction(self, X):
        """
        Predict direction (1=UP, 0=DOWN).
        """
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X)
        
        probs = self._predict_proba(X_scaled)
        directions = (probs >= self.threshold).astype(int)
        confidence = np.abs(probs - 0.5) * 2
        
        return directions, probs, confidence


# ============================================================================
# ORIGINAL MODELS (RETAINED WITH ENHANCEMENTS)
# ============================================================================

class ARIMAModel:
    """ARIMA model with rolling prediction capability."""
    
    def __init__(self, max_p=5, max_d=2, max_q=5, search_max_points=2500, refit_every=60):
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.search_max_points = search_max_points
        self.refit_every = refit_every
        self.model = None
        self.fitted = None
        self.best_order = None
        self.series = None
        self._rolling_cache = {}
        
    def fit(self, series, order=None, verbose=True):
        series = np.array(series).flatten()
        series = series[~np.isnan(series)]
        self.series = series
        
        if order is None:
            self.best_order = self._grid_search(series, verbose)
        else:
            self.best_order = order
            
        if not STATSMODELS_AVAILABLE:
            self._fit_ar_fallback(series)
            return self
            
        try:
            self.model = ARIMA(series, order=self.best_order)
            self.fitted = self.model.fit()
            if verbose:
                print(f"  ARIMA{self.best_order} fitted, AIC={self.fitted.aic:.2f}")
        except (ValueError, np.linalg.LinAlgError, RuntimeError) as e:
            if verbose:
                print(f"  ARIMA fitting failed ({e}), using AR fallback")
            self._fit_ar_fallback(series)
        return self
    
    def _grid_search(self, series, verbose):
        if not STATSMODELS_AVAILABLE:
            return (1, 1, 1)
        if self.search_max_points is not None and len(series) > self.search_max_points:
            series = series[-self.search_max_points:]
        best_aic, best_order = np.inf, (1, 1, 1)
        for p in range(min(3, self.max_p + 1)):
            for d in range(min(2, self.max_d + 1)):
                for q in range(min(3, self.max_q + 1)):
                    if p == 0 and q == 0:
                        continue
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                    except (ValueError, np.linalg.LinAlgError, RuntimeError):
                        continue
        if verbose:
            print(f"  Best order: ARIMA{best_order}")
        return best_order
    
    def _fit_ar_fallback(self, series, p=1):
        self.ar_mean = np.mean(series)
        self.ar_coeffs = [0.9] * p
        self.fitted = None
        
    def predict(self, steps=1):
        if self.fitted:
            return self.fitted.forecast(steps=steps)
        predictions = []
        history = list(self.series[-len(self.ar_coeffs):])
        for _ in range(steps):
            pred = self.ar_mean
            for i, coef in enumerate(self.ar_coeffs):
                if i < len(history):
                    pred += coef * (history[-(i+1)] - self.ar_mean)
            predictions.append(pred)
            history.append(pred)
        return np.array(predictions)
    
    def predict_rolling(self, y_test, verbose=True):
        """
        ENHANCED: Rolling one-step-ahead predictions.
        """
        y_test = np.array(y_test).flatten()
        if not STATSMODELS_AVAILABLE:
            return np.full(len(y_test), self.series[-1])

        cache_key = (
            len(y_test),
            round(float(np.sum(y_test)), 6) if len(y_test) > 0 else 0.0,
            round(float(y_test[0]), 6) if len(y_test) > 0 else 0.0,
            round(float(y_test[-1]), 6) if len(y_test) > 0 else 0.0,
        )
        if cache_key in self._rolling_cache:
            return self._rolling_cache[cache_key].copy()

        history = list(self.series)
        predictions = np.zeros(len(y_test))
        n_test = len(y_test)
        print_interval = max(1, n_test // 5)
        fitted = self.fitted

        if fitted is None:
            preds = np.full(n_test, self.series[-1] if self.series is not None else np.nan)
            self._rolling_cache[cache_key] = preds.copy()
            return preds

        for t in range(n_test):
            try:
                predictions[t] = fitted.forecast(steps=1)[0]
            except (ValueError, np.linalg.LinAlgError, RuntimeError, AttributeError):
                predictions[t] = history[-1]

            history.append(y_test[t])

            try:
                if self.refit_every and (t + 1) % self.refit_every == 0:
                    refit_history = history[-self.search_max_points:] if self.search_max_points else history
                    fitted = ARIMA(refit_history, order=self.best_order).fit()
                else:
                    fitted = fitted.append([y_test[t]], refit=False)
            except (ValueError, np.linalg.LinAlgError, RuntimeError, AttributeError, TypeError):
                try:
                    refit_history = history[-self.search_max_points:] if self.search_max_points else history
                    fitted = ARIMA(refit_history, order=self.best_order).fit()
                except (ValueError, np.linalg.LinAlgError, RuntimeError):
                    pass

            if verbose and t % print_interval == 0:
                print(f"    Rolling ARIMA: {t+1}/{n_test}")

        self._rolling_cache[cache_key] = predictions.copy()
        return predictions
    
    def get_residuals(self):
        if self.fitted:
            return np.array(self.fitted.resid)
        return self.series[1:] - self.series[:-1]


if TORCH_AVAILABLE:
    class PyTorchLSTM(nn.Module):
        """Deep Learning LSTM model for time series forecasting."""
        
        def __init__(self, input_size, hidden_size1=64, hidden_size2=32, output_size=1, dropout=0.2):
            super(PyTorchLSTM, self).__init__()
            self.hidden_size1 = hidden_size1
            self.hidden_size2 = hidden_size2
            
            self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
            self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size2, 32)
            self.relu = nn.ReLU()
            self.output = nn.Linear(32, output_size)

        def forward(self, x):
            lstm1_out, _ = self.lstm1(x)
            lstm2_out, (h_n, c_n) = self.lstm2(lstm1_out)
            last_output = lstm2_out[:, -1, :]
            x = self.dropout(last_output)
            x = self.fc(x)
            x = self.relu(x)
            x = self.output(x)
            return x
else:
    PyTorchLSTM = None


if TORCH_AVAILABLE:
    class PyTorchGRU(nn.Module):
        """Deep Learning GRU model for time series forecasting."""

        def __init__(self, input_size, hidden_size1=64, hidden_size2=32, output_size=1, dropout=0.2):
            super(PyTorchGRU, self).__init__()
            self.gru1 = nn.GRU(input_size, hidden_size1, batch_first=True)
            self.gru2 = nn.GRU(hidden_size1, hidden_size2, batch_first=True)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size2, 32)
            self.relu = nn.ReLU()
            self.output = nn.Linear(32, output_size)

        def forward(self, x):
            gru1_out, _ = self.gru1(x)
            gru2_out, _ = self.gru2(gru1_out)
            last_output = gru2_out[:, -1, :]
            x = self.dropout(last_output)
            x = self.fc(x)
            x = self.relu(x)
            x = self.output(x)
            return x
else:
    PyTorchGRU = None


class LSTMModel:
    """LSTM-based forecasting model."""
    
    def __init__(self, input_size, sequence_length=60, batch_size=32, epochs=40,
                 patience=8, learning_rate=0.001):
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.model = None
        self.best_model_state = None
        self.sklearn_model = None
        self.use_pytorch = TORCH_AVAILABLE

        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = None

    def _build_torch_model(self, input_size):
        return PyTorchLSTM(input_size=input_size).to(self.device)

    def create_sequences(self, X, y):
        X_seq, y_seq = [], []
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])
        return np.array(X_seq), np.array(y_seq)

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        self.scaler_X.fit(X_train)
        self.scaler_y.fit(y_train.reshape(-1, 1))

        if not TORCH_AVAILABLE:
            return self._fit_sklearn_fallback(X_train, y_train, verbose)

        X_scaled = self.scaler_X.transform(X_train)
        y_scaled = self.scaler_y.transform(y_train.reshape(-1, 1)).flatten()

        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled)

        if len(X_seq) == 0:
            if verbose:
                print("  Insufficient data for sequences, using sklearn fallback")
            return self._fit_sklearn_fallback(X_train, y_train, verbose)

        actual_batch_size = min(self.batch_size, len(X_seq))
        if actual_batch_size < 1:
            actual_batch_size = 1

        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).reshape(-1, 1).to(self.device)

        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=actual_batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None and len(X_val) >= self.sequence_length:
            X_val_scaled = self.scaler_X.transform(X_val)
            y_val_scaled = self.scaler_y.transform(y_val.reshape(-1, 1)).flatten()
            X_val_seq, y_val_seq = self.create_sequences(X_val_scaled, y_val_scaled)
            if len(X_val_seq) > 0:
                X_val_tensor = torch.FloatTensor(X_val_seq).to(self.device)
                y_val_tensor = torch.FloatTensor(y_val_seq).reshape(-1, 1).to(self.device)
                val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
                val_loader = DataLoader(val_dataset, batch_size=actual_batch_size, shuffle=False)

        if PyTorchLSTM is None:
            return self._fit_sklearn_fallback(X_train, y_train, verbose)
        self.model = self._build_torch_model(X_seq.shape[2])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            if val_loader is not None:
                val_loss = self._validate(val_loader, criterion)
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch+1}")
                    break
            else:
                if train_loss < best_val_loss:
                    best_val_loss = train_loss
                    self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        if self.best_model_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in self.best_model_state.items()})

        if verbose:
            print(f"  LSTM trained for {epoch+1} epochs")

        return {}

    def _validate(self, val_loader, criterion):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()
        return val_loss / len(val_loader)

    def _fit_sklearn_fallback(self, X_train, y_train, verbose=True):
        if verbose:
            print("  Using GradientBoosting fallback")
        X_scaled = self.scaler_X.transform(X_train)
        y_scaled = self.scaler_y.transform(y_train.reshape(-1, 1)).flatten()
        self.sklearn_model = GradientBoostingRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        )
        self.sklearn_model.fit(X_scaled, y_scaled)
        self.use_pytorch = False
        return {}

    def predict(self, X):
        if self.sklearn_model is not None:
            X_scaled = self.scaler_X.transform(X)
            pred_scaled = self.sklearn_model.predict(X_scaled)
            return self.scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

        if self.model is None:
            return np.full(len(X), np.nan)

        if len(X) < self.sequence_length:
            return np.full(len(X), np.nan)

        X_scaled = self.scaler_X.transform(X)
        X_seq, _ = self.create_sequences(X_scaled, np.zeros(len(X_scaled)))

        if len(X_seq) == 0:
            return np.full(len(X), np.nan)

        X_tensor = torch.FloatTensor(X_seq).to(self.device)

        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_tensor)

        y_pred = y_pred.cpu().numpy().flatten()
        y_pred_original = self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

        padding = len(X) - len(y_pred_original)
        if padding > 0:
            pad_value = y_pred_original[0] if len(y_pred_original) > 0 else 0
            y_pred_original = np.concatenate([np.full(padding, pad_value), y_pred_original])

        return y_pred_original[:len(X)]


class GRUModel(LSTMModel):
    """GRU-based forecasting model with the same interface as LSTMModel."""

    def _build_torch_model(self, input_size):
        if PyTorchGRU is None:
            return None
        return PyTorchGRU(input_size=input_size).to(self.device)

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        self.scaler_X.fit(X_train)
        self.scaler_y.fit(y_train.reshape(-1, 1))

        if not TORCH_AVAILABLE or PyTorchGRU is None:
            return self._fit_sklearn_fallback(X_train, y_train, verbose)

        return super().fit(X_train, y_train, X_val=X_val, y_val=y_val, verbose=verbose)


class ARIMAXModel:
    """ARIMAX/SARIMAX model with exogenous regressors and rolling prediction."""

    def __init__(self, max_p=3, max_d=2, max_q=3, search_max_points=2500, refit_every=60):
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.search_max_points = search_max_points
        self.refit_every = refit_every
        self.model = None
        self.fitted = None
        self.best_order = None
        self.series = None
        self.exog = None
        self.train_mean = None

    def fit(self, y_train, X_train, order=None, verbose=True):
        y_train = np.array(y_train).flatten()
        X_train = np.array(X_train)
        self.series = y_train
        self.exog = X_train
        self.train_mean = float(np.mean(y_train)) if len(y_train) > 0 else 0.0

        if order is None:
            self.best_order = self._grid_search(y_train, X_train, verbose)
        else:
            self.best_order = order

        if not STATSMODELS_AVAILABLE:
            return self

        try:
            self.model = SARIMAX(
                y_train,
                exog=X_train,
                order=self.best_order,
                trend='c',
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.fitted = self.model.fit(disp=False)
            if verbose:
                print(f"  ARIMAX{self.best_order} fitted, AIC={self.fitted.aic:.2f}")
        except (ValueError, np.linalg.LinAlgError, RuntimeError) as e:
            if verbose:
                print(f"  ARIMAX fitting failed ({e}), falling back to Random Walk-style behavior")
            self.fitted = None
        return self

    def _grid_search(self, y_train, X_train, verbose):
        if not STATSMODELS_AVAILABLE:
            return (1, 1, 1)
        if self.search_max_points is not None and len(y_train) > self.search_max_points:
            y_train = y_train[-self.search_max_points:]
            X_train = X_train[-self.search_max_points:]

        best_aic, best_order = np.inf, (1, 1, 1)
        for p in range(self.max_p + 1):
            for d in range(min(2, self.max_d + 1)):
                for q in range(self.max_q + 1):
                    if p == 0 and q == 0:
                        continue
                    try:
                        model = SARIMAX(
                            y_train,
                            exog=X_train,
                            order=(p, d, q),
                            trend='c',
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        )
                        fitted = model.fit(disp=False)
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                    except (ValueError, np.linalg.LinAlgError, RuntimeError):
                        continue
        if verbose:
            print(f"  Best order: ARIMAX{best_order}")
        return best_order

    def predict_rolling(self, X_test, y_test_actual, verbose=True):
        X_test = np.array(X_test)
        y_test_actual = np.array(y_test_actual).flatten()
        if self.fitted is None or not STATSMODELS_AVAILABLE:
            history = list(self.series)
            preds = np.zeros(len(y_test_actual))
            for i in range(len(y_test_actual)):
                preds[i] = history[-1] if history else self.train_mean
                history.append(y_test_actual[i])
            return preds

        fitted = self.fitted
        history_y = list(self.series)
        history_X = np.array(self.exog)
        preds = np.zeros(len(y_test_actual))
        print_interval = max(1, len(y_test_actual) // 5)

        for t in range(len(y_test_actual)):
            exog_step = X_test[t:t+1]
            try:
                preds[t] = fitted.forecast(steps=1, exog=exog_step)[0]
            except (ValueError, np.linalg.LinAlgError, RuntimeError, AttributeError):
                preds[t] = history_y[-1] if history_y else self.train_mean

            history_y.append(y_test_actual[t])
            history_X = np.vstack([history_X, exog_step])

            try:
                if self.refit_every and (t + 1) % self.refit_every == 0:
                    refit_y = np.array(history_y[-self.search_max_points:]) if self.search_max_points else np.array(history_y)
                    refit_X = history_X[-len(refit_y):]
                    fitted = SARIMAX(
                        refit_y,
                        exog=refit_X,
                        order=self.best_order,
                        trend='c',
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit(disp=False)
                else:
                    fitted = fitted.append([y_test_actual[t]], exog=exog_step, refit=False)
            except (ValueError, np.linalg.LinAlgError, RuntimeError, AttributeError, TypeError):
                try:
                    refit_y = np.array(history_y[-self.search_max_points:]) if self.search_max_points else np.array(history_y)
                    refit_X = history_X[-len(refit_y):]
                    fitted = SARIMAX(
                        refit_y,
                        exog=refit_X,
                        order=self.best_order,
                        trend='c',
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit(disp=False)
                except (ValueError, np.linalg.LinAlgError, RuntimeError):
                    pass

            if verbose and t % print_interval == 0:
                print(f"    Rolling ARIMAX: {t+1}/{len(y_test_actual)}")

        return preds


# ============================================================================
# ENHANCED HYBRID MODEL WITH DIRECTIONAL ACCURACY FOCUS
# ============================================================================

class HybridARIMALSTM:
    """
    ENHANCED Hybrid ARIMA-LSTM model with Directional Accuracy focus.
    
    Key Enhancements:
    1. Direct direction classification (ensemble) alongside level prediction
    2. Direction-focused feature engineering
    3. Optimized combination weights
    4. Confidence-based predictions
    """
    
    def __init__(self, arima_order=None, feature_weights=None, sequence_length=60,
                 arima_weight=0.25, use_directional_ensemble=True):
        self.arima_order = arima_order
        self.feature_weights = feature_weights
        self.sequence_length = sequence_length
        self.arima_weight = arima_weight  # Reduced from 0.3 to 0.25 - LSTM more weight for better DA
        self.use_directional_ensemble = use_directional_ensemble
        self.shock_threshold = 2.5
        self.regime_profiles = {
            'calm': 0.55,
            'trend': 0.80,
            'volatile': 0.65,
            'shock': 0.30,
        }
        self.mr_speed = 0.05
        self.contrarian_magnitude = 0.75
        self.streak_boost = 1.25
        self.shock_scale = 1.0
        self.component_weights = {
            'base': 0.55,
            'arima': 0.15,
            'mr': 0.15,
            'streak': 0.15,
        }

        self.arima_model = None
        self.lstm_model = None
        self.directional_ensemble = None
        self.feature_engineer = DirectionalFeatureEngineer()
        
        self.target_series = None
        self.train_mean = None
        self.history_X = None
        self.history_resid = None
        self.last_train_y = None
        self.direction_X_train = None
        self.train_returns = None
        self.train_vol_reference = 0.0
        self.regime_history = []
        self.validation_tuned = False

    def apply_feature_weights(self, X, weights):
        if weights is None:
            return X
        weights = np.array(weights).flatten()
        if len(weights) != X.shape[1]:
            weights = np.ones(X.shape[1])
        return X * weights

    @staticmethod
    def _pct_returns(y):
        y = np.array(y, dtype=float).flatten()
        if len(y) < 2:
            return np.array([])
        return np.diff(y) / (np.abs(y[:-1]) + 1e-8)

    def _detect_regime(self, history):
        history = np.array(history, dtype=float).flatten()
        returns = self._pct_returns(history)
        if len(returns) < 5:
            return 'calm'

        recent_window = returns[-min(20, len(returns)):]
        short_vol = np.std(recent_window)
        long_vol = np.std(returns[-min(120, len(returns)):]) if len(returns) >= 20 else short_vol
        long_vol = max(long_vol, self.train_vol_reference, 1e-6)

        momentum = np.mean(recent_window[-min(5, len(recent_window)):])
        shock_score = abs(recent_window[-1]) / long_vol

        if shock_score >= self.shock_threshold:
            return 'shock'
        if short_vol > 1.75 * long_vol:
            return 'volatile'
        if abs(momentum) > 0.75 * long_vol:
            return 'trend'
        return 'calm'

    def _shock_adjustment(self, history, regime):
        history = np.array(history, dtype=float).flatten()
        if len(history) < 3 or regime != 'shock':
            return 0.0

        returns = self._pct_returns(history)
        if len(returns) == 0:
            return 0.0

        recent_return = returns[-1]
        vol = max(np.std(returns[-min(60, len(returns)):]), self.train_vol_reference, 1e-6)
        intensity = min(1.5, abs(recent_return) / vol)
        return 0.15 * intensity * recent_return * history[-1]

    def _compute_heuristic_terms(self, history):
        history = np.array(history, dtype=float).flatten()
        current = history[-1] if len(history) > 0 else self.train_mean

        if len(history) >= 20:
            ma = np.mean(history[-20:])
        else:
            ma = np.mean(history) if len(history) > 0 else current
        reversion_term = ma - current

        if len(history) >= 2:
            last_change = history[-1] - history[-2]
            contrarian_term = -np.sign(last_change)
        else:
            contrarian_term = 0.0

        streak_term = 0.0
        if len(history) >= 4:
            recent_changes = np.diff(history[-4:])
            if np.all(recent_changes > 0):
                streak_term = -1.0
            elif np.all(recent_changes < 0):
                streak_term = 1.0

        return current, reversion_term, contrarian_term, streak_term

    def _assemble_prediction(self, arima_step, lstm_step, history, params=None):
        params = params or {}
        mr_speed = params.get('mr_speed', self.mr_speed)
        contrarian_magnitude = params.get('contrarian_magnitude', self.contrarian_magnitude)
        streak_boost = params.get('streak_boost', self.streak_boost)
        shock_scale = params.get('shock_scale', self.shock_scale)
        component_weights = params.get('component_weights', self.component_weights)

        regime = self._detect_regime(history)
        residual_weight = self.regime_profiles.get(regime, 1 - self.arima_weight)
        current, reversion_term, contrarian_term, streak_term = self._compute_heuristic_terms(history)

        base_pred = arima_step + residual_weight * lstm_step
        mr_pred = (
            current
            + mr_speed * reversion_term
            + contrarian_magnitude * contrarian_term
        )
        streak_pred = mr_pred + streak_boost * streak_term
        shock_adjustment = shock_scale * self._shock_adjustment(history, regime)
        final_pred = (
            component_weights.get('base', 0.0) * base_pred
            + component_weights.get('arima', 0.0) * arima_step
            + component_weights.get('mr', 0.0) * mr_pred
            + component_weights.get('streak', 0.0) * streak_pred
            + shock_adjustment
        )
        return final_pred, regime

    def _combine_prediction(self, arima_step, lstm_step, history):
        return self._assemble_prediction(arima_step, lstm_step, history)

    def _rolling_components(self, X_roll, y_actual):
        X_weighted_all = self.apply_feature_weights(X_roll, self.feature_weights)
        history_X = self.history_X.copy() if self.history_X is not None else None
        history_resid = self.history_resid.copy() if self.history_resid is not None else None

        try:
            arima_roll = self.arima_model.predict_rolling(y_actual, verbose=False)
        except Exception:
            arima_roll = np.full(len(y_actual), self.train_mean)

        history_y = list(self.target_series)
        components = []

        for i in range(len(X_weighted_all)):
            x_step = X_weighted_all[i:i+1]
            arima_step = float(arima_roll[i]) if i < len(arima_roll) else float(self.train_mean)

            if history_X is not None and history_resid is not None:
                X_combined = np.concatenate([history_X, x_step])
                resid_combined = np.concatenate([history_resid, np.zeros(1)])
                X_lstm_full = np.column_stack([X_combined, resid_combined])
                lstm_full = self.lstm_model.predict(X_lstm_full)
                lstm_step = float(lstm_full[-1])
            else:
                X_lstm = np.column_stack([x_step, np.zeros(1)])
                lstm_step = float(self.lstm_model.predict(X_lstm)[-1])

            components.append({
                'arima_step': arima_step,
                'lstm_step': lstm_step,
                'history': np.array(history_y, dtype=float).copy(),
            })

            if history_X is not None and history_resid is not None:
                history_X = np.concatenate([history_X, x_step])[-self.sequence_length:]
                residual = float(y_actual[i]) - arima_step
                history_resid = np.concatenate([history_resid, [residual]])[-self.sequence_length:]
            history_y.append(float(y_actual[i]))

        return components

    def _predict_from_components(self, components, params=None):
        preds = np.zeros(len(components))
        regimes = []
        for i, comp in enumerate(components):
            preds[i], regime = self._assemble_prediction(
                comp['arima_step'],
                comp['lstm_step'],
                comp['history'],
                params=params,
            )
            regimes.append(regime)
        return preds, regimes

    def _tune_with_validation(self, X_val, y_val):
        if X_val is None or y_val is None or len(y_val) < 100:
            return

        components = self._rolling_components(X_val, y_val)
        if len(components) == 0:
            return

        prev_values = np.concatenate([[self.last_train_y], y_val[:-1]])
        candidates = []
        weight_candidates = [
            {'base': 0.55, 'arima': 0.15, 'mr': 0.15, 'streak': 0.15},
            {'base': 0.45, 'arima': 0.20, 'mr': 0.20, 'streak': 0.15},
            {'base': 0.35, 'arima': 0.25, 'mr': 0.20, 'streak': 0.20},
            {'base': 0.25, 'arima': 0.30, 'mr': 0.20, 'streak': 0.25},
            {'base': 0.40, 'arima': 0.10, 'mr': 0.25, 'streak': 0.25},
        ]
        rng = np.random.default_rng(42)
        for _ in range(6):
            sample = rng.dirichlet(np.array([3.0, 2.0, 2.0, 2.0]))
            weight_candidates.append({
                'base': float(sample[0]),
                'arima': float(sample[1]),
                'mr': float(sample[2]),
                'streak': float(sample[3]),
            })

        for component_weights in weight_candidates:
            for mr_speed in [0.03, 0.06]:
                for contrarian_magnitude in [0.5, 1.0]:
                    for streak_boost in [0.5, 1.25]:
                        for shock_scale in [0.5, 1.0]:
                            candidates.append({
                                'component_weights': component_weights,
                                'mr_speed': mr_speed,
                                'contrarian_magnitude': contrarian_magnitude,
                                'streak_boost': streak_boost,
                                'shock_scale': shock_scale,
                            })

        candidates.append({
            'component_weights': self.component_weights.copy(),
            'mr_speed': self.mr_speed,
            'contrarian_magnitude': self.contrarian_magnitude,
            'streak_boost': self.streak_boost,
            'shock_scale': self.shock_scale,
        })

        best_params = None
        best_rmse = np.inf
        best_da = -np.inf

        for params in candidates:
            preds, _ = self._predict_from_components(components, params=params)
            rmse = np.sqrt(np.mean((y_val - preds) ** 2))
            pred_up = (preds > prev_values).astype(int)
            actual_up = (y_val > prev_values).astype(int)
            da = 100 * np.mean(pred_up == actual_up)

            if rmse + 1e-8 < best_rmse or (abs(rmse - best_rmse) <= 0.02 and da > best_da):
                best_rmse = rmse
                best_da = da
                best_params = params

        if best_params is not None:
            self.component_weights = best_params['component_weights'].copy()
            self.mr_speed = best_params['mr_speed']
            self.contrarian_magnitude = best_params['contrarian_magnitude']
            self.streak_boost = best_params['streak_boost']
            self.shock_scale = best_params['shock_scale']
            self.validation_tuned = True

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        """Fit the enhanced hybrid model."""
        self.target_series = y_train.copy()
        self.train_mean = np.mean(y_train)
        self.last_train_y = y_train[-1]
        self.train_returns = self._pct_returns(y_train)
        self.train_vol_reference = max(np.std(self.train_returns), 1e-6) if len(self.train_returns) else 1e-6

        # Stage 1: Fit ARIMA
        if verbose:
            print("\n[Hybrid] Stage 1: Fitting ARIMA...")
        self.arima_model = ARIMAModel()
        self.arima_model.fit(y_train, order=self.arima_order, verbose=verbose)
        arima_residuals = self.arima_model.get_residuals()

        # Stage 2: Fit LSTM
        if verbose:
            print("[Hybrid] Stage 2: Fitting LSTM on residuals + features...")

        X_weighted = self.apply_feature_weights(X_train, self.feature_weights)
        self.direction_X_train = X_weighted.copy()

        if len(X_weighted) >= self.sequence_length:
            self.history_X = X_weighted[-self.sequence_length:]
            
        min_len = min(len(X_weighted), len(arima_residuals))
        X_aligned = X_weighted[-min_len:]
        residuals_aligned = arima_residuals[-min_len:]
        
        if len(X_aligned) >= self.sequence_length:
            self.history_X = X_aligned[-self.sequence_length:]
            self.history_resid = residuals_aligned[-self.sequence_length:]

        # LSTM learns to predict ARIMA residuals given features + past residuals.
        X_lstm = np.column_stack([X_aligned, residuals_aligned])

        X_val_lstm = None
        y_val_aligned = None
        if X_val is not None and y_val is not None:
            X_val_weighted = self.apply_feature_weights(X_val, self.feature_weights)
            # Build residuals for validation to avoid train/val distribution shift.
            try:
                arima_val_pred = self.arima_model.predict_rolling(y_val, verbose=False)
                val_residuals = y_val - arima_val_pred
            except Exception:
                val_residuals = np.zeros(len(y_val))
            X_val_lstm = np.column_stack([X_val_weighted, val_residuals])
            y_val_aligned = val_residuals

        self.lstm_model = LSTMModel(
            input_size=X_lstm.shape[1],
            sequence_length=min(self.sequence_length, min_len // 2),
            epochs=40,
            patience=8
        )
        # Train on residual targets instead of levels.
        self.lstm_model.fit(X_lstm, residuals_aligned, X_val_lstm, y_val_aligned, verbose=verbose)

        # Stage 3: Fit Directional Ensemble (KEY ENHANCEMENT!)
        if self.use_directional_ensemble:
            if verbose:
                print("[Hybrid] Stage 3: Fitting Directional Ensemble...")
            
            # Create direction-focused features
            direction_features = self.feature_engineer.create_direction_features(y_train, X_weighted)
            
            self.directional_ensemble = DirectionalEnsemble()
            
            if X_val is not None and y_val is not None and len(y_val) > 10:
                direction_features_val = self.feature_engineer.create_direction_features(y_val, X_val_weighted)
                self.directional_ensemble.fit(
                    direction_features, y_train,
                    direction_features_val, y_val,
                    verbose=verbose
                )
            else:
                self.directional_ensemble.fit(direction_features, y_train, verbose=verbose)

        if X_val is not None and y_val is not None:
            self._tune_with_validation(X_val, y_val)

        if verbose:
            print(
                f"[Hybrid] Training complete! "
                f"weights={self.component_weights}, mr_speed={self.mr_speed:.2f}, "
                f"contrarian={self.contrarian_magnitude:.2f}, streak={self.streak_boost:.2f}"
            )

    def predict(self, X, n_steps=None):
        """Generate level predictions."""
        if n_steps is None:
            n_steps = len(X)

        try:
            arima_pred = self.arima_model.predict(steps=n_steps)
        except (ValueError, np.linalg.LinAlgError, RuntimeError):
            arima_pred = np.full(n_steps, self.train_mean)

        X_weighted = self.apply_feature_weights(X, self.feature_weights)
        current_resid = np.zeros(len(X))
        
        if self.history_X is not None and self.history_resid is not None:
            X_combined = np.concatenate([self.history_X, X_weighted])
            resid_combined = np.concatenate([self.history_resid, current_resid])
            X_lstm_full = np.column_stack([X_combined, resid_combined])
            full_pred = self.lstm_model.predict(X_lstm_full)
            lstm_pred = full_pred[-len(X):]
        else:
            X_lstm = np.column_stack([X_weighted, current_resid])
            lstm_pred = self.lstm_model.predict(X_lstm)

        min_len = min(len(arima_pred), len(lstm_pred), len(X))
        hybrid_pred = np.zeros(min_len)
        history = list(self.target_series)

        for i in range(min_len):
            hybrid_pred[i], _ = self._combine_prediction(
                float(arima_pred[i]),
                float(lstm_pred[i]),
                history
            )
            history.append(hybrid_pred[i])

        return hybrid_pred

    def predict_rolling(self, X_test, y_test_actual):
        """
        Rolling one-step predictions using actuals to update history.

        This avoids multi-step flatlining in DA while keeping the model fixed.

        Key fix: The ARIMA component now uses rolling actuals (last known value)
        rather than a stale forecast from end of training. This approximates
        what ARIMA(0,1,q) does for one-step-ahead prediction.
        """
        components = self._rolling_components(X_test, y_test_actual)
        preds, regimes = self._predict_from_components(components)
        self.regime_history = regimes
        return preds

    def predict_with_direction(self, X, y_context=None, y_actual=None, use_rolling_levels=True):
        """
        ENHANCED: Predict with both level and direction outputs.
        
        Uses ensemble of:
        1. Direction from level predictions
        2. Direction from ensemble classifier (when features available)
        3. Weighted combination based on confidence
        """
        if use_rolling_levels and y_actual is not None:
            level_pred = self.predict_rolling(X, y_actual)
        else:
            level_pred = self.predict(X)
        
        # Method 1: Direction from level predictions
        if y_context is not None:
            prev_value = y_context[-1]
        else:
            prev_value = self.last_train_y
        
        level_direction = (np.diff(np.concatenate([[prev_value], level_pred])) > 0).astype(int)
        
        # Method 2: Direction from ensemble (if available and properly fitted)
        ensemble_direction = None
        ensemble_prob = None
        ensemble_conf = None
        
        if self.directional_ensemble is not None and self.directional_ensemble.is_fitted and y_context is not None:
            try:
                # Build full history for feature creation using actuals when available.
                if y_actual is not None:
                    y_full = np.concatenate([y_context, y_actual])
                else:
                    y_full = np.concatenate([y_context, level_pred])

                if self.direction_X_train is not None:
                    X_full = np.concatenate([self.direction_X_train, X])
                else:
                    X_full = X

                min_len = min(len(y_full), len(X_full))
                y_full = y_full[:min_len]
                X_full = X_full[:min_len]

                direction_features = self.feature_engineer.create_direction_features(y_full, X_full)

                # Use features aligned with test window (one-step-ahead).
                start_idx = max(0, len(y_full) - len(X))
                end_idx = max(0, len(y_full) - 1)
                features_for_pred = direction_features[start_idx:end_idx]

                if len(features_for_pred) > 0:
                    dir_pred, prob, conf = self.directional_ensemble.predict_direction(features_for_pred)

                    ensemble_direction = dir_pred
                    ensemble_prob = prob
                    ensemble_conf = conf
                        
            except Exception as e:
                # Fallback on any error
                pass
        
        # Combine methods
        if ensemble_direction is not None:
            # Weight ensemble more for early predictions, level-based for later
            min_len = min(len(level_direction), len(ensemble_direction))
            combined_direction = np.zeros(min_len, dtype=int)
            combined_prob = np.zeros(min_len)
            combined_conf = np.zeros(min_len)

            for i in range(min_len):
                ensemble_weight = 0.85 * (0.95 ** i)  # BOOSTED: More ensemble weight for DA improvement
                level_weight = 1 - ensemble_weight
                
                # Weighted probability
                level_prob = 0.5 + 0.3 * (1 if level_direction[i] == 1 else -1)  # Convert to soft probability
                combined_prob[i] = ensemble_weight * ensemble_prob[i] + level_weight * level_prob
                
                # Direction from combined probability
                combined_direction[i] = 1 if combined_prob[i] >= 0.5 else 0
                
                # Confidence from agreement - BOOSTED confidence calculation
                agreement = 1.0 if ensemble_direction[i] == level_direction[i] else 0.4  # Increased from 0.3
                combined_conf[i] = agreement * ensemble_conf[i]
            
            direction_pred = combined_direction
            direction_prob = combined_prob
            confidence = combined_conf
        else:
            direction_pred = level_direction
            direction_prob = np.where(level_direction == 1, 0.55, 0.45)  # Slight confidence
            confidence = np.abs(direction_prob - 0.5) * 2
        
        min_len = min(len(level_pred), len(direction_pred))
        
        return {
            'level_pred': level_pred[:min_len],
            'direction_pred': direction_pred[:min_len],
            'direction_prob': direction_prob[:min_len],
            'confidence': confidence[:min_len]
        }
    
    def compute_directional_accuracy(self, y_true, predictions_dict):
        """Compute directional accuracy metrics."""
        actual_direction = (np.diff(y_true) > 0).astype(int)
        pred_direction = predictions_dict['direction_pred']
        
        min_len = min(len(actual_direction), len(pred_direction))
        actual_direction = actual_direction[:min_len]
        pred_direction = pred_direction[:min_len]
        confidence = predictions_dict['confidence'][:min_len]
        
        overall_da = np.mean(actual_direction == pred_direction)
        
        high_conf_mask = confidence > 0.3
        if np.sum(high_conf_mask) > 10:
            high_conf_da = np.mean(actual_direction[high_conf_mask] == pred_direction[high_conf_mask])
            high_conf_coverage = np.mean(high_conf_mask)
        else:
            high_conf_da = overall_da
            high_conf_coverage = 0.0
        
        n_correct = np.sum(actual_direction == pred_direction)
        n_total = len(actual_direction)
        
        try:
            binom_result = stats.binomtest(n_correct, n_total, p=0.5, alternative='greater')
            p_value = binom_result.pvalue
        except AttributeError:
            from scipy.stats import binom
            p_value = 1 - binom.cdf(n_correct - 1, n_total, 0.5)

        return {
            'overall_da': overall_da,
            'high_conf_da': high_conf_da,
            'high_conf_coverage': high_conf_coverage,
            'n_correct': n_correct,
            'n_total': n_total,
            'p_value': p_value,
            'significant': p_value < 0.05
        }


class RandomWalkModel:
    """Random Walk baseline model."""
    
    def __init__(self):
        self.last_value = None
        
    def fit(self, y_train):
        self.last_value = y_train[-1]
        return self
        
    def predict(self, n_steps=1):
        return np.full(n_steps, self.last_value)


class MovingAverageModel:
    """Simple moving average baseline (multi-step)."""

    def __init__(self, window=20):
        self.window = window
        self.history = None

    def fit(self, y_train):
        self.history = list(y_train)
        return self

    def predict(self, n_steps=1):
        if not self.history:
            return np.full(n_steps, np.nan)

        history = list(self.history)
        preds = np.zeros(n_steps)

        for i in range(n_steps):
            window = history[-self.window:] if len(history) >= self.window else history
            preds[i] = np.mean(window)
            history.append(preds[i])

        return preds


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_directional_accuracy(y_true, y_pred):
    """Compute directional accuracy between actual and predicted."""
    actual_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    min_len = min(len(actual_dir), len(pred_dir))
    return np.mean(actual_dir[:min_len] == pred_dir[:min_len])


def evaluate_model_with_da(y_true, y_pred, model_name="Model"):
    """Comprehensive evaluation with DA focus."""
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    
    actual_dir = (np.diff(y_true) > 0).astype(int)
    pred_dir = (np.diff(y_pred) > 0).astype(int)
    da = np.mean(actual_dir == pred_dir)
    
    n_correct = np.sum(actual_dir == pred_dir)
    n_total = len(actual_dir)
    
    try:
        binom_result = stats.binomtest(n_correct, n_total, p=0.5, alternative='greater')
        p_value = binom_result.pvalue
    except AttributeError:
        from scipy.stats import binom
        p_value = 1 - binom.cdf(n_correct - 1, n_total, 0.5)
    
    print(f"\n{model_name} Results:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  Directional Accuracy: {da:.1%}")
    print(f"  DA p-value: {p_value:.4f} {'(Significant!)' if p_value < 0.05 else ''}")
    
    return {
        'rmse': rmse, 'mae': mae, 'mape': mape,
        'da': da, 'da_p_value': p_value,
        'da_significant': p_value < 0.05
    }


# ============================================================================
# MAIN - TEST THE ENHANCED MODELS
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TESTING ENHANCED HYBRID MODEL WITH DIRECTIONAL ACCURACY FOCUS")
    print("=" * 70)
    
    np.random.seed(42)
    n = 1500
    
    trend = np.cumsum(np.random.randn(n) * 0.3 + 0.05)
    regime_shift = np.where(np.arange(n) > 1000, 150, 0)
    noise = np.random.randn(n) * 10
    usdngn = 800 + trend * 30 + regime_shift + noise
    
    oil = 70 + np.cumsum(np.random.randn(n) * 0.2)
    mpr = 15 + np.cumsum(np.random.randn(n) * 0.01)
    cpi = 20 + np.cumsum(np.random.randn(n) * 0.03)
    
    X = np.column_stack([oil, mpr, cpi])
    y = usdngn
    
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    print(f"\nData: {n} observations")
    print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    
    # Test Random Walk
    print("\n" + "-" * 50)
    print("Testing Random Walk Baseline...")
    rw = RandomWalkModel()
    rw.fit(y_train)
    rw_pred = np.concatenate([[y_train[-1]], y_test[:-1]])
    rw_results = evaluate_model_with_da(y_test, rw_pred, "Random Walk")
    
    # Test Enhanced Hybrid
    print("\n" + "-" * 50)
    print("Testing Enhanced Hybrid ARIMA-LSTM...")
    hybrid = HybridARIMALSTM(
        sequence_length=30,
        arima_weight=0.3,
        use_directional_ensemble=True
    )
    hybrid.fit(X_train, y_train, X_val, y_val, verbose=True)
    
    predictions = hybrid.predict_with_direction(X_test, y_context=y_train)
    level_results = evaluate_model_with_da(y_test, predictions['level_pred'], "Hybrid (Levels)")
    
    print("\n" + "-" * 50)
    print("Enhanced Direction Predictions:")
    da_results = hybrid.compute_directional_accuracy(y_test, predictions)
    
    print(f"  Overall DA: {da_results['overall_da']:.1%}")
    print(f"  High-Confidence DA: {da_results['high_conf_da']:.1%} (coverage: {da_results['high_conf_coverage']:.0%})")
    print(f"  P-value: {da_results['p_value']:.4f}")
    print(f"  Statistically Significant: {'Yes ✓' if da_results['significant'] else 'No'}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nRandom Walk DA: {rw_results['da']:.1%}")
    print(f"Enhanced Hybrid DA: {da_results['overall_da']:.1%}")
    print(f"Improvement: +{(da_results['overall_da'] - rw_results['da']) * 100:.1f} percentage points")
    
    if da_results['high_conf_coverage'] > 0.2:
        print(f"\nHigh-Confidence Predictions ({da_results['high_conf_coverage']:.0%} of cases):")
        print(f"  DA: {da_results['high_conf_da']:.1%}")
    
    print("\n" + "=" * 70)
