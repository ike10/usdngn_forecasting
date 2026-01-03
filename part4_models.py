"""
PART 4: MODEL DEVELOPMENT (ARIMA, LSTM, HYBRID)
PhD Thesis: USD-NGN Exchange Rate Forecasting
Author: Oche Emmanuel Ike (242220011)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.arima.model import ARIMA
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

class ARIMAModel:
    def __init__(self, max_p=5, max_d=2, max_q=5):
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.model = None
        self.fitted = None
        self.best_order = None
        self.series = None
        
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
        except:
            self._fit_ar_fallback(series)
        return self
    
    def _grid_search(self, series, verbose):
        if not STATSMODELS_AVAILABLE:
            return (1, 1, 1)
        best_aic, best_order = np.inf, (1, 1, 1)
        # Reduced grid search for speed
        for p in range(min(3, self.max_p + 1)):
            for d in range(min(2, self.max_d + 1)):
                for q in range(min(3, self.max_q + 1)):
                    if p == 0 and q == 0:
                        continue
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        fitted = model.fit(disp=False)
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                    except:
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
    
    def get_residuals(self):
        if self.fitted:
            return np.array(self.fitted.resid)
        return self.series[1:] - self.series[:-1]

class PyTorchLSTM(nn.Module):
    """Deep Learning LSTM model for time series forecasting."""
    def __init__(self, input_size, hidden_size1=128, hidden_size2=64, output_size=1, dropout=0.2):
        super(PyTorchLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True, return_sequences=True)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True, return_sequences=False)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size2, 32)
        self.relu = nn.ReLU()
        self.output = nn.Linear(32, output_size)
    
    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.relu(x)
        x = self.output(x)
        return x

class LSTMModel:
    def __init__(self, input_size, sequence_length=60, batch_size=32, epochs=100, patience=20):
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def create_sequences(self, X, y):
        X_seq, y_seq = [], []
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])
        return np.array(X_seq), np.array(y_seq)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        if not TORCH_AVAILABLE:
            if verbose:
                print("  PyTorch not available")
            return {}
        
        self.scaler_X.fit(X_train)
        self.scaler_y.fit(y_train.reshape(-1, 1))
        X_scaled = self.scaler_X.transform(X_train)
        y_scaled = self.scaler_y.transform(y_train.reshape(-1, 1)).flatten()
        
        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled)
        
        if len(X_seq) < self.batch_size:
            self.batch_size = max(1, len(X_seq) // 4)
        
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).reshape(-1, 1).to(self.device)
        
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model = PyTorchLSTM(input_size=X_seq.shape[2]).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
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
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            if X_val is not None and y_val is not None:
                val_loss = self._evaluate(X_val, y_val, criterion)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch+1}")
                    break
        
        if verbose:
            print(f"  LSTM model trained for {epoch+1} epochs")
        
        return {}
    
    def _evaluate(self, X_val, y_val, criterion):
        if len(X_val) < self.sequence_length:
            return float('inf')
        
        self.scaler_X.fit(X_val)
        X_scaled = self.scaler_X.transform(X_val)
        y_scaled = self.scaler_y.transform(y_val.reshape(-1, 1)).flatten()
        
        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled)
        
        if len(X_seq) == 0:
            return float('inf')
        
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).reshape(-1, 1).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_tensor)
            loss = criterion(y_pred, y_tensor)
        
        return loss.item()
    
    def predict(self, X):
        if self.model is None:
            return np.zeros(len(X))
        
        if len(X) < self.sequence_length:
            return np.zeros(len(X))
        
        X_scaled = self.scaler_X.transform(X)
        X_seq, _ = self.create_sequences(X_scaled, np.zeros(len(X_scaled)))
        
        if len(X_seq) == 0:
            return np.zeros(len(X))
        
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_tensor)
        
        y_pred = y_pred.cpu().numpy().flatten()
        y_pred_original = self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        
        padding = len(X) - len(y_pred_original)
        if padding > 0:
            y_pred_original = np.concatenate([np.zeros(padding), y_pred_original])
        
        return y_pred_original[:len(X)]

class HybridARIMALSTM:
    def __init__(self, arima_order=None, feature_weights=None, sequence_length=60):
        self.arima_order = arima_order
        self.feature_weights = feature_weights
        self.sequence_length = sequence_length
        self.arima_model = None
        self.lstm_model = None
        self.target_series = None
        
    def apply_feature_weights(self, X, weights):
        if weights is None:
            return X
        weights = np.array(weights).flatten()
        if len(weights) != X.shape[1]:
            weights = np.ones(X.shape[1])
        return X * weights
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        self.target_series = y_train
        print("\n[Hybrid] Stage 1: Fitting ARIMA...")
        self.arima_model = ARIMAModel()
        self.arima_model.fit(y_train, order=self.arima_order, verbose=verbose)
        arima_residuals = self.arima_model.get_residuals()
        print("[Hybrid] Stage 2: Fitting LSTM on residuals...")
        X_weighted = self.apply_feature_weights(X_train, self.feature_weights)
        min_len = min(len(X_weighted), len(arima_residuals))
        X_weighted = X_weighted[-min_len:]
        arima_residuals = arima_residuals[-min_len:]
        y_train_aligned = y_train[-min_len:]
        X_lstm = np.column_stack([X_weighted, arima_residuals])
        self.lstm_model = LSTMModel(input_size=X_lstm.shape[1], sequence_length=self.sequence_length)
        self.lstm_model.fit(X_lstm, y_train_aligned, verbose=verbose)
        print("[Hybrid] Training complete!")
        
    def predict(self, X, n_steps=None):
        X_weighted = self.apply_feature_weights(X, self.feature_weights)
        if n_steps is None:
            n_steps = len(X)
        try:
            arima_pred = self.arima_model.predict(steps=n_steps)
        except:
            arima_pred = np.full(n_steps, np.mean(self.target_series))
        X_lstm = np.column_stack([X_weighted, np.zeros(len(X_weighted))])
        try:
            lstm_pred = self.lstm_model.predict(X_lstm)
        except:
            lstm_pred = np.zeros(len(X))
        min_len = min(len(arima_pred), len(lstm_pred), len(X))
        hybrid_pred = lstm_pred[:min_len]
        if len(arima_pred) >= min_len:
            arima_trend = arima_pred[:min_len] - np.mean(arima_pred[:min_len])
            hybrid_pred = hybrid_pred + 0.3 * arima_trend
        return hybrid_pred

class RandomWalkModel:
    def __init__(self):
        self.last_value = None
    def fit(self, y_train):
        self.last_value = y_train[-1]
        return self
    def predict(self, n_steps=1):
        return np.full(n_steps, self.last_value)

if __name__ == "__main__":
    from part1_data_collection import DataCollector
    from part2_preprocessing import DataPreprocessor, DataSplitter
    collector = DataCollector()
    raw_df = collector.collect_all_data()
    preprocessor = DataPreprocessor(raw_df)
    processed_df, _ = preprocessor.preprocess()
    splitter = DataSplitter()
    train, val, test = splitter.split(processed_df)
    feature_cols = ['brent_oil', 'mpr', 'cpi', 'oil_return', 'usdngn_volatility']
    available = [f for f in feature_cols if f in train.columns]
    X_train, y_train = train[available].values, train['usdngn'].values
    hybrid = HybridARIMALSTM()
    hybrid.fit(X_train, y_train)
