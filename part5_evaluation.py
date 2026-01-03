"""
PART 5: MODEL EVALUATION AND SHAP EXPLAINABILITY
PhD Thesis: USD-NGN Exchange Rate Forecasting
Author: Oche Emmanuel Ike (242220011)
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

class ModelEvaluator:
    @staticmethod
    def rmse(y_true, y_pred):
        return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))
    
    @staticmethod
    def mae(y_true, y_pred):
        return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
    
    @staticmethod
    def mape(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mask = y_true != 0
        return 100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
    
    @staticmethod
    def directional_accuracy(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        if len(y_true) < 2:
            return 50.0
        actual_dir = np.sign(np.diff(y_true))
        pred_dir = np.sign(np.diff(y_pred))
        return 100 * np.mean(actual_dir == pred_dir)
    
    @staticmethod
    def compute_all_metrics(y_true, y_pred):
        y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()
        min_len = min(len(y_true), len(y_pred))
        y_true, y_pred = y_true[:min_len], y_pred[:min_len]
        return {
            'RMSE': ModelEvaluator.rmse(y_true, y_pred),
            'MAE': ModelEvaluator.mae(y_true, y_pred),
            'MAPE': ModelEvaluator.mape(y_true, y_pred),
            'DA': ModelEvaluator.directional_accuracy(y_true, y_pred),
            'N': min_len
        }

class DieboldMarianoTest:
    @staticmethod
    def test(y_true, pred1, pred2, loss='MSE'):
        y_true, pred1, pred2 = np.array(y_true), np.array(pred1), np.array(pred2)
        min_len = min(len(y_true), len(pred1), len(pred2))
        y_true, pred1, pred2 = y_true[:min_len], pred1[:min_len], pred2[:min_len]
        e1, e2 = y_true - pred1, y_true - pred2
        d = e1**2 - e2**2 if loss == 'MSE' else np.abs(e1) - np.abs(e2)
        d_bar = np.mean(d)
        T = len(d)
        if T < 10:
            return {'dm_stat': np.nan, 'p_value': np.nan, 'conclusion': 'Insufficient data'}
        var_d = np.var(d, ddof=1) / T
        dm_stat = d_bar / np.sqrt(max(var_d, 1e-10))
        p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
        if p_value < 0.05:
            conclusion = "Model 1 better ***" if dm_stat < 0 else "Model 2 better ***"
        else:
            conclusion = "No significant difference"
        return {'dm_stat': dm_stat, 'p_value': p_value, 'conclusion': conclusion}

class RegimeEvaluator:
    REGIMES = {
        'Pre-Crisis': ('2010-01-01', '2014-06-30'),
        'Oil Crisis': ('2014-07-01', '2016-12-31'),
        'Recovery': ('2017-01-01', '2019-12-31'),
        'COVID-19': ('2020-01-01', '2021-12-31'),
        'Post-COVID': ('2022-01-01', '2023-05-31'),
        'Depegging': ('2023-06-01', '2025-12-31')
    }
    
    @staticmethod
    def evaluate_by_regime(y_true, y_pred, dates):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        if not isinstance(dates, pd.DatetimeIndex):
            dates = pd.to_datetime(dates)
        min_len = min(len(y_true), len(y_pred), len(dates))
        y_true, y_pred, dates = y_true[:min_len], y_pred[:min_len], dates[:min_len]
        results = {}
        for regime_name, (start, end) in RegimeEvaluator.REGIMES.items():
            mask = (dates >= start) & (dates <= end)
            if mask.sum() > 10:
                metrics = ModelEvaluator.compute_all_metrics(y_true[mask], y_pred[mask])
                metrics['Regime'] = regime_name
                results[regime_name] = metrics
        return pd.DataFrame(results).T

class SHAPExplainer:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.shap_values = None
        self.explainer = None
    
    def compute_importance(self, X, n_repeats=10):
        X = np.array(X)
        n_features = X.shape[1]
        
        if not SHAP_AVAILABLE:
            return self._fallback_importance(X, n_repeats)
        
        try:
            def model_predict(x):
                if isinstance(x, np.ndarray):
                    return self.model.predict(x)
                return self.model.predict(np.array(x))
            
            self.explainer = shap.KernelExplainer(model_predict, X[:min(100, len(X))])
            
            shap_values = self.explainer.shap_values(X[:min(100, len(X))])
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            if shap_values is None or len(shap_values) == 0:
                return self._fallback_importance(X, n_repeats)
            
            importance = np.abs(shap_values).mean(axis=0)
            
            if importance.sum() > 0:
                importance = importance / importance.sum()
            else:
                importance = np.ones(n_features) / n_features
            
            self.shap_values = importance
            return importance
        
        except Exception as e:
            return self._fallback_importance(X, n_repeats)
    
    def _fallback_importance(self, X, n_repeats):
        n_features = X.shape[1]
        try:
            baseline_pred = self.model.predict(X)
            baseline_var = np.var(baseline_pred)
        except:
            return np.ones(n_features) / n_features
        
        importance = np.zeros(n_features)
        for i in range(n_features):
            importance_scores = []
            for _ in range(n_repeats):
                X_perm = X.copy()
                np.random.shuffle(X_perm[:, i])
                try:
                    perm_var = np.var(self.model.predict(X_perm))
                    importance_scores.append(abs(baseline_var - perm_var))
                except:
                    pass
            
            if importance_scores:
                importance[i] = np.mean(importance_scores)
        
        if importance.sum() > 0:
            importance = importance / importance.sum()
        else:
            importance = np.ones(n_features) / n_features
        
        self.shap_values = importance
        return importance
    
    def get_importance_df(self):
        if self.shap_values is None:
            return pd.DataFrame()
        
        return pd.DataFrame({
            'feature': self.feature_names[:len(self.shap_values)],
            'importance': self.shap_values,
            'importance_pct': 100 * self.shap_values
        }).sort_values('importance', ascending=False)

if __name__ == "__main__":
    np.random.seed(42)
    y_true = 400 + 2 * np.arange(500) + 50 * np.random.randn(500)
    y_pred = y_true + 20 * np.random.randn(500)
    metrics = ModelEvaluator.compute_all_metrics(y_true, y_pred)
    print(f"RMSE: {metrics['RMSE']:.2f}, MAE: {metrics['MAE']:.2f}, DA: {metrics['DA']:.1f}%")
