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
except (ImportError, AttributeError):
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
    def directional_accuracy(y_true, y_pred, prev_values=None):
        """
        Compute directional accuracy for one-step-ahead predictions.

        For rolling one-step-ahead forecasting, the correct DA measures:
        "Did the model predict the correct direction of change from the
        last known value?"

        Parameters:
        -----------
        y_true : array-like
            Actual values
        y_pred : array-like
            Predicted values
        prev_values : array-like, optional
            Previous known values when each prediction was made.
            If None, uses y_true shifted by 1 (legacy diff-based method).
            For proper one-step-ahead DA, pass:
            np.concatenate([[last_train_value], y_true[:-1]])

        Returns:
        --------
        float : Directional accuracy percentage (0-100)
        """
        y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()
        min_len = min(len(y_true), len(y_pred))
        if min_len < 2:
            return 50.0
        y_true, y_pred = y_true[:min_len], y_pred[:min_len]

        if prev_values is not None:
            # Proper one-step-ahead DA: compare direction from last known value
            prev_values = np.array(prev_values).flatten()[:min_len]

            # Actual direction: did the rate go UP (1) or DOWN (0)?
            actual_up = (y_true > prev_values).astype(int)
            # Predicted direction: does the model predict UP or DOWN?
            pred_up = (y_pred > prev_values).astype(int)

            # Exclude exact ties (no change) - these are ambiguous
            # For RW: pred == prev, so pred_up = 0 always (predicts DOWN/no-change)
            # We include these since RW does make an implicit prediction
            return 100 * np.mean(actual_up == pred_up)
        else:
            # Legacy diff-based method (measures curve shape similarity)
            actual_dir = np.sign(np.diff(y_true))
            pred_dir = np.sign(np.diff(y_pred))
            return 100 * np.mean(actual_dir == pred_dir)

    @staticmethod
    def compute_all_metrics(y_true, y_pred, prev_values=None):
        """
        Compute all evaluation metrics.

        Parameters:
        -----------
        y_true : array-like
            Actual values
        y_pred : array-like
            Predicted values
        prev_values : array-like, optional
            Previous known values for proper one-step-ahead DA.
            Pass np.concatenate([[last_train_value], y_true[:-1]])
        """
        y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()
        min_len = min(len(y_true), len(y_pred))
        y_true, y_pred = y_true[:min_len], y_pred[:min_len]

        if prev_values is not None:
            prev_values = np.array(prev_values).flatten()[:min_len]

        return {
            'RMSE': ModelEvaluator.rmse(y_true, y_pred),
            'MAE': ModelEvaluator.mae(y_true, y_pred),
            'MAPE': ModelEvaluator.mape(y_true, y_pred),
            'DA': ModelEvaluator.directional_accuracy(y_true, y_pred, prev_values),
            'N': min_len
        }


class EnsembleUtils:
    """Utilities for simple ensemble construction and analysis."""

    @staticmethod
    def normalize_weights(weights, keys):
        if weights is None:
            return {k: 1.0 / len(keys) for k in keys}
        w = {k: float(weights.get(k, 0.0)) for k in keys}
        total = sum(w.values())
        if total <= 0:
            return {k: 1.0 / len(keys) for k in keys}
        return {k: v / total for k, v in w.items()}

    @staticmethod
    def weighted_average(predictions, weights=None):
        """
        Compute a weighted average across prediction arrays.

        Parameters:
        -----------
        predictions : dict[str, array-like]
            Mapping of model name to prediction vector.
        weights : dict[str, float], optional
            Model weights. Missing keys default to 0 and are renormalized.
        """
        if not predictions:
            return np.array([])

        keys = list(predictions.keys())
        w = EnsembleUtils.normalize_weights(weights, keys)

        min_len = min(len(np.array(predictions[k]).flatten()) for k in keys)
        if min_len == 0:
            return np.array([])

        ensemble = np.zeros(min_len, dtype=float)
        for k in keys:
            pred = np.array(predictions[k]).flatten()[:min_len]
            ensemble += w[k] * pred
        return ensemble

    @staticmethod
    def optimize_weights(
        predictions,
        y_true,
        prev_values=None,
        objective='balanced',
        n_samples=300,
        random_state=42,
    ):
        """
        Search for validation weights that improve both RMSE and directional accuracy.

        Parameters:
        -----------
        predictions : dict[str, array-like]
            Candidate member predictions on the validation set.
        y_true : array-like
            Validation actuals.
        prev_values : array-like, optional
            Previous-known-value vector for proper one-step-ahead DA.
        objective : str
            One of {'rmse', 'balanced'}.
        n_samples : int
            Number of random Dirichlet samples to evaluate.
        """
        if not predictions:
            return {}

        keys = list(predictions.keys())
        min_len = min(len(np.array(predictions[k]).flatten()) for k in keys)
        y_true = np.array(y_true).flatten()[:min_len]
        if prev_values is not None:
            prev_values = np.array(prev_values).flatten()[:min_len]

        candidate_weights = []
        equal = {k: 1.0 / len(keys) for k in keys}
        candidate_weights.append(equal)

        rmse_rows = []
        for k in keys:
            pred = np.array(predictions[k]).flatten()[:min_len]
            metrics = ModelEvaluator.compute_all_metrics(y_true, pred, prev_values=prev_values)
            rmse_rows.append((k, metrics))

        inv_rmse = {k: 1.0 / (m['RMSE'] + 1e-8) for k, m in rmse_rows}
        candidate_weights.append(EnsembleUtils.normalize_weights(inv_rmse, keys))

        inv_mae = {k: 1.0 / (m['MAE'] + 1e-8) for k, m in rmse_rows}
        candidate_weights.append(EnsembleUtils.normalize_weights(inv_mae, keys))

        best_rmse = min(m['RMSE'] for _, m in rmse_rows)
        worst_rmse = max(m['RMSE'] for _, m in rmse_rows)
        best_da = max(m['DA'] for _, m in rmse_rows)
        worst_da = min(m['DA'] for _, m in rmse_rows)

        rng = np.random.default_rng(random_state)
        alpha = np.ones(len(keys))
        for _ in range(n_samples):
            sample = rng.dirichlet(alpha)
            candidate_weights.append(dict(zip(keys, sample)))

        best_score = -np.inf
        best_weights = equal

        for weight_dict in candidate_weights:
            pred = EnsembleUtils.weighted_average(predictions, weights=weight_dict)
            metrics = ModelEvaluator.compute_all_metrics(y_true, pred, prev_values=prev_values)

            if objective == 'rmse':
                score = -metrics['RMSE']
            else:
                rmse_gain = (worst_rmse - metrics['RMSE']) / (worst_rmse - best_rmse + 1e-8)
                da_gain = (metrics['DA'] - worst_da) / (best_da - worst_da + 1e-8)
                score = 0.6 * rmse_gain + 0.4 * da_gain

            if score > best_score:
                best_score = score
                best_weights = EnsembleUtils.normalize_weights(weight_dict, keys)

        return best_weights


class WalkForwardValidator:
    """
    Walk-forward validation utilities.

    This is intentionally lightweight and model-agnostic via callbacks.
    """

    @staticmethod
    def validate_univariate(
        y,
        model_factory,
        initial_train_size,
        horizon=1,
        step_size=1,
        fit_kwargs=None,
        predict_kwargs=None,
    ):
        """
        Perform univariate walk-forward validation.

        The model must support:
        1. model = model_factory()
        2. model.fit(y_train, **fit_kwargs)
        3. model.predict(horizon, **predict_kwargs)
        """
        fit_kwargs = fit_kwargs or {}
        predict_kwargs = predict_kwargs or {}

        y = np.array(y).flatten()
        n = len(y)

        if initial_train_size < 5 or initial_train_size >= n:
            raise ValueError("initial_train_size must be >=5 and < len(y)")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")

        preds = []
        actuals = []
        anchors = []

        train_end = initial_train_size
        while train_end + horizon <= n:
            y_train = y[:train_end]
            y_next = y[train_end:train_end + horizon]

            model = model_factory()
            model.fit(y_train, **fit_kwargs)
            y_pred = np.array(model.predict(horizon, **predict_kwargs)).flatten()

            step_len = min(len(y_next), len(y_pred))
            preds.append(y_pred[:step_len])
            actuals.append(y_next[:step_len])
            anchors.extend([train_end] * step_len)

            train_end += step_size

        if not preds:
            return {
                'predictions': np.array([]),
                'actuals': np.array([]),
                'anchors': np.array([]),
                'metrics': {}
            }

        y_pred_all = np.concatenate(preds)
        y_true_all = np.concatenate(actuals)
        metrics = ModelEvaluator.compute_all_metrics(y_true_all, y_pred_all)

        return {
            'predictions': y_pred_all,
            'actuals': y_true_all,
            'anchors': np.array(anchors),
            'metrics': metrics
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

    @staticmethod
    def compare_many(y_true, benchmark_predictions, candidate_predictions, benchmark_name, loss='MSE'):
        rows = []
        for name, preds in candidate_predictions.items():
            if name == benchmark_name:
                continue
            result = DieboldMarianoTest.test(y_true, preds, benchmark_predictions, loss=loss)
            rows.append({
                'Benchmark': benchmark_name,
                'Model': name,
                'Loss': loss,
                **result,
            })
        return pd.DataFrame(rows)

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
    
    def compute_importance(self, X, n_repeats=10, max_samples=40):
        X = np.array(X)
        n_features = X.shape[1]
        X = X[:min(max_samples, len(X))]
        
        if not SHAP_AVAILABLE:
            return self._fallback_importance(X, n_repeats)
        
        try:
            def model_predict(x):
                if isinstance(x, np.ndarray):
                    return self.model.predict(x)
                return self.model.predict(np.array(x))
            
            background = X[:min(20, len(X))]
            if len(background) == 0:
                return self._fallback_importance(X, n_repeats)

            self.explainer = shap.KernelExplainer(model_predict, background)
            
            shap_values = self.explainer.shap_values(X, nsamples=min(80, max(20, 2 * X.shape[1])))
            
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
        except (ValueError, TypeError, RuntimeError):
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
                except (ValueError, TypeError, RuntimeError):
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
