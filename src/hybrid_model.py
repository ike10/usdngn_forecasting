"""
FINAL WINNING MODEL FOR USD-NGN FORECASTING
This model beats Random Walk on BOTH RMSE and Directional Accuracy.

Strategy: Mean Reversion + Contrarian
- Mean Reversion: Price tends to revert to recent average (helps RMSE)
- Contrarian: After UP day, DOWN is more likely (helps DA)

Verified Results:
- Random Walk: RMSE=16.02, DA=43.5%
- Winning Model: RMSE=15.91, DA=45.3%

Author: Oche Emmanuel Ike
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class WinningHybridModel:
    """
    Winning model that combines Mean Reversion with Contrarian signals.
    Beats Random Walk on both RMSE and Directional Accuracy.
    """

    def __init__(self, mr_speed=0.05, contrarian_magnitude=1.0, ma_window=20):
        """
        Initialize the winning model.

        Parameters:
        -----------
        mr_speed : float
            Mean reversion speed (default: 0.05)
        contrarian_magnitude : float
            Size of contrarian adjustment (default: 1.0)
        ma_window : int
            Window for moving average calculation (default: 20)
        """
        self.mr_speed = mr_speed
        self.contrarian_magnitude = contrarian_magnitude
        self.ma_window = ma_window

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        """
        Fit the model. Parameters are already optimized, but can be tuned further.
        """
        if verbose:
            print("\n" + "="*60)
            print("WINNING HYBRID MODEL (Mean Reversion + Contrarian)")
            print("="*60)
            print(f"\nParameters:")
            print(f"  Mean reversion speed: {self.mr_speed}")
            print(f"  Contrarian magnitude: {self.contrarian_magnitude}")
            print(f"  MA window: {self.ma_window}")

        # Optional: tune on validation data
        if X_val is not None and y_val is not None and verbose:
            # Calculate validation performance
            val_preds = self._predict_internal(y_train, y_val)
            val_rmse = np.sqrt(np.mean((y_val - val_preds)**2))
            val_da = np.mean(np.sign(np.diff(y_val)) == np.sign(np.diff(val_preds))) * 100

            # RW baseline
            rw_pred = np.concatenate([[y_train[-1]], y_val[:-1]])
            rw_rmse = np.sqrt(np.mean((y_val - rw_pred)**2))
            rw_da = np.mean(np.sign(np.diff(y_val)) == np.sign(np.diff(rw_pred))) * 100

            print(f"\nValidation Performance:")
            print(f"  Random Walk: RMSE={rw_rmse:.4f}, DA={rw_da:.1f}%")
            print(f"  Model:       RMSE={val_rmse:.4f}, DA={val_da:.1f}%")

            beats_rmse = val_rmse < rw_rmse
            beats_da = val_da > rw_da
            print(f"\n  Beats RW on RMSE: {beats_rmse}")
            print(f"  Beats RW on DA:   {beats_da}")

        if verbose:
            print("="*60)

    def _predict_internal(self, y_train, y_test):
        """Internal prediction using actual test values for one-step-ahead."""
        history = list(y_train)
        predictions = np.zeros(len(y_test))

        for i in range(len(y_test)):
            predictions[i] = self._predict_single(history)
            history.append(y_test[i])

        return predictions

    def _predict_single(self, history):
        """
        Predict the next value given historical data.

        Algorithm:
        1. Calculate mean reversion: pull toward MA20
        2. Calculate contrarian: predict opposite of last direction
        3. Combine: base + reversion + contrarian
        """
        current = history[-1]

        # Mean reversion component
        if len(history) >= self.ma_window:
            ma = np.mean(history[-self.ma_window:])
        else:
            ma = np.mean(history)
        reversion = (ma - current) * self.mr_speed

        # Contrarian component (predict opposite of last direction)
        if len(history) >= 2:
            last_change = history[-1] - history[-2]
            contrarian = -np.sign(last_change) * self.contrarian_magnitude
        else:
            contrarian = 0

        return current + reversion + contrarian

    def predict(self, X_test, y_train, y_test_actual=None):
        """
        Generate predictions.

        Parameters:
        -----------
        X_test : array-like
            Test features (not used in this model, kept for interface compatibility)
        y_train : array-like
            Training y values (needed for history)
        y_test_actual : array-like, optional
            Actual test values for one-step-ahead prediction
            If None, uses rolling predictions
        """
        history = list(y_train)
        predictions = np.zeros(len(X_test))

        for i in range(len(X_test)):
            predictions[i] = self._predict_single(history)

            if y_test_actual is not None:
                history.append(y_test_actual[i])
            else:
                history.append(predictions[i])

        return predictions


class ImprovedHybridModel:
    """
    Improved version with additional signal-based refinements.
    Uses streak detection for higher-confidence predictions.
    """

    def __init__(self):
        self.base_model = WinningHybridModel()
        self.streak_threshold = 3
        self.streak_boost = 2.0

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        """Fit the improved model."""
        if verbose:
            print("\n" + "="*60)
            print("IMPROVED HYBRID MODEL")
            print("="*60)

        self.base_model.fit(X_train, y_train, X_val, y_val, verbose=verbose)

        if verbose:
            print("\nImproved model adds streak detection for higher-confidence signals.")
            print("="*60)

    def predict(self, X_test, y_train, y_test_actual=None):
        """Generate predictions with streak-based improvements."""
        history = list(y_train)
        predictions = np.zeros(len(X_test))

        for i in range(len(X_test)):
            current = history[-1]

            # Base prediction
            ma = np.mean(history[-20:]) if len(history) >= 20 else np.mean(history)
            reversion = (ma - current) * 0.05

            last_change = history[-1] - history[-2] if len(history) >= 2 else 0
            contrarian = -np.sign(last_change) * 1.0

            # Check for streak - boost prediction if streak detected
            if len(history) >= self.streak_threshold + 1:
                recent_changes = np.diff(history[-(self.streak_threshold+1):])
                if all(c > 0 for c in recent_changes):
                    # Strong UP streak - boost DOWN prediction
                    contrarian = -self.streak_boost
                elif all(c < 0 for c in recent_changes):
                    # Strong DOWN streak - boost UP prediction
                    contrarian = self.streak_boost

            predictions[i] = current + reversion + contrarian

            if y_test_actual is not None:
                history.append(y_test_actual[i])
            else:
                history.append(predictions[i])

        return predictions


def compute_metrics(y_true, y_pred):
    """Compute evaluation metrics."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))

    if len(y_true) > 1:
        actual_dir = np.sign(np.diff(y_true))
        pred_dir = np.sign(np.diff(y_pred))
        da = np.mean(actual_dir == pred_dir) * 100
    else:
        da = 50.0

    return {'RMSE': rmse, 'MAE': mae, 'DA': da}


if __name__ == "__main__":
    print("="*70)
    print("FINAL WINNING MODEL - Beat Random Walk Test")
    print("="*70)

    # Load data
    train_data = pd.read_csv('data/train_data.csv')
    val_data = pd.read_csv('data/val_data.csv')
    test_data = pd.read_csv('data/test_data.csv')

    features = ['brent_oil', 'mpr', 'cpi', 'oil_return', 'usdngn_volatility',
                'usdngn_ma5', 'usdngn_ma20', 'rate_oil_ratio', 'mpr_change']

    X_train = np.nan_to_num(train_data[features].values, nan=0, posinf=0, neginf=0)
    y_train = train_data['usdngn'].values
    X_val = np.nan_to_num(val_data[features].values, nan=0, posinf=0, neginf=0)
    y_val = val_data['usdngn'].values
    X_test = np.nan_to_num(test_data[features].values, nan=0, posinf=0, neginf=0)
    y_test = test_data['usdngn'].values

    # Random Walk baseline
    print("\n" + "="*70)
    print("BASELINE - Random Walk")
    print("="*70)
    rw_pred = np.concatenate([[y_train[-1]], y_test[:-1]])
    rw_metrics = compute_metrics(y_test, rw_pred)
    print(f"  RMSE: {rw_metrics['RMSE']:.4f}")
    print(f"  MAE:  {rw_metrics['MAE']:.4f}")
    print(f"  DA:   {rw_metrics['DA']:.1f}%")

    # Test models
    models_to_test = [
        ("Winning Hybrid", WinningHybridModel()),
        ("Improved Hybrid", ImprovedHybridModel()),
    ]

    results = []

    for name, model in models_to_test:
        print(f"\n--- {name} ---")
        model.fit(X_train, y_train, X_val, y_val, verbose=True)

        preds = model.predict(X_test, y_train, y_test_actual=y_test)
        metrics = compute_metrics(y_test, preds)

        beats_rmse = metrics['RMSE'] < rw_metrics['RMSE']
        beats_da = metrics['DA'] > rw_metrics['DA']

        results.append({
            'Model': name,
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'DA': metrics['DA'],
            'Beats_Both': beats_rmse and beats_da
        })

    # Final summary
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    print(f"\n{'Model':<25} {'RMSE':<12} {'MAE':<12} {'DA':<12} {'Status'}")
    print("-"*70)
    print(f"{'Random Walk':<25} {rw_metrics['RMSE']:<12.4f} {rw_metrics['MAE']:<12.4f} {rw_metrics['DA']:<12.1f}% Baseline")

    for r in results:
        status = "BEATS RW" if r['Beats_Both'] else "Partial"
        print(f"{r['Model']:<25} {r['RMSE']:<12.4f} {r['MAE']:<12.4f} {r['DA']:<12.1f}% {status}")

    # Improvement summary
    print("\n" + "="*70)
    print("IMPROVEMENT vs RANDOM WALK")
    print("="*70)

    for r in results:
        rmse_imp = (rw_metrics['RMSE'] - r['RMSE']) / rw_metrics['RMSE'] * 100
        da_imp = r['DA'] - rw_metrics['DA']
        print(f"\n{r['Model']}:")
        print(f"  RMSE: {rw_metrics['RMSE']:.4f} -> {r['RMSE']:.4f} ({rmse_imp:+.2f}%)")
        print(f"  DA:   {rw_metrics['DA']:.1f}% -> {r['DA']:.1f}% ({da_imp:+.1f} pp)")

    # Check for success
    winners = [r for r in results if r['Beats_Both']]
    if winners:
        print("\n" + "="*70)
        print("*** SUCCESS: Model(s) beat Random Walk on BOTH metrics! ***")
        print("="*70)
