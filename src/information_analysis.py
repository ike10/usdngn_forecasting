"""
PART 3: INFORMATION FLOW ANALYSIS (TRANSFER ENTROPY)
PhD Thesis: USD-NGN Exchange Rate Forecasting
Author: Oche Emmanuel Ike (242220011)
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

warnings.filterwarnings('ignore')


class TransferEntropyAnalyzer:
    def __init__(self, n_bins=6, k=1, n_bootstrap=40, random_state=42):
        self.n_bins = n_bins
        self.k = k
        self.n_bootstrap = n_bootstrap
        np.random.seed(random_state)

    def discretize(self, series):
        series = np.array(series).flatten()
        series = series[~np.isnan(series)]
        try:
            discretized = pd.qcut(series, q=self.n_bins, labels=False, duplicates='drop')
        except (ValueError, IndexError):
            discretized = pd.cut(series, bins=self.n_bins, labels=False)
        return np.nan_to_num(np.array(discretized), nan=0).astype(int)

    def compute_transfer_entropy(self, source, target, k=None):
        if k is None:
            k = self.k
        source = np.array(source).flatten()
        target = np.array(target).flatten()
        min_len = min(len(source), len(target))
        source, target = source[:min_len], target[:min_len]
        if min_len < k + 2:
            return 0.0

        source_d = self.discretize(source)
        target_d = self.discretize(target)
        n_states = self.n_bins

        y_next = target_d[k + 1:]
        y_curr = target_d[k:-1]
        x_curr = source_d[k:-1]
        n_samples = len(y_next)
        if n_samples < 10:
            return 0.0

        joint_idx = y_next * n_states**2 + y_curr * n_states + x_curr
        marginal_yy_idx = y_next * n_states + y_curr
        y_idx = y_curr
        yx_idx = y_curr * n_states + x_curr

        joint_counts = np.bincount(joint_idx.astype(int), minlength=n_states**3)
        p_joint = (joint_counts + 1e-10) / (n_samples + 1e-10 * n_states**3)
        marginal_yy_counts = np.bincount(marginal_yy_idx.astype(int), minlength=n_states**2)
        p_marginal_yy = (marginal_yy_counts + 1e-10) / (n_samples + 1e-10 * n_states**2)
        y_counts = np.bincount(y_idx.astype(int), minlength=n_states)
        p_y = (y_counts + 1e-10) / (n_samples + 1e-10 * n_states)
        yx_counts = np.bincount(yx_idx.astype(int), minlength=n_states**2)
        p_yx = (yx_counts + 1e-10) / (n_samples + 1e-10 * n_states**2)

        te = 0.0
        for y_next_val in range(n_states):
            for y_curr_val in range(n_states):
                for x_curr_val in range(n_states):
                    joint_state = y_next_val * n_states**2 + y_curr_val * n_states + x_curr_val
                    marg_yy_state = y_next_val * n_states + y_curr_val
                    yx_state = y_curr_val * n_states + x_curr_val

                    p_yyx = p_joint[joint_state]
                    p_yy = p_marginal_yy[marg_yy_state]
                    p_y_val = p_y[y_curr_val]
                    p_yx_val = p_yx[yx_state]

                    if p_yyx > 1e-10 and p_y_val > 1e-10 and p_yx_val > 1e-10 and p_yy > 1e-10:
                        ratio = (p_yyx * p_y_val) / (p_yx_val * p_yy)
                        if ratio > 0:
                            te += p_yyx * np.log2(ratio)

        return max(0.0, te)

    def compute_significance(self, source, target, te_observed):
        te_null = np.zeros(self.n_bootstrap)
        for i in range(self.n_bootstrap):
            shuffled_source = np.random.permutation(source)
            te_null[i] = self.compute_transfer_entropy(shuffled_source, target)
        p_value = np.mean(te_null >= te_observed)
        return p_value, te_null

    def analyze_pair(self, source, target, source_name, target_name):
        te_forward = self.compute_transfer_entropy(source, target)
        p_forward, _ = self.compute_significance(source, target, te_forward)
        te_reverse = self.compute_transfer_entropy(target, source)
        p_reverse, _ = self.compute_significance(target, source, te_reverse)
        sig = "***" if p_forward < 0.001 else "**" if p_forward < 0.01 else "*" if p_forward < 0.05 else "ns"
        print(f"  TE({source_name}->{target_name}): {te_forward:.4f} bits, p={p_forward:.4f} {sig}")
        return {
            'source': source_name,
            'target': target_name,
            'te_forward': te_forward,
            'p_forward': p_forward,
            'te_reverse': te_reverse,
            'p_reverse': p_reverse,
        }


class FeatureWeightComputer:
    def __init__(self, alpha=0.6, min_weight=0.35, max_weight=1.35):
        self.alpha = alpha
        self.min_weight = min_weight
        self.max_weight = max_weight

    @staticmethod
    def _normalize(scores):
        scores = np.array(scores, dtype=float)
        if len(scores) == 0:
            return scores
        span = scores.max() - scores.min()
        if span <= 1e-10:
            return np.ones_like(scores)
        return (scores - scores.min()) / span

    def compute_weights(self, te_scores, mi_scores, feature_names, p_values=None):
        te_norm = self._normalize(te_scores)
        mi_norm = self._normalize(mi_scores)
        combined = self.alpha * te_norm + (1 - self.alpha) * mi_norm

        if p_values is None:
            significance_bonus = np.ones_like(combined)
        else:
            p_values = np.array(p_values, dtype=float)
            significance_bonus = np.clip(1.0 - p_values, 0.25, 1.0)

        weights = combined * significance_bonus
        weights = self.min_weight + (self.max_weight - self.min_weight) * self._normalize(weights)

        return pd.DataFrame({
            'feature': feature_names,
            'te_score': np.array(te_scores, dtype=float),
            'mi_score': np.array(mi_scores, dtype=float),
            'p_value': np.array(p_values, dtype=float) if p_values is not None else np.nan,
            'weight': weights,
        }).sort_values('weight', ascending=False)


def _aligned_arrays(df, feature, target, lag=1):
    subset = df[[feature, target]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(subset) <= lag + 2:
        return np.array([]), np.array([])

    x = subset[feature].values[:-lag]
    y = subset[target].values[lag:]
    min_len = min(len(x), len(y))
    return x[:min_len], y[:min_len]


def run_information_analysis(
    df,
    target='usdngn',
    source_vars=None,
    lag=1,
    alpha=0.6,
    n_bootstrap=40,
):
    print("\n" + "=" * 70)
    print("INFORMATION FLOW ANALYSIS")
    print("=" * 70)

    te_analyzer = TransferEntropyAnalyzer(k=lag, n_bootstrap=n_bootstrap)

    if source_vars is None:
        source_vars = [col for col in df.columns if col != target]

    candidate_vars = [
        col for col in source_vars
        if col in df.columns and col != target and pd.api.types.is_numeric_dtype(df[col])
    ]

    feature_rows = []
    mi_features = []
    mi_targets = []

    for feature in candidate_vars:
        x, y = _aligned_arrays(df, feature, target, lag=lag)
        if len(x) < 50:
            continue

        te_score = te_analyzer.compute_transfer_entropy(x, y, k=lag)
        p_value, _ = te_analyzer.compute_significance(x, y, te_score)
        mi_score = mutual_info_regression(x.reshape(-1, 1), y, random_state=42)[0]

        feature_rows.append({
            'feature': feature,
            'te_score': te_score,
            'mi_score': mi_score,
            'p_value': p_value,
        })
        mi_features.append(feature)
        mi_targets.append(mi_score)

    feature_scores = pd.DataFrame(feature_rows)
    if feature_scores.empty:
        empty = pd.DataFrame(columns=['feature', 'te_score', 'mi_score', 'p_value', 'weight'])
        return {
            'te_results': empty,
            'mi_results': pd.DataFrame(columns=['feature', 'mi_score']),
            'feature_weights': empty,
            'weight_map': {},
        }

    mi_results = pd.DataFrame({'feature': mi_features, 'mi_score': mi_targets}).sort_values(
        'mi_score', ascending=False
    )

    weight_computer = FeatureWeightComputer(alpha=alpha)
    weight_df = weight_computer.compute_weights(
        feature_scores['te_score'].values,
        feature_scores['mi_score'].values,
        feature_scores['feature'].tolist(),
        p_values=feature_scores['p_value'].values,
    )

    return {
        'te_results': feature_scores.sort_values('te_score', ascending=False),
        'mi_results': mi_results,
        'feature_weights': weight_df,
        'weight_map': dict(zip(weight_df['feature'], weight_df['weight'])),
    }


if __name__ == "__main__":
    try:
        from .data_collection import DataCollector
        from .preprocessing import DataPreprocessor
    except ImportError:
        from data_collection import DataCollector
        from preprocessing import DataPreprocessor

    collector = DataCollector()
    raw_df = collector.collect_all_data()
    preprocessor = DataPreprocessor(raw_df)
    processed_df, _ = preprocessor.preprocess()
    info_results = run_information_analysis(processed_df)
    print(info_results['feature_weights'].head(10))
