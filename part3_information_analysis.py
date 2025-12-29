"""
PART 3: INFORMATION FLOW ANALYSIS (TRANSFER ENTROPY)
PhD Thesis: USD-NGN Exchange Rate Forecasting
Author: Oche Emmanuel Ike (242220011)
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

class TransferEntropyAnalyzer:
    def __init__(self, n_bins=6, k=1, n_bootstrap=500, random_state=42):
        self.n_bins = n_bins
        self.k = k
        self.n_bootstrap = n_bootstrap
        np.random.seed(random_state)
        
    def discretize(self, series):
        series = np.array(series).flatten()
        series = series[~np.isnan(series)]
        try:
            discretized = pd.qcut(series, q=self.n_bins, labels=False, duplicates='drop')
        except:
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
        return max(0, te)
    
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
        print(f"  TE({source_name}→{target_name}): {te_forward:.4f} bits, p={p_forward:.4f} {sig}")
        return {'source': source_name, 'target': target_name, 'te_forward': te_forward, 
                'p_forward': p_forward, 'te_reverse': te_reverse, 'p_reverse': p_reverse}

class FeatureWeightComputer:
    def __init__(self, alpha=0.6):
        self.alpha = alpha
    
    def compute_weights(self, te_scores, mi_scores, feature_names):
        te_scores = np.array(te_scores)
        mi_scores = np.array(mi_scores)
        te_norm = (te_scores - te_scores.min()) / (te_scores.max() - te_scores.min() + 1e-10)
        mi_norm = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min() + 1e-10)
        weights = self.alpha * te_norm + (1 - self.alpha) * mi_norm
        return pd.DataFrame({'variable': feature_names, 'te_score': te_scores, 
                           'mi_score': mi_scores, 'weight': weights}).sort_values('weight', ascending=False)

def run_information_analysis(df, target='usdngn', source_vars=None):
    print("\n" + "=" * 70)
    print("INFORMATION FLOW ANALYSIS")
    print("=" * 70)
    te_analyzer = TransferEntropyAnalyzer()
    if source_vars is None:
        source_vars = ['brent_oil', 'mpr', 'cpi']
    available_vars = [v for v in source_vars if v in df.columns]
    te_results = []
    for var in available_vars:
        result = te_analyzer.analyze_pair(df[var].values, df[target].values, var, target)
        te_results.append(result)
    te_df = pd.DataFrame(te_results)
    mi_features = [col for col in df.columns if col != target and df[col].dtype in ['float64', 'int64']][:10]
    mi_scores = mutual_info_regression(df[mi_features].values, df[target].values, random_state=42)
    mi_results = pd.DataFrame({'feature': mi_features, 'mi_score': mi_scores}).sort_values('mi_score', ascending=False)
    weight_computer = FeatureWeightComputer(alpha=0.6)
    te_scores = {r['source']: r['te_forward'] for r in te_results}
    mi_scores_dict = {row['feature']: row['mi_score'] for _, row in mi_results.iterrows()}
    common_vars = [v for v in available_vars if v in te_scores and v in mi_scores_dict]
    if common_vars:
        te_arr = np.array([te_scores[v] for v in common_vars])
        mi_arr = np.array([mi_scores_dict[v] for v in common_vars])
        weight_df = weight_computer.compute_weights(te_arr, mi_arr, common_vars)
    else:
        weight_df = pd.DataFrame()
    return {'te_results': te_df, 'mi_results': mi_results, 'feature_weights': weight_df}

if __name__ == "__main__":
    from part1_data_collection import DataCollector
    from part2_preprocessing import DataPreprocessor
    collector = DataCollector()
    raw_df = collector.collect_all_data()
    preprocessor = DataPreprocessor(raw_df)
    processed_df, _ = preprocessor.preprocess()
    info_results = run_information_analysis(processed_df)
