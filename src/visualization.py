"""
Visualization module for thesis figures.
"""

import os
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 150


class ThesisVisualizer:
    def __init__(self, results_dict, output_dir='figures'):
        self.results = results_dict
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _get_processed_df(self):
        data_dict = self.results.get('data', {})
        processed = data_dict.get('processed')
        if isinstance(processed, pd.DataFrame) and not processed.empty:
            return processed

        parts = []
        for key in ['train_df', 'val_df', 'test_df']:
            df = data_dict.get(key)
            if isinstance(df, pd.DataFrame) and not df.empty:
                parts.append(df)
        if parts:
            return pd.concat(parts).sort_index()
        return None

    def _get_test_df(self):
        data_dict = self.results.get('data', {})
        test_df = data_dict.get('test_df')
        if isinstance(test_df, pd.DataFrame) and not test_df.empty:
            return test_df
        return None

    def _get_metrics_df(self):
        metrics = self.results.get('metrics')
        if isinstance(metrics, pd.DataFrame) and not metrics.empty:
            return metrics.copy()
        return pd.DataFrame()

    def _get_regime_df(self):
        regime = self.results.get('regime_analysis')
        if isinstance(regime, pd.DataFrame) and not regime.empty:
            return regime.copy()
        return pd.DataFrame()

    def _get_analysis_df(self, key):
        analysis = self.results.get('analysis', {})
        df = analysis.get(key)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df.copy()
        return pd.DataFrame()

    def _hybrid_key(self, predictions):
        for key in ['Hybrid ARIMA-LSTM', 'Hybrid']:
            if key in predictions:
                return key
        return None

    def plot_data_overview(self):
        data = self._get_processed_df()
        if data is None:
            print("  Warning: No processed data available for fig1")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax1 = axes[0, 0]
        ax1.plot(data.index, data['usdngn'], color='#1f77b4', linewidth=0.8)
        ax1.set_title('USD-NGN Exchange Rate (1995-2024)')
        ax1.set_ylabel('NGN per USD')
        ax1.set_xlabel('Date')

        for start, end, color in [
            ('2014-07-01', '2016-12-31', '#FFB6C1'),
            ('2023-06-01', '2024-12-31', '#FFD700'),
        ]:
            ax1.axvspan(pd.to_datetime(start), pd.to_datetime(end), alpha=0.25, color=color)

        ax2 = axes[0, 1]
        ax2.plot(data.index, data['brent_oil'], color='#d62728', linewidth=0.8)
        ax2.set_title('Brent Crude Oil Price')
        ax2.set_ylabel('USD per Barrel')
        ax2.set_xlabel('Date')

        ax3 = axes[1, 0]
        ax3.step(data.index, data['mpr'], where='post', color='#2ca02c', linewidth=1)
        ax3.set_title('Monetary Policy Rate')
        ax3.set_ylabel('Rate (%)')
        ax3.set_xlabel('Date')

        ax4 = axes[1, 1]
        ax4.plot(data.index, data['cpi'], color='#9467bd', linewidth=0.8)
        ax4.set_title('Consumer Price Index / Inflation')
        ax4.set_ylabel('Inflation (%)')
        ax4.set_xlabel('Date')

        for ax in axes.flat:
            ax.xaxis.set_major_locator(mdates.YearLocator(5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/fig1_data_overview.png', bbox_inches='tight')
        plt.close()
        print("  Saved: fig1_data_overview.png")

    def plot_transfer_entropy_results(self):
        te_results = self._get_analysis_df('te_results')
        weight_df = self._get_analysis_df('feature_weights')
        if te_results.empty:
            print("  Warning: No transfer entropy results available for fig2")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        source_col = 'source' if 'source' in te_results.columns else 'feature'
        te_col = 'te_forward' if 'te_forward' in te_results.columns else 'te_score'
        p_col = 'p_forward' if 'p_forward' in te_results.columns else 'p_value'
        te_plot = te_results.sort_values(te_col, ascending=True)

        ax1 = axes[0]
        bars = ax1.barh(te_plot[source_col], te_plot[te_col], color='#1f77b4')
        ax1.set_title('Transfer Entropy Scores')
        ax1.set_xlabel('Transfer Entropy')
        for bar, p in zip(bars, te_plot[p_col].values):
            marker = ''
            if p < 0.001:
                marker = '***'
            elif p < 0.01:
                marker = '**'
            elif p < 0.05:
                marker = '*'
            if marker:
                ax1.annotate(marker, (bar.get_width(), bar.get_y() + bar.get_height() / 2), va='center')

        ax2 = axes[1]
        if not weight_df.empty:
            var_col = 'variable' if 'variable' in weight_df.columns else 'feature'
            weight_plot = weight_df.sort_values('weight', ascending=True)
            ax2.barh(weight_plot[var_col], weight_plot['weight'], color='#ff7f0e')
            ax2.set_title('Feature Weights')
            ax2.set_xlabel('Weight')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/fig2_transfer_entropy.png', bbox_inches='tight')
        plt.close()
        print("  Saved: fig2_transfer_entropy.png")

    def plot_model_predictions(self):
        test_df = self._get_test_df()
        predictions = self.results.get('predictions', {})
        if test_df is None or not predictions:
            print("  Warning: No prediction data available for fig3")
            return

        y_true = test_df['usdngn'].values
        dates = test_df.index
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        focus_models = ['Random Walk', 'ARIMA', 'ARIMAX', 'Hybrid ARIMA-LSTM', 'Mean Reversion + Streak']
        ax1 = axes[0]
        ax1.plot(dates, y_true, color='black', linewidth=1.4, label='Actual')
        for name in focus_models:
            if name in predictions:
                pred = np.array(predictions[name]).flatten()
                min_len = min(len(dates), len(pred))
                ax1.plot(dates[:min_len], pred[:min_len], linewidth=1, linestyle='--', label=name)
        ax1.set_title('Actual vs Predicted USD-NGN on Test Set')
        ax1.set_ylabel('Exchange Rate')
        ax1.set_xlabel('Date')
        ax1.legend(loc='upper left', ncol=2)
        ax1.axvline(pd.to_datetime('2023-06-14'), color='red', linestyle=':', linewidth=1.5)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        ax2 = axes[1]
        hybrid_key = self._hybrid_key(predictions)
        if hybrid_key is not None:
            hybrid_pred = np.array(predictions[hybrid_key]).flatten()
            min_len = min(len(y_true), len(hybrid_pred))
            errors = y_true[:min_len] - hybrid_pred[:min_len]
            ax2.plot(dates[:min_len], errors, color='#d62728', linewidth=0.8)
            ax2.axhline(0, color='black', linewidth=0.6)
            ax2.set_title(f'{hybrid_key} Prediction Errors')
            ax2.set_ylabel('Actual - Predicted')
            ax2.set_xlabel('Date')
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/fig3_predictions.png', bbox_inches='tight')
        plt.close()
        print("  Saved: fig3_predictions.png")

    def plot_model_comparison(self):
        metrics_df = self._get_metrics_df()
        if metrics_df.empty:
            print("  Warning: No metrics available for fig4")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        metrics_df = metrics_df[['Model', 'RMSE', 'MAE', 'MAPE', 'DA']].copy()
        metric_names = ['RMSE', 'MAE', 'MAPE', 'DA']

        for idx, metric in enumerate(metric_names):
            ax = axes[idx // 2, idx % 2]
            values = metrics_df[metric].values
            labels = metrics_df['Model'].values
            bars = ax.bar(labels, values, color='#1f77b4')
            if metric == 'DA':
                best_idx = int(np.argmax(values))
                ax.set_ylabel('Percent')
            else:
                best_idx = int(np.argmin(values))
                ax.set_ylabel(metric)
            bars[best_idx].set_color('#2ca02c')
            ax.set_title(f'{metric} Comparison')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/fig4_model_comparison.png', bbox_inches='tight')
        plt.close()
        print("  Saved: fig4_model_comparison.png")

    def plot_regime_analysis(self):
        regime_df = self._get_regime_df()
        if regime_df.empty:
            print("  Warning: No regime analysis available for fig5")
            return

        focus_model = self._hybrid_key(self.results.get('predictions', {})) or regime_df['Model'].iloc[0]
        regime_plot = regime_df[regime_df['Model'] == focus_model].copy()
        if regime_plot.empty:
            regime_plot = regime_df.copy()
        regime_plot = regime_plot.set_index('Regime')
        regime_plot.index = [str(idx[0]) if isinstance(idx, tuple) else str(idx) for idx in regime_plot.index]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for idx, metric in enumerate(['RMSE', 'MAE', 'DA']):
            ax = axes[idx]
            ax.bar(regime_plot.index, regime_plot[metric].values, color='#1f77b4')
            ax.set_title(f'{focus_model}: {metric} by Regime')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/fig5_regime_analysis.png', bbox_inches='tight')
        plt.close()
        print("  Saved: fig5_regime_analysis.png")

    def plot_feature_importance(self):
        shap_results = self.results.get('explainability', {})
        importance = shap_results.get('importance')
        if not isinstance(importance, pd.DataFrame) or importance.empty:
            print("  Warning: No SHAP results available for fig6")
            return

        importance = importance.sort_values('importance', ascending=True)
        values = importance['importance_pct'].values if 'importance_pct' in importance.columns else importance['importance'].values

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(importance['feature'].values, values, color=plt.cm.Blues(np.linspace(0.3, 0.9, len(importance))))
        ax.set_title('SHAP-based Feature Importance')
        ax.set_xlabel('Importance (%)' if 'importance_pct' in importance.columns else 'Importance')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/fig6_feature_importance.png', bbox_inches='tight')
        plt.close()
        print("  Saved: fig6_feature_importance.png")

    def plot_directional_accuracy(self):
        test_df = self._get_test_df()
        predictions = self.results.get('predictions', {})
        hybrid_key = self._hybrid_key(predictions)
        if test_df is None or hybrid_key is None:
            print("  Warning: No hybrid prediction data available for fig7")
            return

        y_true = test_df['usdngn'].values
        y_pred = np.array(predictions[hybrid_key]).flatten()
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        dates = test_df.index[:min_len]
        prev_values = np.concatenate([[self.results['data']['train'][1][-1]], y_true[:-1]])
        correct = ((y_true > prev_values).astype(int) == (y_pred > prev_values).astype(int)).astype(float)
        rolling_da = pd.Series(correct).rolling(window=60).mean() * 100

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(dates, rolling_da.values, color='#2ca02c', linewidth=1.4)
        ax.axhline(50, color='red', linestyle='--', linewidth=1, label='Random Walk level')
        ax.axhline(65, color='orange', linestyle='--', linewidth=1, label='65% reference')
        ax.set_title(f'Rolling Directional Accuracy: {hybrid_key}')
        ax.set_ylabel('Directional Accuracy (%)')
        ax.set_xlabel('Date')
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/fig7_directional_accuracy.png', bbox_inches='tight')
        plt.close()
        print("  Saved: fig7_directional_accuracy.png")

    def generate_all_figures(self):
        print("\n" + "=" * 60)
        print("GENERATING THESIS FIGURES")
        print("=" * 60)
        self.plot_data_overview()
        self.plot_transfer_entropy_results()
        self.plot_model_predictions()
        self.plot_model_comparison()
        self.plot_regime_analysis()
        self.plot_feature_importance()
        self.plot_directional_accuracy()
        print("\n" + "=" * 60)
        print(f"All figures saved to: {self.output_dir}")
        print("=" * 60)


def generate_visualizations(results_dict, output_dir='figures'):
    ThesisVisualizer(results_dict, output_dir).generate_all_figures()


if __name__ == "__main__":
    from run_pipeline import run_pipeline

    results = run_pipeline(verbose=False, runtime_profile='fast', benchmark_mode='fast_benchmarks')
    generate_visualizations(results, output_dir='figures')
