"""
================================================================================
VISUALIZATION MODULE
================================================================================
PhD Thesis: Forecasting USD-NGN Exchange Rate Using Information Theory,
            Hybrid Machine Learning and Explainable AI

Author: Oche Emmanuel Ike (Student ID: 242220011)
Institution: International Institute for Financial Engineering (IIFE)

This module generates publication-quality visualizations for the thesis.
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality defaults
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 150


class ThesisVisualizer:
    """
    Generate all thesis visualizations.
    """
    
    def __init__(self, results_dict, output_dir='/home/claude/usdngn_forecasting/figures'):
        """
        Initialize visualizer.
        
        Parameters:
        -----------
        results_dict : dict
            Complete results from pipeline
        output_dir : str
            Directory to save figures
        """
        self.results = results_dict
        self.output_dir = output_dir
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_data_overview(self):
        """
        Figure 1: Historical USD-NGN and Brent Oil Prices.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        data = self.results['data']['processed']
        
        # Plot 1: USD-NGN Exchange Rate
        ax1 = axes[0, 0]
        ax1.plot(data.index, data['usdngn'], color='#1f77b4', linewidth=0.8)
        ax1.set_title('USD-NGN Exchange Rate (1995-2025)')
        ax1.set_ylabel('NGN per 1 USD')
        ax1.set_xlabel('Date')
        
        # Add regime shading
        regimes = {
            'Pre-Crisis': ('2010-01-01', '2014-06-30', '#90EE90', 0.2),
            'Oil Crisis': ('2014-07-01', '2016-12-31', '#FFB6C1', 0.3),
            'Depegging': ('2023-06-01', '2025-12-31', '#FFD700', 0.3)
        }
        
        for name, (start, end, color, alpha) in regimes.items():
            try:
                ax1.axvspan(pd.to_datetime(start), pd.to_datetime(end), 
                           alpha=alpha, color=color, label=name)
            except Exception:
                pass
        
        # Plot 2: Brent Oil Prices
        ax2 = axes[0, 1]
        ax2.plot(data.index, data['brent_oil'], color='#d62728', linewidth=0.8)
        ax2.set_title('Brent Crude Oil Price (1995-2025)')
        ax2.set_ylabel('USD per Barrel')
        ax2.set_xlabel('Date')
        
        # Plot 3: MPR
        ax3 = axes[1, 0]
        ax3.step(data.index, data['mpr'], where='post', color='#2ca02c', linewidth=1)
        ax3.set_title('Monetary Policy Rate')
        ax3.set_ylabel('Rate (%)')
        ax3.set_xlabel('Date')
        
        # Plot 4: CPI/Inflation
        ax4 = axes[1, 1]
        ax4.plot(data.index, data['cpi'], color='#9467bd', linewidth=0.8)
        ax4.set_title('Consumer Price Index / Inflation Rate')
        ax4.set_ylabel('Inflation (%)')
        ax4.set_xlabel('Date')
        
        for ax in axes.flat:
            ax.xaxis.set_major_locator(mdates.YearLocator(5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/fig1_data_overview.png', bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: fig1_data_overview.png")
        
    def plot_transfer_entropy_results(self):
        """
        Figure 2: Transfer Entropy Analysis Results.
        """
        te_results = self.results['analysis']['te_results']
        
        if te_results is None or len(te_results) == 0:
            print("  ⚠ No TE results available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Bar chart of TE values
        ax1 = axes[0]
        
        sources = te_results['source'].values
        te_forward = te_results['te_forward'].values
        te_reverse = te_results['te_reverse'].values if 'te_reverse' in te_results.columns else np.zeros(len(sources))
        
        x = np.arange(len(sources))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, te_forward, width, label='X → USD-NGN', color='#1f77b4')
        bars2 = ax1.bar(x + width/2, te_reverse, width, label='USD-NGN → X', color='#ff7f0e')
        
        ax1.set_ylabel('Transfer Entropy (bits)')
        ax1.set_title('Directional Information Flow')
        ax1.set_xticks(x)
        ax1.set_xticklabels(sources)
        ax1.legend()
        
        # Add significance markers
        for i, (bar, p) in enumerate(zip(bars1, te_results['p_forward'].values)):
            if p < 0.001:
                ax1.annotate('***', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=12)
            elif p < 0.01:
                ax1.annotate('**', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=12)
            elif p < 0.05:
                ax1.annotate('*', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=12)
        
        # Plot 2: Feature weights
        ax2 = axes[1]
        
        weights = self.results['analysis']['feature_weights']
        if weights is not None and len(weights) > 0:
            vars_ = weights['variable'].values
            w = weights['weight'].values
            
            colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(vars_)))
            bars = ax2.barh(vars_, w, color=colors)
            ax2.set_xlabel('Feature Weight (Eq. 3.8)')
            ax2.set_title('TE-Derived Feature Weights (α=0.6)')
            
            for bar, val in zip(bars, w):
                ax2.annotate(f'{val:.3f}', (bar.get_width(), bar.get_y() + bar.get_height()/2),
                           ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/fig2_transfer_entropy.png', bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: fig2_transfer_entropy.png")
        
    def plot_model_predictions(self):
        """
        Figure 3: Model Predictions vs Actual (Test Set).
        """
        test_data = self.results['data']['test']
        predictions = self.results['predictions']
        
        if test_data is None or not predictions:
            print("  ⚠ No prediction data available")
            return
        
        y_true = test_data['usdngn'].values
        dates = test_data.index
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Full test set comparison
        ax1 = axes[0]
        
        ax1.plot(dates, y_true, 'k-', linewidth=1.5, label='Actual', alpha=0.8)
        
        colors = {'Random Walk': '#1f77b4', 'ARIMA': '#ff7f0e', 
                  'LSTM': '#2ca02c', 'Hybrid': '#d62728'}
        
        for model_name, pred in predictions.items():
            min_len = min(len(dates), len(pred))
            ax1.plot(dates[:min_len], pred[:min_len], 
                    linestyle='--', linewidth=1, alpha=0.7,
                    color=colors.get(model_name, 'gray'),
                    label=model_name)
        
        ax1.set_title('Model Predictions vs Actual (Test Set: 2020-2025)')
        ax1.set_ylabel('USD-NGN Exchange Rate')
        ax1.set_xlabel('Date')
        ax1.legend(loc='upper left')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Add depegging marker
        ax1.axvline(pd.to_datetime('2023-06-14'), color='red', 
                   linestyle=':', linewidth=2, alpha=0.7)
        ax1.annotate('CBN Depegging\n(June 2023)', 
                    xy=(pd.to_datetime('2023-06-14'), ax1.get_ylim()[1] * 0.9),
                    fontsize=9, ha='center')
        
        # Plot 2: Prediction errors (Hybrid model)
        ax2 = axes[1]
        
        if 'Hybrid' in predictions:
            hybrid_pred = predictions['Hybrid']
            min_len = min(len(y_true), len(hybrid_pred))
            errors = y_true[:min_len] - hybrid_pred[:min_len]
            
            ax2.plot(dates[:min_len], errors, color='#d62728', linewidth=0.8)
            ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
            ax2.fill_between(dates[:min_len], errors, 0, 
                            where=(errors >= 0), alpha=0.3, color='green', label='Underprediction')
            ax2.fill_between(dates[:min_len], errors, 0, 
                            where=(errors < 0), alpha=0.3, color='red', label='Overprediction')
            
            ax2.set_title('Hybrid Model Prediction Errors')
            ax2.set_ylabel('Error (Actual - Predicted)')
            ax2.set_xlabel('Date')
            ax2.legend()
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/fig3_predictions.png', bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: fig3_predictions.png")
        
    def plot_model_comparison(self):
        """
        Figure 4: Model Performance Comparison.
        """
        metrics = self.results['evaluation'].get('metrics', {})
        
        if not metrics:
            print("  ⚠ No metrics available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        models = list(metrics.keys())
        
        # Metrics to plot
        metric_names = ['RMSE', 'MAE', 'MAPE', 'DA']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for idx, metric in enumerate(metric_names):
            ax = axes[idx // 2, idx % 2]
            
            values = [metrics[m][metric] for m in models]
            
            bars = ax.bar(models, values, color=colors)
            ax.set_title(f'{metric} Comparison')
            
            if metric in ['RMSE', 'MAE']:
                ax.set_ylabel(f'{metric} (NGN)')
            elif metric in ['MAPE', 'DA']:
                ax.set_ylabel(f'{metric} (%)')
            
            # Highlight best model
            if metric == 'DA':
                best_idx = np.argmax(values)
            else:
                best_idx = np.argmin(values)
            bars[best_idx].set_color('#2ca02c')
            bars[best_idx].set_edgecolor('black')
            bars[best_idx].set_linewidth(2)
            
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{val:.2f}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/fig4_model_comparison.png', bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: fig4_model_comparison.png")
        
    def plot_regime_analysis(self):
        """
        Figure 5: Regime-Specific Performance.
        """
        regime_results = self.results['evaluation'].get('regime_analysis')
        
        if regime_results is None or len(regime_results) == 0:
            print("  ⚠ No regime analysis available")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        regimes = regime_results.index.tolist()
        
        # Metrics to plot
        metrics = ['RMSE', 'MAE', 'DA']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            values = [regime_results.loc[r][metric] for r in regimes]
            
            bars = ax.bar(regimes, values, color=colors[idx])
            ax.set_title(f'{metric} by Economic Regime')
            ax.set_xlabel('Regime')
            
            if metric in ['RMSE', 'MAE']:
                ax.set_ylabel(f'{metric} (NGN)')
            else:
                ax.set_ylabel(f'{metric} (%)')
            
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Add value labels
            for bar, val in zip(bars, values):
                ax.annotate(f'{val:.1f}',
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/fig5_regime_analysis.png', bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: fig5_regime_analysis.png")
        
    def plot_feature_importance(self):
        """
        Figure 6: Feature Importance (SHAP Analysis).
        """
        shap_results = self.results.get('explainability')
        
        if shap_results is None or 'importance' not in shap_results:
            print("  ⚠ No SHAP results available")
            return
        
        importance = shap_results['importance']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort by importance
        importance = importance.sort_values('importance', ascending=True)
        
        # Use importance_pct if available, otherwise importance
        if 'importance_pct' in importance.columns:
            values = importance['importance_pct'].values
            xlabel = 'Feature Importance (%)'
        else:
            values = importance['importance'].values
            xlabel = 'Feature Importance'
        
        features = importance['feature'].values
        
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(features)))
        
        bars = ax.barh(features, values, color=colors)
        ax.set_xlabel(xlabel)
        ax.set_title('Feature Importance (SHAP-based Analysis)')
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.1f}%',
                       xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                       ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/fig6_feature_importance.png', bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: fig6_feature_importance.png")
        
    def plot_directional_accuracy(self):
        """
        Figure 7: Directional Accuracy Over Time.
        """
        test_data = self.results['data']['test']
        predictions = self.results['predictions']
        
        if 'Hybrid' not in predictions:
            print("  ⚠ Hybrid predictions not available")
            return
        
        y_true = test_data['usdngn'].values
        hybrid_pred = predictions['Hybrid']
        dates = test_data.index
        
        min_len = min(len(y_true), len(hybrid_pred))
        y_true = y_true[:min_len]
        hybrid_pred = hybrid_pred[:min_len]
        dates = dates[:min_len]
        
        # Compute rolling directional accuracy
        window = 60
        
        actual_dir = np.sign(np.diff(y_true))
        pred_dir = np.sign(np.diff(hybrid_pred))
        
        correct = (actual_dir == pred_dir).astype(float)
        rolling_da = pd.Series(correct).rolling(window=window).mean() * 100
        
        fig, ax = plt.subplots(figsize=(12, 5))
        
        ax.plot(dates[1:], rolling_da.values, color='#2ca02c', linewidth=1.5)
        ax.axhline(50, color='red', linestyle='--', linewidth=1, label='Random (50%)')
        ax.axhline(65, color='orange', linestyle='--', linewidth=1, label='Target (65%)')
        
        ax.set_title(f'Rolling Directional Accuracy ({window}-day window)')
        ax.set_ylabel('Directional Accuracy (%)')
        ax.set_xlabel('Date')
        ax.set_ylim(30, 100)
        ax.legend()
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Add depegging marker
        ax.axvline(pd.to_datetime('2023-06-14'), color='red', 
                  linestyle=':', linewidth=2, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/fig7_directional_accuracy.png', bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: fig7_directional_accuracy.png")
        
    def generate_all_figures(self):
        """Generate all thesis figures."""
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


def generate_visualizations(results_dict, output_dir='/home/claude/usdngn_forecasting/figures'):
    """
    Generate all visualizations from pipeline results.
    
    Parameters:
    -----------
    results_dict : dict
        Complete results from pipeline
    output_dir : str
        Output directory for figures
    """
    visualizer = ThesisVisualizer(results_dict, output_dir)
    visualizer.generate_all_figures()


# ============================================================
# STANDALONE EXECUTION
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("GENERATING THESIS VISUALIZATIONS")
    print("=" * 70)
    
    # Run pipeline first to get results
    from part6_pipeline import USDNGNForecastingPipeline
    
    pipeline = USDNGNForecastingPipeline()
    results = pipeline.run_complete_pipeline(verbose=False)
    
    # Generate visualizations
    generate_visualizations(results)
    
    print("\n✓ Visualization generation completed!")
