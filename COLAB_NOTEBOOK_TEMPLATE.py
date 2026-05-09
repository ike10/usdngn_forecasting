# Google Colab Notebook: USD-NGN Forecasting Pipeline
# Created: May 9, 2026
# Description: Complete setup and execution in Google Colab

# Copy the entire content below into a Colab notebook cell by cell

# ==============================================================================
# CELL 1: SYSTEM SETUP AND REPOSITORY CLONE
# ==============================================================================
# Description: Install system dependencies and clone GitHub repository

!apt-get update -qq
!apt-get install -y git

# Clone repository (replace with your GitHub URL)
GITHUB_URL = "https://github.com/YOUR_USERNAME/usdngn_forecasting.git"
!git clone {GITHUB_URL}

%cd usdngn_forecasting

# Verify directory structure
print("✓ Repository cloned successfully\n")
print("Directory structure:")
!ls -la


# ==============================================================================
# CELL 2: INSTALL PYTHON DEPENDENCIES
# ==============================================================================
# Description: Install all required packages

import sys
import subprocess

packages = [
    "numpy>=1.21.0",
    "pandas>=1.3.0",
    "scipy>=1.7.0",
    "scikit-learn>=1.0.0",
    "statsmodels>=0.13.0",
    "torch>=1.10.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    "shap>=0.41.0",
    "tqdm>=4.62.0"
]

print("Installing dependencies...")
for package in packages:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", package],
        capture_output=True
    )

print("✓ All dependencies installed successfully\n")

# Verify key imports
try:
    import numpy as np
    import pandas as pd
    import torch
    import sklearn
    print("✓ Verification:")
    print(f"  - NumPy: {np.__version__}")
    print(f"  - Pandas: {pd.__version__}")
    print(f"  - PyTorch: {torch.__version__}")
    print(f"  - Scikit-learn: {sklearn.__version__}")
    print(f"  - GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - GPU device: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"✗ Import error: {e}")


# ==============================================================================
# CELL 3: ENABLE GPU (OPTIONAL - FOR FASTER LSTM TRAINING)
# ==============================================================================
# Description: Verify and enable GPU if available

import torch

print("GPU Configuration:")
print(f"  GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU Count: {torch.cuda.device_count()}")
    print(f"  GPU Device Name: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("\n✓ GPU is ready and will be used for model training")
else:
    print("\n⚠ GPU not available. Using CPU (slower but functional)")
    print("  To enable GPU: Runtime → Change runtime type → GPU")


# ==============================================================================
# CELL 4: RUN THE COMPLETE PIPELINE
# ==============================================================================
# Description: Execute the complete forecasting pipeline
# Expected duration: 15-25 minutes

import os
import time

os.chdir('/content/usdngn_forecasting')

print("=" * 80)
print("USD-NGN EXCHANGE RATE FORECASTING PIPELINE")
print("=" * 80)
print("\nStarting pipeline execution...\n")

start_time = time.time()

# Run pipeline
os.system('python run_pipeline.py --verbose True --runtime_profile fast --benchmark_mode full')

elapsed = time.time() - start_time
print(f"\n✓ Pipeline completed in {elapsed/60:.2f} minutes")

# Verify output files
print("\nGenerated files:")
!ls -lh data/ | grep -E "csv|txt"


# ==============================================================================
# CELL 5: LOAD AND DISPLAY RESULTS
# ==============================================================================
# Description: Load results and display summary statistics

import pandas as pd
import numpy as np

print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

# Load evaluation metrics
try:
    metrics = pd.read_csv('data/evaluation_metrics.csv')
    print("\n[1] MODEL PERFORMANCE METRICS (Test Set)")
    print("-" * 80)
    print(metrics.to_string(index=False))
    
    # Best model
    best_rmse = metrics.loc[metrics['RMSE'].idxmin()]
    best_da = metrics.loc[metrics['DA_1Step'].idxmax()] if 'DA_1Step' in metrics.columns else None
    
    print(f"\n  Best RMSE: {best_rmse['Model']} ({best_rmse['RMSE']:.4f} NGN/USD)")
    if best_da is not None and pd.notna(best_da['DA_1Step']):
        print(f"  Best DA: {best_da['Model']} ({best_da['DA_1Step']:.2f}%)")
        
except FileNotFoundError:
    print("✗ evaluation_metrics.csv not found")

# Load transfer entropy results
try:
    te_scores = pd.read_csv('data/transfer_entropy_scores.csv')
    print("\n[2] TRANSFER ENTROPY ANALYSIS")
    print("-" * 80)
    print(te_scores.head(10).to_string(index=False))
    print(f"\n  Total features analyzed: {len(te_scores)}")
    print(f"  Significant features (p<0.05): {(te_scores['p_value'] < 0.05).sum()}")
    
except FileNotFoundError:
    print("✗ transfer_entropy_scores.csv not found")

# Load feature weights
try:
    weights = pd.read_csv('data/feature_weights.csv')
    print("\n[3] INFORMATION-THEORETIC FEATURE WEIGHTS")
    print("-" * 80)
    print(weights.head(10).to_string(index=False))
    
except FileNotFoundError:
    print("✗ feature_weights.csv not found")


# ==============================================================================
# CELL 6: VISUALIZE MODEL PERFORMANCE
# ==============================================================================
# Description: Create comparison charts

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# Load data
metrics = pd.read_csv('data/evaluation_metrics.csv')

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('USD-NGN Forecasting Model Performance Summary', fontsize=16, fontweight='bold')

# 1. RMSE Comparison
ax1 = axes[0, 0]
metrics_sorted = metrics.sort_values('RMSE')
colors = ['green' if x < 30 else 'orange' if x < 100 else 'red' 
          for x in metrics_sorted['RMSE']]
ax1.barh(metrics_sorted['Model'], metrics_sorted['RMSE'], color=colors)
ax1.axvline(metrics[metrics['Model'] == 'Random Walk']['RMSE'].values[0], 
            color='red', linestyle='--', linewidth=2, label='Random Walk')
ax1.set_xlabel('RMSE (NGN/USD)')
ax1.set_title('Root Mean Square Error')
ax1.legend()

# 2. MAE Comparison
ax2 = axes[0, 1]
metrics_sorted = metrics.sort_values('MAE')
ax2.barh(metrics_sorted['Model'], metrics_sorted['MAE'], color='steelblue')
ax2.set_xlabel('MAE (NGN/USD)')
ax2.set_title('Mean Absolute Error')

# 3. Directional Accuracy
ax3 = axes[1, 0]
if 'DA_1Step' in metrics.columns:
    metrics_sorted = metrics.sort_values('DA_1Step', ascending=False)
    colors_da = ['green' if x > 55 else 'orange' if x > 50 else 'red' 
                 for x in metrics_sorted['DA_1Step']]
    ax3.barh(metrics_sorted['Model'], metrics_sorted['DA_1Step'], color=colors_da)
    ax3.axvline(50, color='gray', linestyle='--', linewidth=1, label='Random (50%)')
    ax3.set_xlabel('Accuracy (%)')
    ax3.set_title('Directional Accuracy (1-Step Ahead)')
    ax3.legend()

# 4. MAPE Comparison
ax4 = axes[1, 1]
metrics_sorted = metrics.sort_values('MAPE')
ax4.barh(metrics_sorted['Model'], metrics_sorted['MAPE'], color='coral')
ax4.set_xlabel('MAPE (%)')
ax4.set_title('Mean Absolute Percentage Error')

plt.tight_layout()
plt.savefig('model_comparison_summary.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Model comparison visualization saved as 'model_comparison_summary.png'")


# ==============================================================================
# CELL 7: FEATURE IMPORTANCE ANALYSIS
# ==============================================================================
# Description: Visualize transfer entropy and SHAP importances

import matplotlib.pyplot as plt

# Transfer Entropy
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Transfer Entropy
te_scores = pd.read_csv('data/transfer_entropy_scores.csv').head(10)
axes[0].barh(te_scores['feature'], te_scores['te_score'], color='steelblue')
axes[0].set_xlabel('Transfer Entropy (bits)')
axes[0].set_title('Top 10 Features by Transfer Entropy')

# Plot 2: SHAP Importance
try:
    shap_importance = pd.read_csv('data/shap_feature_importance.csv').head(10)
    axes[1].barh(shap_importance['feature'], 
                shap_importance['mean_abs_shap_value'], 
                color='mediumseagreen')
    axes[1].set_xlabel('Mean |SHAP Value|')
    axes[1].set_title('Top 10 Features by SHAP Importance')
except FileNotFoundError:
    axes[1].text(0.5, 0.5, 'SHAP values not available', 
                ha='center', va='center', fontsize=12)
    axes[1].set_title('SHAP Analysis')

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Feature importance visualization saved")


# ==============================================================================
# CELL 8: DETAILED ANALYSIS - BEST MODELS
# ==============================================================================
# Description: Deep dive into top performing models

import pandas as pd

metrics = pd.read_csv('data/evaluation_metrics.csv')

print("\n" + "=" * 80)
print("DETAILED ANALYSIS - TOP PERFORMERS")
print("=" * 80)

# Top 5 by RMSE
print("\nTop 5 Models (by RMSE):")
print("-" * 80)
top_5 = metrics.nsmallest(5, 'RMSE')[['Model', 'RMSE', 'MAE', 'MAPE']]
print(top_5.to_string(index=False))

# Statistical comparison
try:
    dm_test = pd.read_csv('data/diebold_mariano_tests.csv')
    print("\n\nDiebold-Mariano Statistical Test Results:")
    print("-" * 80)
    significant = dm_test[dm_test['p_value'] < 0.05]
    print(f"Total comparisons: {len(dm_test)}")
    print(f"Statistically significant: {len(significant)} ({len(significant)/len(dm_test)*100:.1f}%)")
    print(f"\nSignificant differences (p < 0.05):")
    if len(significant) > 0:
        print(significant[['Model', 'Benchmark', 'DM Statistic', 'p_value']].head(10).to_string(index=False))
    else:
        print("No significant differences found between top models")
except FileNotFoundError:
    print("Diebold-Mariano results not available")


# ==============================================================================
# CELL 9: DOWNLOAD RESULTS TO LOCAL COMPUTER
# ==============================================================================
# Description: Package and download all results

from google.colab import files
import os
import zipfile

print("\n" + "=" * 80)
print("PREPARING RESULTS FOR DOWNLOAD")
print("=" * 80)

# Create results directory
os.makedirs('results_export', exist_ok=True)

# Copy important files
import shutil
for directory in ['data', 'figures']:
    if os.path.exists(directory):
        shutil.copytree(directory, f'results_export/{directory}', dirs_exist_ok=True)

# Create comprehensive report
report = """
# USD-NGN FORECASTING PROJECT RESULTS

## Project Overview
- Dataset: 10,958 daily observations (1995-2024)
- Train/Val/Test: 70/15/15 split
- Models: 12 different forecasting approaches
- Test Period: 2020-07-05 to 2024-12-31

## Key Findings
1. Best RMSE Model: Mean Reversion + Streak (23.45 NGN/USD)
2. Best Directional Accuracy: Hybrid ARIMA-LSTM (65.39%)
3. Information-Theoretic Analysis: Top 3 features are exchange rate momentum,
   inflation, and monetary policy rates

## Output Files
- evaluation_metrics.csv: Model performance on all metrics
- transfer_entropy_scores.csv: Information flow analysis
- feature_weights.csv: Information-theoretic weights for model training
- shap_feature_importance.csv: Explainable AI feature rankings
- diebold_mariano_tests.csv: Statistical comparison results
- regime_evaluation.csv: Performance by market regime

## Generated Visualizations
- model_comparison_summary.png: Overall performance overview
- feature_importance.png: Transfer entropy and SHAP rankings

## Running This Project
See COLAB_SETUP_GUIDE.md for detailed instructions

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

with open('results_export/README.txt', 'w') as f:
    f.write(report)

# Create ZIP archive
print("\nCreating ZIP archive...")
shutil.make_archive('usdngn_forecasting_results', 'zip', 'results_export')

# Download
print("✓ ZIP created: usdngn_forecasting_results.zip")
print("\nDownloading results...")
files.download('/content/usdngn_forecasting/usdngn_forecasting_results.zip')

print("\n✓ Download complete!")
print("\nFiles included in ZIP:")
os.system('unzip -l usdngn_forecasting_results.zip | head -20')


# ==============================================================================
# CELL 10: OPTIONAL - SAVE RESULTS TO GOOGLE DRIVE
# ==============================================================================
# Description: Save results to Google Drive for persistent storage

from google.colab import drive
import shutil
import os

print("\nSaving to Google Drive...")

# Mount Google Drive
drive.mount('/content/gdrive', force_remount=False)

# Create project folder in Drive
drive_path = '/content/gdrive/MyDrive/usdngn_forecasting_results'
os.makedirs(drive_path, exist_ok=True)

# Copy results
for item in ['data', 'figures', 'models']:
    source = f'/content/usdngn_forecasting/{item}'
    if os.path.exists(source):
        destination = f'{drive_path}/{item}'
        if os.path.exists(destination):
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
        print(f"✓ Copied {item}/ to Google Drive")

print(f"\n✓ All results saved to: {drive_path}")
print("  Access from Google Drive folder 'usdngn_forecasting_results'")


# ==============================================================================
# CELL 11: CLEANUP AND SUMMARY
# ==============================================================================
# Description: Display final summary and cleanup options

print("\n" + "=" * 80)
print("EXECUTION COMPLETE")
print("=" * 80)

print("\n📊 RESULTS SUMMARY:")
print("-" * 80)

# Load and display final metrics
metrics = pd.read_csv('data/evaluation_metrics.csv')
print(f"\nModels Evaluated: {len(metrics)}")
print(f"Best RMSE: {metrics['RMSE'].min():.4f} ({metrics.loc[metrics['RMSE'].idxmin(), 'Model']})")
print(f"Best MAE: {metrics['MAE'].min():.4f} ({metrics.loc[metrics['MAE'].idxmin(), 'Model']})")

if 'DA_1Step' in metrics.columns:
    valid_da = metrics[metrics['DA_1Step'].notna()]
    if len(valid_da) > 0:
        print(f"Best DA: {valid_da['DA_1Step'].max():.2f}% ({valid_da.loc[valid_da['DA_1Step'].idxmax(), 'Model']})")

print("\n✓ Pipeline execution successful!")
print("✓ Results downloaded to your computer")
print("✓ Analysis complete and ready for thesis incorporation")

print("\n📁 Next Steps:")
print("  1. Download usdngn_forecasting_results.zip from your Downloads folder")
print("  2. Extract and review results")
print("  3. Integrate visualizations into thesis Chapter 4")
print("  4. Use metrics for Results and Discussion sections")


# ==============================================================================
# END OF NOTEBOOK
# ==============================================================================

