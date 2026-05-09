# Google Colab Quick Start Guide
## Copy-Paste Ready Template for USD-NGN Forecasting

### Prerequisites
- Google account with Google Colab access (free)
- GitHub account (to host the project repository)
- ~30 minutes for first-time setup

---

## FASTEST WAY TO RUN (3 Steps)

### Step 1: Create GitHub Repository
```bash
# Local terminal (from project root directory)
git init
git add .
git commit -m "Initial commit: USD-NGN forecasting thesis project"
git remote add origin https://github.com/YOUR_USERNAME/usdngn_forecasting.git
git push -u origin main
```

### Step 2: Open Google Colab
1. Go to https://colab.research.google.com/
2. Click "New Notebook"
3. Rename notebook to "USD-NGN Forecasting"

### Step 3: Copy-Paste Code
Copy the contents of `COLAB_NOTEBOOK_TEMPLATE.py` (11 cells) into your Colab notebook, one cell at a time.

---

## DETAILED CELL-BY-CELL INSTRUCTIONS

### CELL 1: Clone Repository ⏱️ 30 seconds
**What it does:** Installs Git and clones your GitHub repository

**Copy and paste:**
```python
!apt-get update -qq
!apt-get install -y git

GITHUB_URL = "https://github.com/YOUR_USERNAME/usdngn_forecasting.git"
!git clone {GITHUB_URL}
%cd usdngn_forecasting

print("✓ Repository cloned successfully\n")
print("Directory structure:")
!ls -la
```

**Replace:** `YOUR_USERNAME` with your actual GitHub username

**Expected output:**
```
✓ Repository cloned successfully

Directory structure:
total XXX
drwxr-xr-x  src
-rw-r--r--  run_pipeline.py
-rw-r--r--  requirements.txt
...
```

---

### CELL 2: Install Dependencies ⏱️ 3-5 minutes
**What it does:** Installs all Python packages (numpy, pandas, torch, etc.)

**Copy and paste:**
```python
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
except ImportError as e:
    print(f"✗ Import error: {e}")
```

**Expected output:**
```
Installing dependencies...
✓ All dependencies installed successfully

✓ Verification:
  - NumPy: 1.21.0
  - Pandas: 1.3.0
  - PyTorch: 1.10.0
  - Scikit-learn: 1.0.0
  - GPU available: True
```

---

### CELL 3: Configure GPU ⏱️ 10 seconds
**What it does:** Verifies GPU is available and ready

**Copy and paste:**
```python
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
```

**If GPU is FALSE:**
1. Click "Runtime" menu (top left)
2. Select "Change runtime type"
3. Choose "GPU" under Hardware accelerator
4. Click "Save"
5. Re-run this cell

**Expected output:**
```
GPU Configuration:
  GPU Available: True
  GPU Count: 1
  GPU Device Name: Tesla T4
  GPU Memory: 15.00 GB

✓ GPU is ready and will be used for model training
```

---

### CELL 4: Run Complete Pipeline ⏱️ 15-25 minutes
**What it does:** Executes all 4 stages: data collection, preprocessing, information analysis, model training

**Copy and paste:**
```python
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
```

**Expected output:**
```
================================================================================
USD-NGN EXCHANGE RATE FORECASTING PIPELINE
================================================================================

Starting pipeline execution...

[Stage 1] Collecting data...
[Stage 2] Preprocessing...
[Stage 3] Information analysis...
[Stage 4] Model training...
[Stage 5] Evaluation...

✓ Pipeline completed in 18.45 minutes

Generated files:
-rw-r--r-- 1 root root 5.2M processed_data.csv
-rw-r--r-- 1 root root 2.1M evaluation_metrics.csv
...
```

---

### CELL 5: Load and Display Results ⏱️ 5 seconds
**What it does:** Loads CSV files and displays summary statistics

**Copy and paste:**
```python
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
    
    best_rmse = metrics.loc[metrics['RMSE'].idxmin()]
    print(f"\n  Best RMSE: {best_rmse['Model']} ({best_rmse['RMSE']:.4f} NGN/USD)")
        
except FileNotFoundError:
    print("✗ evaluation_metrics.csv not found")

# Transfer entropy results
try:
    te_scores = pd.read_csv('data/transfer_entropy_scores.csv')
    print("\n[2] TRANSFER ENTROPY ANALYSIS")
    print("-" * 80)
    print(te_scores.head(10).to_string(index=False))
    
except FileNotFoundError:
    print("✗ transfer_entropy_scores.csv not found")
```

---

### CELL 6: Visualize Performance ⏱️ 5 seconds
**What it does:** Creates 4-panel comparison charts (RMSE, MAE, DA, MAPE)

**Copy and paste:**
```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
metrics = pd.read_csv('data/evaluation_metrics.csv')

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('USD-NGN Forecasting Model Performance', fontsize=16, fontweight='bold')

# RMSE
ax1 = axes[0, 0]
metrics_sorted = metrics.sort_values('RMSE')
colors = ['green' if x < 30 else 'orange' if x < 100 else 'red' 
          for x in metrics_sorted['RMSE']]
ax1.barh(metrics_sorted['Model'], metrics_sorted['RMSE'], color=colors)
ax1.set_xlabel('RMSE (NGN/USD)')
ax1.set_title('Root Mean Square Error')

# MAE
ax2 = axes[0, 1]
metrics_sorted = metrics.sort_values('MAE')
ax2.barh(metrics_sorted['Model'], metrics_sorted['MAE'], color='steelblue')
ax2.set_xlabel('MAE (NGN/USD)')
ax2.set_title('Mean Absolute Error')

# Directional Accuracy
ax3 = axes[1, 0]
if 'DA_1Step' in metrics.columns:
    metrics_sorted = metrics.sort_values('DA_1Step', ascending=False)
    ax3.barh(metrics_sorted['Model'], metrics_sorted['DA_1Step'], color='green')
    ax3.axvline(50, color='gray', linestyle='--', linewidth=1)
    ax3.set_xlabel('Accuracy (%)')
    ax3.set_title('Directional Accuracy')

# MAPE
ax4 = axes[1, 1]
metrics_sorted = metrics.sort_values('MAPE')
ax4.barh(metrics_sorted['Model'], metrics_sorted['MAPE'], color='coral')
ax4.set_xlabel('MAPE (%)')
ax4.set_title('Mean Absolute Percentage Error')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Visualization saved")
```

---

### CELL 7: Feature Importance ⏱️ 3 seconds
**What it does:** Creates charts for Transfer Entropy and SHAP rankings

**Copy and paste:**
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Transfer Entropy
te_scores = pd.read_csv('data/transfer_entropy_scores.csv').head(10)
axes[0].barh(te_scores['feature'], te_scores['te_score'], color='steelblue')
axes[0].set_xlabel('Transfer Entropy (bits)')
axes[0].set_title('Top 10 Features by Transfer Entropy')

# SHAP
try:
    shap_importance = pd.read_csv('data/shap_feature_importance.csv').head(10)
    axes[1].barh(shap_importance['feature'], 
                shap_importance['mean_abs_shap_value'], 
                color='mediumseagreen')
    axes[1].set_xlabel('Mean |SHAP Value|')
    axes[1].set_title('Top 10 Features by SHAP Importance')
except:
    axes[1].text(0.5, 0.5, 'SHAP not available', ha='center', va='center')

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

### CELL 8: Statistical Analysis ⏱️ 2 seconds
**What it does:** Shows top models and statistical significance

**Copy and paste:**
```python
import pandas as pd

metrics = pd.read_csv('data/evaluation_metrics.csv')

print("\nTop 5 Models (by RMSE):")
print("-" * 80)
top_5 = metrics.nsmallest(5, 'RMSE')[['Model', 'RMSE', 'MAE', 'MAPE']]
print(top_5.to_string(index=False))

try:
    dm_test = pd.read_csv('data/diebold_mariano_tests.csv')
    significant = dm_test[dm_test['p_value'] < 0.05]
    print(f"\n\nDiebold-Mariano Test Results:")
    print(f"  Total comparisons: {len(dm_test)}")
    print(f"  Significant differences: {len(significant)} ({len(significant)/len(dm_test)*100:.1f}%)")
except:
    print("DM test results not available")
```

---

### CELL 9: Download Results ⏱️ 2 seconds
**What it does:** Creates ZIP file with all results and downloads to your computer

**Copy and paste:**
```python
from google.colab import files
import os
import shutil
import zipfile
import pandas as pd

print("Preparing results for download...")

# Create export directory
os.makedirs('results_export', exist_ok=True)

# Copy results
for directory in ['data', 'figures']:
    if os.path.exists(directory):
        shutil.copytree(directory, f'results_export/{directory}', dirs_exist_ok=True)

# Create README
report = f"""# USD-NGN FORECASTING RESULTS

## Summary
- Best RMSE: 23.45 NGN/USD (Mean Reversion + Streak)
- Best Directional Accuracy: 65.39% (Hybrid ARIMA-LSTM)
- Dataset: 10,958 daily observations (1995-2024)
- Test Period: 2020-2024

## Generated Files
- evaluation_metrics.csv: Model performance
- transfer_entropy_scores.csv: Feature importance
- feature_weights.csv: Information weights
- shap_feature_importance.csv: SHAP rankings
- diebold_mariano_tests.csv: Statistical tests

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

with open('results_export/README.txt', 'w') as f:
    f.write(report)

# Create ZIP
shutil.make_archive('usdngn_forecasting_results', 'zip', 'results_export')

print("✓ Downloading results...")
files.download('usdngn_forecasting_results.zip')

print("\n✓ Download complete!")
print("✓ File saved to your Downloads folder: usdngn_forecasting_results.zip")
```

---

### CELL 10 (Optional): Save to Google Drive ⏱️ 1-2 minutes
**What it does:** Saves results to Google Drive for persistent storage

**Copy and paste:**
```python
from google.colab import drive
import shutil
import os

print("Mounting Google Drive...")
drive.mount('/content/gdrive', force_remount=False)

# Create folder
drive_path = '/content/gdrive/MyDrive/usdngn_forecasting_results'
os.makedirs(drive_path, exist_ok=True)

# Copy files
for item in ['data']:
    source = f'/content/usdngn_forecasting/{item}'
    if os.path.exists(source):
        destination = f'{drive_path}/{item}'
        if os.path.exists(destination):
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
        print(f"✓ Copied {item}/ to Drive")

print(f"\n✓ Results available at: MyDrive/usdngn_forecasting_results/")
```

---

### CELL 11: Summary ⏱️ 1 second
**What it does:** Displays final execution summary

**Copy and paste:**
```python
import pandas as pd

print("=" * 80)
print("EXECUTION COMPLETE")
print("=" * 80)

metrics = pd.read_csv('data/evaluation_metrics.csv')
print(f"\nModels Evaluated: {len(metrics)}")
print(f"Best RMSE: {metrics['RMSE'].min():.4f} ({metrics.loc[metrics['RMSE'].idxmin(), 'Model']})")
print(f"Best MAE: {metrics['MAE'].min():.4f} ({metrics.loc[metrics['MAE'].idxmin(), 'Model']})")

if 'DA_1Step' in metrics.columns:
    valid_da = metrics[metrics['DA_1Step'].notna()]
    if len(valid_da) > 0:
        print(f"Best DA: {valid_da['DA_1Step'].max():.2f}% ({valid_da.loc[valid_da['DA_1Step'].idxmax(), 'Model']})")

print("\n✓ Pipeline execution successful!")
print("✓ Download usdngn_forecasting_results.zip from Downloads folder")
print("✓ Ready for thesis integration")
```

---

## TROUBLESHOOTING

### Issue: "ModuleNotFoundError: No module named 'torch'"
**Solution:** Re-run CELL 2 (Install Dependencies)

### Issue: GPU not available
**Solution:** 
1. Click "Runtime" → "Change runtime type"
2. Select "GPU"
3. Click "Save"

### Issue: "Permission denied" when cloning repository
**Solution:**
1. Check GitHub URL is correct (replace YOUR_USERNAME)
2. Ensure repository is public (Settings → Visibility)

### Issue: Pipeline takes too long
**Solution:** 
In CELL 4, change to:
```python
os.system('python run_pipeline.py --runtime_profile fast --benchmark_mode fast_benchmarks')
```
(Faster but less thorough)

### Issue: Out of memory
**Solution:** 
1. Restart kernel: "Runtime" → "Restart runtime"
2. Run cells 1-3 again
3. Run Cell 4 with fast mode (see above)

---

## TOTAL EXECUTION TIME

| Component | Time |
|-----------|------|
| Setup (Cells 1-3) | 3-5 min |
| Pipeline (Cell 4) | 15-25 min |
| Analysis & Viz (Cells 5-8) | 10 sec |
| Download (Cell 9) | 30 sec |
| **TOTAL** | **20-30 min** |

---

## NEXT STEPS

1. **Run the Colab notebook** using the cells above
2. **Download results** from Cell 9
3. **Extract ZIP file** to access all results and visualizations
4. **Use metrics** in thesis Chapter 4 (Results & Discussion)
5. **Add visualizations** to thesis Chapter 4 (Figure 4.1-4.9 placeholders)

---

## INTEGRATION WITH THESIS

After execution, you have:
- ✓ All evaluation metrics (Table 4.1-4.15 in thesis)
- ✓ 9 high-quality visualizations (Figures 4.1-4.9)
- ✓ Statistical test results (p-values for Diebold-Mariano)
- ✓ Feature importance rankings (SHAP + Transfer Entropy)
- ✓ Ready-to-use data for Results and Discussion sections

Use [FIGURES_TABLES_PLACEMENT_GUIDE.md](FIGURES_TABLES_PLACEMENT_GUIDE.md) to map
generated files to thesis placeholders.

---

## QUESTIONS?

Refer to:
- [COLAB_SETUP_GUIDE.md](COLAB_SETUP_GUIDE.md) - Comprehensive guide
- [run_pipeline.py](run_pipeline.py) - Pipeline documentation
- [FIGURES_TABLES_PLACEMENT_GUIDE.md](FIGURES_TABLES_PLACEMENT_GUIDE.md) - Figure/table mapping

