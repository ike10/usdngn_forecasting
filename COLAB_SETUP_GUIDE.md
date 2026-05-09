# Running USD-NGN Forecasting Project on Google Colab

## Quick Start (3 Steps)

```python
# Step 1: Clone repository
!git clone https://github.com/YOUR_USERNAME/usdngn_forecasting.git
%cd usdngn_forecasting

# Step 2: Install dependencies
!pip install -q -r requirements.txt

# Step 3: Run pipeline
!python run_pipeline.py
```

---

## DETAILED SETUP GUIDE

### Part 1: Create Colab Notebook

1. Go to [Google Colab](https://colab.research.google.com)
2. Click "New Notebook" or "File" → "New Notebook"
3. Rename: "USD-NGN Forecasting"
4. You're ready to begin!

---

### Part 2: Complete Setup Code

Run this comprehensive setup in Colab cells:

```python
# Cell 1: Install System Dependencies
!apt-get update -qq
!apt-get install -y git

# Cell 2: Clone Repository from GitHub
!git clone https://github.com/YOUR_USERNAME/usdngn_forecasting.git
%cd usdngn_forecasting

# Display directory structure
!ls -la
```

**Expected Output:**
```
total XX
drwxr-xr-x ...  Algo.txt
drwxr-xr-x ...  ALGORITHM.md
drwxr-xr-x ...  README.md
drwxr-xr-x ...  requirements.txt
drwxr-xr-x ...  run_pipeline.py
drwxr-xr-x ...  src/
drwxr-xr-x ...  data/
```

---

### Part 3: Install Python Dependencies

```python
# Cell 3: Install Required Packages
import sys
!{sys.executable} -m pip install -q \
    numpy>=1.21.0 \
    pandas>=1.3.0 \
    scipy>=1.7.0 \
    scikit-learn>=1.0.0 \
    statsmodels>=0.13.0 \
    torch>=1.10.0 \
    matplotlib>=3.5.0 \
    seaborn>=0.11.0 \
    shap>=0.41.0 \
    tqdm>=4.62.0

# Verify installations
print("✓ All dependencies installed successfully")
```

**Expected Duration**: 2-5 minutes

---

### Part 4: Run the Pipeline

#### Option A: Full Pipeline (Recommended)

```python
# Cell 4: Run Complete Pipeline
%cd /content/usdngn_forecasting

!python run_pipeline.py \
    --verbose True \
    --runtime_profile fast \
    --benchmark_mode full
```

**Expected Duration**: 15-25 minutes
**Output**: 
- `data/raw_data.csv` — Original data (10,958 observations)
- `data/processed_data.csv` — With 27 engineered features
- `data/transfer_entropy_scores.csv` — Information flow analysis
- `data/evaluation_metrics.csv` — Model performance results
- `figures/` — Visualization outputs

#### Option B: Fast Run (For Testing)

```python
# Cell 4 (Alternative): Quick Test Run
import os
os.chdir('/content/usdngn_forecasting')

from src.data_collection import DataCollector
from src.preprocessing import DataPreprocessor, DataSplitter
from src.models import ARIMAModel, RandomWalkModel

# Quick data collection and processing
print("Loading data...")
collector = DataCollector(start_date='2020-01-01', end_date='2024-12-31')
raw_data = collector.collect_all_data()
print(f"✓ Collected {len(raw_data)} observations")

# Verify key variables
print("\nData Summary:")
print(raw_data.describe())
```

**Expected Duration**: 2-3 minutes

---

### Part 5: Access and Visualize Results

```python
# Cell 5: Load and Display Results
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load evaluation metrics
metrics_df = pd.read_csv('data/evaluation_metrics.csv')
print("=" * 70)
print("MODEL PERFORMANCE SUMMARY")
print("=" * 70)
print(metrics_df.to_string(index=False))

# Plot model comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# RMSE Comparison
metrics_df.sort_values('RMSE').plot(
    x='Model', y='RMSE', kind='barh', ax=axes[0], color='steelblue'
)
axes[0].set_title('Model RMSE Comparison (Test Set)')
axes[0].set_xlabel('RMSE (NGN/USD)')

# Directional Accuracy
if 'DA_1Step' in metrics_df.columns:
    metrics_df.plot(
        x='Model', y='DA_1Step', kind='barh', ax=axes[1], color='coral'
    )
    axes[1].set_title('Directional Accuracy (Test Set)')
    axes[1].set_xlabel('Accuracy (%)')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✓ Results visualization saved as 'model_comparison.png'")
```

---

### Part 6: Download Results to Your Computer

```python
# Cell 6: Prepare Results for Download
from google.colab import files
import os

# Create ZIP archive of all results
!cd /content/usdngn_forecasting && zip -r results.zip data/ figures/

# Download
files.download('/content/usdngn_forecasting/results.zip')

print("✓ Results downloaded to your computer")
print("\nFile structure:")
!unzip -l /content/usdngn_forecasting/results.zip | head -20
```

**Output Files to Download:**
- `data/evaluation_metrics.csv` — Model performance
- `data/feature_weights.csv` — Information-theoretic weights
- `data/transfer_entropy_scores.csv` — Causality analysis
- `figures/` — All generated plots

---

## Alternative: Mount Google Drive

If you prefer working from Google Drive:

```python
# Cell 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/gdrive')

# Cell 2: Copy project from Drive (or clone from GitHub)
!cp -r /content/gdrive/MyDrive/usdngn_forecasting /content/

# Or clone from GitHub
!git clone https://github.com/YOUR_USERNAME/usdngn_forecasting.git

%cd usdngn_forecasting

# Cell 3-4: Install dependencies and run as above
```

---

## ADVANCED: Custom Analysis

### Extract and Analyze Specific Results

```python
# Cell 7: Information Flow Analysis
import pandas as pd

te_df = pd.read_csv('data/transfer_entropy_scores.csv')
print("TOP FEATURES BY TRANSFER ENTROPY")
print(te_df.head(10).to_string(index=False))

# Visualize feature importance
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.barh(te_df['feature'].head(10), te_df['te_score'].head(10))
plt.xlabel('Transfer Entropy (bits)')
plt.title('Feature Importance (Information Flow)')
plt.tight_layout()
plt.show()
```

### Compare Model Predictions

```python
# Cell 8: Detailed Model Comparison
import pandas as pd
import numpy as np

metrics = pd.read_csv('data/evaluation_metrics.csv')

# Find best model by RMSE
best_model = metrics.loc[metrics['RMSE'].idxmin()]
print(f"Best Model (RMSE): {best_model['Model']}")
print(f"RMSE: {best_model['RMSE']:.4f}")
print(f"MAE: {best_model['MAE']:.4f}")
print(f"MAPE: {best_model['MAPE']:.2f}%")

# Statistical comparison
dm_results = pd.read_csv('data/diebold_mariano_tests.csv')
print("\nStatistically Significant Model Differences:")
significant = dm_results[dm_results['p_value'] < 0.05]
print(f"Found {len(significant)} significant comparisons (p < 0.05)")
```

### SHAP Feature Importance

```python
# Cell 9: Explainability Analysis
shap_df = pd.read_csv('data/shap_feature_importance.csv')
print("FEATURE IMPORTANCE (SHAP VALUES)")
print(shap_df.head(10).to_string(index=False))

# Plot
plt.figure(figsize=(10, 6))
plt.barh(shap_df['feature'].head(10), 
         shap_df['mean_abs_shap_value'].head(10), 
         color='mediumseagreen')
plt.xlabel('Mean |SHAP Value|')
plt.title('Model Explainability: Top Features')
plt.tight_layout()
plt.show()
```

---

## TROUBLESHOOTING

### Issue 1: "pip install" Times Out

**Solution:**
```python
# Use pip with timeout and retries
!pip install --default-timeout=1000 -r requirements.txt

# Or install packages individually
!pip install numpy pandas scipy scikit-learn
!pip install statsmodels torch
!pip install matplotlib seaborn shap
```

### Issue 2: "ModuleNotFoundError" After Installation

**Solution:**
```python
# Restart kernel to load new packages
from IPython.display import clear_output
import os
os.kill(os.getpid(), 9)

# Or manually restart: Runtime → Restart runtime
```

### Issue 3: Memory Error During LSTM Training

**Solution:**
```python
# Use fast benchmark mode (default)
!python run_pipeline.py --runtime_profile fast --benchmark_mode fast_benchmarks

# Or modify run_pipeline.py to use smaller batch sizes
# Reduce sequence_length: from 15 to 12
# Reduce epochs: from 10 to 8
```

### Issue 4: Data Not Found When Cloning from GitHub

**Solution:**
```python
# If data/ directory is not in GitHub (usually gitignored), 
# let pipeline regenerate it:
!python -c "
from src.data_collection import DataCollector
from src.preprocessing import DataPreprocessor
import os

os.makedirs('data', exist_ok=True)

# Collect fresh data
collector = DataCollector()
raw_data = collector.collect_all_data()
raw_data.to_csv('data/raw_data.csv')
print('✓ Data regenerated')
"
```

### Issue 5: GPU/TPU Not Available

**Solution:**
```python
# Check available hardware
import torch
print(f"GPU Available: {torch.cuda.is_available()}")

# Enable GPU in Colab:
# Runtime → Change runtime type → Hardware accelerator → GPU

# If GPU not needed, pipeline works fine on CPU
# (just slower for LSTM/GRU models)
```

---

## PERFORMANCE OPTIMIZATION

### Use GPU for LSTM Training

```python
# Cell 2: Enable GPU
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Then run pipeline - LSTM training will automatically use GPU
!python run_pipeline.py
```

**Expected Speedup**: 5-10× faster for LSTM/GRU models

### Reduce Runtime with Fast Profile

```python
!python run_pipeline.py \
    --runtime_profile fast \
    --benchmark_mode fast_benchmarks
```

**Reduces:**
- Transfer entropy bootstrap from 100 to 40 iterations
- LSTM epochs from 10 to 8
- ARIMAX search points from 1500 to 1200

**Time**: 15 minutes instead of 45 minutes

---

## COMPLETE COLAB NOTEBOOK TEMPLATE

Create a new Colab notebook with these cells:

```python
# ============================================================
# Cell 1: System Setup
# ============================================================
!apt-get update -qq
!git clone https://github.com/YOUR_USERNAME/usdngn_forecasting.git
%cd usdngn_forecasting

# ============================================================
# Cell 2: Install Dependencies
# ============================================================
import sys
!{sys.executable} -m pip install -q \
    numpy pandas scipy scikit-learn statsmodels torch \
    matplotlib seaborn shap tqdm

# ============================================================
# Cell 3: Verify GPU
# ============================================================
import torch
print(f"GPU: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# ============================================================
# Cell 4: Run Pipeline
# ============================================================
!python run_pipeline.py --verbose True --runtime_profile fast

# ============================================================
# Cell 5: Display Results
# ============================================================
import pandas as pd
metrics = pd.read_csv('data/evaluation_metrics.csv')
print(metrics.to_string(index=False))

# ============================================================
# Cell 6: Visualize Results
# ============================================================
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 6))
metrics.sort_values('RMSE').plot(x='Model', y='RMSE', kind='barh', ax=ax)
ax.set_title('Model Performance Comparison (RMSE)')
ax.set_xlabel('RMSE (NGN/USD)')
plt.tight_layout()
plt.show()

# ============================================================
# Cell 7: Download Results
# ============================================================
from google.colab import files
!zip -r results.zip data/ figures/
files.download('results.zip')
```

---

## SCHEDULING COLAB NOTEBOOK TO RUN PERIODICALLY

Use Google Colab with GitHub integration:

```python
# Cell 0: Auto-save to GitHub (optional)
!git config user.email "your_email@gmail.com"
!git config user.name "Your Name"
!git add -A
!git commit -m "Automated run: $(date)"
!git push origin main
```

**Note**: Colab notebooks stop after 90 minutes of inactivity. For longer runs, connect a local runtime or use Colab with Google Cloud.

---

## USING WITH GOOGLE CLOUD

For production runs longer than Colab timeout:

```bash
# 1. Install Google Cloud CLI locally
gcloud init

# 2. Create Cloud VM
gcloud compute instances create usdngn-forecast \
    --image-family=ubuntu-2004-lts \
    --image-project=ubuntu-os-cloud

# 3. SSH into VM
gcloud compute ssh usdngn-forecast

# 4. Clone and run
git clone https://github.com/YOUR_USERNAME/usdngn_forecasting.git
cd usdngn_forecasting
pip install -r requirements.txt
python run_pipeline.py
```

---

## COMPARISON: LOCAL vs COLAB vs CLOUD

| Feature | Local | Colab | Cloud VM |
|---------|-------|-------|----------|
| **Setup Time** | 10 min | 3 min | 15 min |
| **Runtime** | Unlimited | 90 min | Unlimited |
| **GPU** | Optional | Yes (free) | Optional |
| **Cost** | None | Free | ~$0.10/hour |
| **Data Storage** | Local | Google Drive | Google Storage |
| **Ease** | Medium | Easy | Medium |

---

## QUICK REFERENCE COMMANDS

```python
# Check current directory
import os
print(os.getcwd())

# List files
!ls -la

# Run pipeline with options
!python run_pipeline.py \
    --verbose True \
    --runtime_profile fast \
    --benchmark_mode full

# Check project size
!du -sh .

# View a specific file
!head -20 data/evaluation_metrics.csv

# Monitor execution time
import time
start = time.time()
# (run code)
print(f"Execution time: {time.time() - start:.2f} seconds")
```

---

## SUCCESS CHECKLIST

- [ ] Colab notebook created
- [ ] Repository cloned successfully
- [ ] Dependencies installed without errors
- [ ] GPU enabled (if using)
- [ ] Pipeline runs and generates output
- [ ] data/ folder contains CSV files
- [ ] Results downloaded to local computer
- [ ] Visualizations generated and viewed

---

## NEXT STEPS

1. **Create GitHub repository**: Push your project code
2. **Get GitHub URL**: `https://github.com/YOUR_USERNAME/usdngn_forecasting.git`
3. **Replace in Colab**: Update `!git clone` with your URL
4. **Share Colab notebook**: Link with others via Google Drive
5. **Automate**: Set up scheduled runs with Cloud Tasks

---

## ADDITIONAL RESOURCES

- [Google Colab Docs](https://colab.research.google.com/notebooks/intro.ipynb)
- [PyTorch in Colab](https://pytorch.org/tutorials/beginner/colab.html)
- [GitHub Integration](https://colab.research.google.com/github/OWNER/REPO)

---

**Document Version**: 1.0  
**Date**: May 9, 2026  
**Status**: Ready for Implementation
