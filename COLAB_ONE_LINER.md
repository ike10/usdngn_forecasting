# USD-NGN Forecasting - Colab One-Liner Launcher
# Copy ONE of these commands into a single Colab cell and run

# ==============================================================================
# OPTION 1: FASTEST - Setup + Run (3-5 minutes)
# ==============================================================================
# For quick testing and validation
# Copy and paste this ENTIRE command into ONE Colab cell:

!git clone https://github.com/YOUR_USERNAME/usdngn_forecasting.git && cd usdngn_forecasting && pip install -q numpy pandas scipy scikit-learn statsmodels torch matplotlib seaborn shap tqdm && python run_pipeline.py --verbose True --runtime_profile fast --benchmark_mode fast_benchmarks && echo "✓ Complete!" && ls -lh data/


# ==============================================================================
# OPTION 2: BALANCED - Setup + Full Pipeline (15-25 minutes)
# ==============================================================================
# For complete thesis-quality results
# Copy and paste this ENTIRE command into ONE Colab cell:

!git clone https://github.com/YOUR_USERNAME/usdngn_forecasting.git && cd usdngn_forecasting && pip install -q numpy pandas scipy scikit-learn statsmodels torch matplotlib seaborn shap tqdm && python run_pipeline.py --verbose True --runtime_profile fast --benchmark_mode full && echo "✓ Complete!" && python -c "import pandas as pd; m = pd.read_csv('data/evaluation_metrics.csv'); print('\n' + '='*60); print('RESULTS SUMMARY'); print('='*60); print(m[['Model','RMSE','MAE','MAPE']].to_string(index=False))"


# ==============================================================================
# OPTION 3: FULL - Complete with GPU + Analysis (20-30 minutes)
# ==============================================================================
# For maximum performance and detailed analysis
# Copy and paste this ENTIRE command into ONE Colab cell:

!apt-get update -qq && apt-get install -y git && git clone https://github.com/YOUR_USERNAME/usdngn_forecasting.git && cd usdngn_forecasting && pip install -q numpy pandas scipy scikit-learn statsmodels torch matplotlib seaborn shap tqdm && python run_pipeline.py --verbose True --runtime_profile full --benchmark_mode full && echo "✓ Pipeline Complete!" && python -c "import pandas as pd; import matplotlib.pyplot as plt; import seaborn as sns; metrics = pd.read_csv('data/evaluation_metrics.csv'); print('\n' + '='*60); print('FINAL RESULTS'); print('='*60); print('Top 5 Models:'); print(metrics.nsmallest(5, 'RMSE')[['Model','RMSE','MAE']].to_string(index=False))" && python -c "import pandas as pd; import matplotlib.pyplot as plt; sns.set_style('whitegrid'); metrics = pd.read_csv('data/evaluation_metrics.csv'); fig, ax = plt.subplots(figsize=(12,6)); m = metrics.sort_values('RMSE'); ax.barh(m['Model'], m['RMSE']); ax.set_xlabel('RMSE'); ax.set_title('Model Comparison'); plt.tight_layout(); plt.savefig('results.png', dpi=100); print('✓ Visualization saved'); plt.show()"


# ==============================================================================
# OPTION 4: DOWNLOAD - Setup + Run + Download Results (25-35 minutes)
# ==============================================================================
# Complete setup with automatic download
# Copy and paste this ENTIRE command into ONE Colab cell:

!git clone https://github.com/YOUR_USERNAME/usdngn_forecasting.git && cd usdngn_forecasting && pip install -q numpy pandas scipy scikit-learn statsmodels torch matplotlib seaborn shap tqdm && python run_pipeline.py --verbose True --runtime_profile fast --benchmark_mode full && cd .. && zip -r -q usdngn_results.zip usdngn_forecasting/data usdngn_forecasting/figures 2>/dev/null || true && from google.colab import files; files.download('usdngn_results.zip'); print('✓ Download Complete!')


# ==============================================================================
# QUICK REFERENCE: STEP-BY-STEP FOR BEGINNERS
# ==============================================================================

# Step 1: Create new Colab notebook at https://colab.research.google.com/
# Step 2: (Optional) Enable GPU: Runtime → Change runtime type → GPU
# Step 3: Copy ONE of the launcher commands above into a cell
# Step 4: Replace YOUR_USERNAME with your GitHub username
# Step 5: Run the cell (Ctrl+Enter or press play button)
# Step 6: Wait for completion and review results


# ==============================================================================
# INSTALLATION ONLY (if you just want dependencies)
# ==============================================================================

!pip install -q numpy pandas scipy scikit-learn statsmodels torch matplotlib seaborn shap tqdm

# Then run:
!git clone https://github.com/YOUR_USERNAME/usdngn_forecasting.git
%cd usdngn_forecasting
!python run_pipeline.py


# ==============================================================================
# CLONE ONLY (if you already have Colab set up)
# ==============================================================================

!git clone https://github.com/YOUR_USERNAME/usdngn_forecasting.git
%cd usdngn_forecasting


# ==============================================================================
# RUN ONLY (if already in project directory)
# ==============================================================================

!python run_pipeline.py --verbose True --runtime_profile fast --benchmark_mode full


# ==============================================================================
# TROUBLESHOOTING ONE-LINERS
# ==============================================================================

# Check GPU availability:
!python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"

# Check package versions:
!python -c "import pandas as pd; import torch; import sklearn; print(f'Pandas: {pd.__version__}\\nPyTorch: {torch.__version__}\\nScikit-learn: {sklearn.__version__}')"

# Quick test run (5 minutes):
!cd usdngn_forecasting && python run_pipeline.py --verbose True --runtime_profile fast --benchmark_mode fast_benchmarks

# Display results:
!python -c "import pandas as pd; print(pd.read_csv('usdngn_forecasting/data/evaluation_metrics.csv').head(10))"

# Create result visualization:
!python -c "import pandas as pd; import matplotlib.pyplot as plt; m = pd.read_csv('usdngn_forecasting/data/evaluation_metrics.csv'); fig, ax = plt.subplots(); ax.barh(m['Model'], m['RMSE']); plt.savefig('comparison.png'); print('✓ Saved')"


# ==============================================================================
# NOTES
# ==============================================================================

# - Replace YOUR_USERNAME with your actual GitHub username
# - Ensure repository is PUBLIC (Settings → Visibility → Public)
# - Total execution time: 5-30 minutes depending on option
# - GPU recommended for faster model training (Option 2-4)
# - Option 1: Testing only (limited training budget)
# - Option 2: Recommended for most users (balanced speed/quality)
# - Option 3: Best quality (uses full training budget)
# - Option 4: Complete package ready to download

# After execution:
# 1. Check data/ folder for CSV results
# 2. Use Option 4 to download and save locally
# 3. Review metrics for thesis integration
# 4. Extract ZIP file to access all results


# ==============================================================================
# TERMINAL COMMANDS (for local machine setup before Colab)
# ==============================================================================

# Initialize Git repository:
# cd your_project_directory
# git init
# git add .
# git commit -m "Initial commit"
# git remote add origin https://github.com/YOUR_USERNAME/usdngn_forecasting.git
# git branch -M main
# git push -u origin main

# Then use Colab with: https://github.com/YOUR_USERNAME/usdngn_forecasting.git

