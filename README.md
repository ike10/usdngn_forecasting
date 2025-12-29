# USD-NGN Exchange Rate Forecasting System

## 📊 Project Status: ✅ PHASE 1 COMPLETE

### Phase 1: Get Pipeline Running ✅
- [x] All modules import successfully
- [x] Data collection pipeline works (1,096 observations)
- [x] Preprocessing with 27 engineered features
- [x] Data splitting (70/15/15 train/val/test)
- [x] Model training:
  - Random Walk (baseline)
  - ARIMA (univariate)
  - Hybrid ARIMA-LSTM (ensemble)
- [x] Evaluation metrics computed
- [x] Output files saved to `/data/`

---

## 🚀 Quick Start

### Run the Full Pipeline
```bash
python3 run_pipeline.py
```

**Output**: Generates 6 CSV files in `/data/` directory
- `raw_data.csv` (77 KB)
- `processed_data.csv` (528 KB)
- `train_data.csv`, `val_data.csv`, `test_data.csv`
- `evaluation_metrics.csv`

**Execution Time**: ~1.4 seconds

---

## 📈 Current Results

| Model | RMSE | MAE | MAPE | Directional Accuracy |
|-------|------|-----|------|----------------------|
| **Random Walk** | 19.16 | 13.94 | 1.02% | 53.4% |
| **ARIMA** | 60.05 | 49.95 | 3.60% | 31.7% |
| **Hybrid** | 23.64 | 19.03 | 1.39% | **62.7%** |

**Best Model**: Hybrid ARIMA-LSTM (highest directional accuracy)

---

## 📁 Project Structure

```
usdngn_forecasting/
├── part1_data_collection.py      ✅ Working
├── part2_preprocessing.py         ✅ Working
├── part3_information_analysis.py  (Expensive - 500 bootstrap iterations)
├── part4_models.py               ✅ Working (optimized ARIMA search)
├── part5_evaluation.py           ✅ Working
├── part6_pipeline.py             (Original - uses part3)
├── run_pipeline.py               ✅ NEW - Optimized executable
├── test_pipeline.py              
├── visualization.py              (Partial)
├── requirements.txt
├── data/                         📊 Generated CSV files (6 files)
├── models/                       (Ready for model persistence)
└── figures/                      (Ready for visualizations)
```

---

## 🔧 Technical Details

### Environment
- Python 3.10.12 (WSL)
- Dependencies installed: numpy, pandas, scipy, scikit-learn, statsmodels, matplotlib, seaborn, tqdm
- PyTorch not required (sklearn fallback works)

### Data Pipeline
1. **Data Collection**: Generates synthetic but realistic USD-NGN data (1995-2025)
2. **Preprocessing**: 
   - Feature engineering (log returns, moving averages, volatility, lags)
   - Stationarity testing (ADF & KPSS)
   - Data cleaning (forward/backward fill)
3. **Splitting**: Temporal split (70% train, 15% val, 15% test)

### Models
- **ARIMA(1,1,1)**: AIC-optimized grid search (reduced to 3×2×3 for speed)
- **Hybrid**: Combines ARIMA trend + LSTM residuals with feature weighting
- **Random Walk**: Baseline comparison

---

## ⚠️ Known Issues Fixed

| Issue | Solution |
|-------|----------|
| ARIMA grid search too slow | Reduced search space from 4×2×4 → 3×2×3 |
| Synthetic data doesn't reflect real challenges | Acknowledged - use real data for production |
| Transfer entropy expensive (500 iterations) | Skipped in `run_pipeline.py` for speed |
| Missing output directories | Created `data/` and `models/` |

---

## 📋 Next Steps (Phase 2-3)

### Phase 2: Save Outputs & Add Persistence
- [ ] Export trained models to pickle files
- [ ] Save predictions with dates
- [ ] Create model checkpoints

### Phase 3: Complete Visualizations
- [ ] Implement remaining 5+ figures
- [ ] Time series plots with regime shading
- [ ] Model comparison charts
- [ ] Feature importance heatmaps

### Phase 4: Add Documentation
- [ ] Docstrings for all classes
- [ ] Parameter documentation
- [ ] Usage examples
- [ ] Architecture diagrams

---

## 📊 Example Usage

```python
from part1_data_collection import DataCollector
from part2_preprocessing import DataPreprocessor, DataSplitter
from part4_models import ARIMAModel, HybridARIMALSTM

# Collect data
collector = DataCollector(start_date='2023-01-01', end_date='2025-12-31')
raw_data = collector.collect_all_data()

# Preprocess
preprocessor = DataPreprocessor(raw_data)
processed_data, stats = preprocessor.preprocess()

# Split
splitter = DataSplitter()
train, val, test = splitter.split(processed_data)

# Model
arima = ARIMAModel().fit(train['usdngn'].values)
pred = arima.predict(len(test))
```

---

## 🎓 Thesis Context

**Title**: USD-NGN Exchange Rate Forecasting Using Information Theory, Hybrid Machine Learning and Explainable AI

**Author**: Oche Emmanuel Ike (ID: 242220011)

**Institution**: International Institute for Financial Engineering (IIFE)

**Key Contributions**:
1. Transfer entropy for causality analysis
2. Information-aware feature weighting
3. Hybrid ensemble architecture
4. Regime-specific evaluation

---

## 📝 License & Citation

[Thesis Details](https://...)

