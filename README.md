# USD-NGN Exchange Rate Forecasting System

A hybrid machine learning system for forecasting USD-NGN exchange rates using 30 years of historical data (1995-2024).

**Author**: Oche Emmanuel Ike (ID: 242220011)
**Thesis**: USD-NGN Exchange Rate Forecasting Using Information Theory, Hybrid Machine Learning and Explainable AI

---

## Overview

This project implements multiple forecasting models for the Nigerian Naira (NGN) to US Dollar (USD) exchange rate, with the goal of beating the Random Walk baseline - a notoriously difficult benchmark in exchange rate forecasting.

### Key Features
- **30 years of verified historical data** (1995-2024) from official sources
- **Multiple model architectures**: ARIMA, LSTM, Hybrid ARIMA-LSTM, Mean Reversion + Contrarian
- **Comprehensive evaluation**: RMSE, MAE, MAPE, Directional Accuracy, Diebold-Mariano test
- **Information-theoretic analysis**: Transfer entropy for causality detection

---

## Data Summary

### Coverage
| Metric | Value |
|--------|-------|
| Total observations | 10,958 daily records |
| Date range | January 1, 1995 - December 31, 2024 |
| Years covered | 30 years |

### Historical USD-NGN Rates (Verified)
| Year | Rate (NGN/USD) | Key Event |
|------|----------------|-----------|
| 1995 | 21.89 | Abacha era (fixed official rate) |
| 1999 | 92.34 | Return to democracy |
| 2008 | 118.55 | Global financial crisis |
| 2015 | 192.44 | Oil price crash |
| 2016 | 253.49 | Major devaluation, recession |
| 2020 | 358.81 | COVID-19 impact |
| 2023 | 645.19 | June 2023 float |
| 2024 | 1,478.97 | Historic depreciation |

### Data Sources
- **USD-NGN**: World Bank (PA.NUS.FCRF, official exchange rate)
- **Brent Crude Oil**: FRED (DCOILBRENTEU, daily; annual averages)
- **Monetary Policy Rate**: CBN Monetary Policy Committee Decisions
- **CPI/Inflation**: World Bank (FP.CPI.TOTL.ZG)

### Data Splits
| Split | Records | Period | USD-NGN Range |
|-------|---------|--------|---------------|
| Train (70%) | 7,655 | 1995-01-22 to 2016-01-06 | 21.27 - 257.36 |
| Validation (15%) | 1,641 | 2016-01-07 to 2020-07-04 | 245.27 - 365.90 |
| Test (15%) | 1,641 | 2020-07-05 to 2024-12-31 | 349.61 - 1,506.02 |

---

## Model Results

Results depend on the data window, feature lagging, and evaluation method.
The pipeline reports multi-step error metrics (RMSE/MAE/MAPE) and a rolling
one-step directional accuracy (`DA_1Step`) to avoid flat multi-step baselines.
Run the pipeline to regenerate `data/evaluation_metrics.csv` for the current setup.

### Model Descriptions

1. **Random Walk**: Baseline model predicting future rates as the last observed value
2. **Moving Average**: Baseline model using a rolling mean forecast
3. **ARIMA**: Auto-regressive Integrated Moving Average with automatic order selection
4. **Hybrid ARIMA-LSTM**: ARIMA for trend + LSTM for residuals
5. **Winning Hybrid**: Mean reversion toward 20-day MA + contrarian signal
6. **Improved Hybrid**: Winning Hybrid + streak detection for trend reversals

---

## Quick Start

### Installation
```bash
# Clone repository
git clone <repository-url>
cd usdngn_forecasting

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Run Full Pipeline
```bash
python run_pipeline.py
```

This executes:
1. Data collection (10,958 observations from verified sources)
2. Preprocessing (27 engineered features, lagged for evaluation)
3. Data splitting (70/15/15 train/val/test)
4. Model training (6 models)
5. Evaluation and comparison
6. Results saved to `data/evaluation_metrics.csv`

### Programmatic Usage
```python
from run_pipeline import run_pipeline

# Run pipeline
results = run_pipeline(verbose=True)

# Access results
models = results['models']
predictions = results['predictions']
metrics = results['metrics']
X_train, y_train = results['data']['train']
```

---

## Project Structure

```
usdngn_forecasting/
├── run_pipeline.py           # Main entry point
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── TODO.md                   # Project roadmap
│
├── src/                      # Core modules
│   ├── __init__.py           # Package exports
│   ├── data_collection.py    # Historical data (30 years)
│   ├── preprocessing.py      # Feature engineering
│   ├── models.py             # ARIMA, LSTM, Hybrid ARIMA-LSTM
│   ├── hybrid_model.py       # WinningHybrid, ImprovedHybrid
│   ├── evaluation.py         # Metrics, statistical tests
│   ├── information_analysis.py  # Transfer entropy
│   └── visualization.py      # Plotting utilities
│
├── data/                     # Generated data files
│   ├── raw_data.csv          # Raw collected data
│   ├── processed_data.csv    # Engineered features
│   ├── train_data.csv        # Training set
│   ├── val_data.csv          # Validation set
│   ├── test_data.csv         # Test set
│   └── evaluation_metrics.csv # Model comparison
│
├── models/                   # Saved model files
├── figures/                  # Generated visualizations
└── .venv/                    # Virtual environment
```

---

## Module Documentation

### src/data_collection.py
Generates daily time series from verified yearly historical data.

```python
from src.data_collection import DataCollector

collector = DataCollector(start_date='1995-01-01', end_date='2024-12-31')
data = collector.collect_all_data()
# Returns DataFrame with: usdngn, brent_oil, mpr, cpi
```

### src/preprocessing.py
Feature engineering and data splitting.

```python
from src.preprocessing import DataPreprocessor, DataSplitter

preprocessor = DataPreprocessor(raw_data)
processed, stationarity = preprocessor.preprocess()
# Adds 27 features: returns, MAs, volatility, lags, etc.

splitter = DataSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
train, val, test = splitter.split(processed)
```

### src/hybrid_model.py
The winning model implementations.

```python
from src.hybrid_model import WinningHybridModel, ImprovedHybridModel

model = WinningHybridModel(mr_speed=0.05, contrarian_magnitude=1.0, ma_window=20)
model.fit(X_train, y_train, X_val, y_val)
predictions = model.predict(X_test, y_train)
```

### src/evaluation.py
Comprehensive metrics and statistical tests.

```python
from src.evaluation import ModelEvaluator, DieboldMarianoTest

# Compute metrics
metrics = ModelEvaluator.compute_all_metrics(y_true, y_pred)
# Returns: RMSE, MAE, MAPE, DA, N

# Statistical significance test
dm = DieboldMarianoTest.test(y_true, pred1, pred2)
# Returns: dm_stat, p_value, conclusion
```

---

## Engineered Features (27 total)

| Category | Features |
|----------|----------|
| Price | usdngn, brent_oil, mpr, cpi |
| Returns | usdngn_return, oil_return |
| Moving Averages | usdngn_ma5, usdngn_ma20, oil_ma5, oil_ma20 |
| Volatility | usdngn_volatility, oil_volatility |
| Momentum | usdngn_momentum, oil_momentum |
| Ratios | rate_oil_ratio, mpr_change |
| Lags | usdngn_lag1-5, oil_lag1-5 |

---

## Key Findings

1. **Random Walk is a strong baseline**: The RW model achieves competitive RMSE because exchange rates follow near-random patterns

2. **ARIMA struggles with regime changes**: The 2023-2024 devaluation (21 NGN → 1,500 NGN) breaks ARIMA assumptions

3. **Mean reversion works**: Exchange rates tend to revert toward moving averages, especially after extreme moves

4. **Contrarian signals add value**: Predicting reversal after consecutive same-direction days improves accuracy

5. **The test period is challenging**: 2020-2024 includes unprecedented volatility (352 → 1,507 NGN)

---

## Requirements

- Python 3.8+
- NumPy
- Pandas
- SciPy
- Scikit-learn
- Statsmodels
- Matplotlib
- Seaborn
- tqdm

---

## Citation

```
@thesis{ike2025usdngn,
  author = {Oche Emmanuel Ike},
  title = {USD-NGN Exchange Rate Forecasting Using Information Theory,
           Hybrid Machine Learning and Explainable AI},
  school = {International Institute for Financial Engineering},
  year = {2025},
  type = {Master's Thesis},
  id = {242220011}
}
```

---

## License

For academic use only. All data derived from publicly available official sources.
