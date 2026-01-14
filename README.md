# USD-NGN Exchange Rate Forecasting System

A hybrid machine learning system for forecasting USD-NGN exchange rates that **beats the Random Walk baseline** on both RMSE and Directional Accuracy metrics.

---

## Results Summary

| Model | RMSE | DA | Status |
|-------|------|-----|--------|
| Random Walk | 24.83 | 50.9% | Baseline |
| ARIMA | 137.18 | 51.6% | DA only |
| Hybrid ARIMA-LSTM | 49.86 | 57.1% | DA only |
| **Winning Hybrid** | **24.61** | **51.6%** | **BEATS RW** |
| **Improved Hybrid** | **24.64** | **51.6%** | **BEATS RW** |

The Winning Hybrid and Improved Hybrid models use a combination of:
- **Mean Reversion**: Price tends to revert toward the 20-day moving average
- **Contrarian Strategy**: Predicts opposite direction after consecutive same-direction days

---

## Quick Start

```bash
# Run the complete pipeline
python run_pipeline.py
```

This will:
1. Generate/collect exchange rate data (1,096 observations)
2. Preprocess and engineer 27 features
3. Split data (70% train / 15% validation / 15% test)
4. Train 5 models (Random Walk, ARIMA, Hybrid ARIMA-LSTM, Winning Hybrid, Improved Hybrid)
5. Evaluate and compare all models
6. Save results to `data/evaluation_metrics.csv`

---

## Project Structure

```
usdngn_forecasting/
├── run_pipeline.py          # Main entry point
├── requirements.txt         # Python dependencies
├── README.md
│
├── src/                     # Core modules
│   ├── __init__.py
│   ├── data_collection.py   # Data generation/collection
│   ├── preprocessing.py     # Feature engineering & data splitting
│   ├── models.py            # ARIMA, LSTM, Hybrid ARIMA-LSTM
│   ├── hybrid_model.py      # Winning & Improved Hybrid models
│   ├── evaluation.py        # Metrics computation
│   ├── information_analysis.py  # Transfer entropy analysis
│   └── visualization.py     # Plotting utilities
│
├── data/                    # Generated CSV files
│   ├── raw_data.csv
│   ├── processed_data.csv
│   ├── train_data.csv
│   ├── val_data.csv
│   ├── test_data.csv
│   └── evaluation_metrics.csv
│
├── models/                  # Saved model files
└── figures/                 # Generated plots
```

---

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Run Full Pipeline
```python
from run_pipeline import run_pipeline

results = run_pipeline(verbose=True)

# Access results
models = results['models']
predictions = results['predictions']
metrics = results['metrics']
```

### Use Individual Components
```python
from src.data_collection import DataCollector
from src.preprocessing import DataPreprocessor, DataSplitter
from src.models import ARIMAModel
from src.hybrid_model import WinningHybridModel
from src.evaluation import ModelEvaluator

# Collect data
collector = DataCollector(start_date='2023-01-01', end_date='2025-12-31')
raw_data = collector.collect_all_data()

# Preprocess
preprocessor = DataPreprocessor(raw_data)
processed_data, stats = preprocessor.preprocess()

# Split
splitter = DataSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
train, val, test = splitter.split(processed_data)

# Train model
model = WinningHybridModel()
model.fit(X_train, y_train, X_val, y_val)
predictions = model.predict(X_test, y_train)

# Evaluate
metrics = ModelEvaluator.compute_all_metrics(y_test, predictions)
```

---

## Thesis Context

**Title**: USD-NGN Exchange Rate Forecasting Using Information Theory, Hybrid Machine Learning and Explainable AI

**Author**: Oche Emmanuel Ike (ID: 242220011)

**Key Contributions**:
1. Hybrid models that beat Random Walk baseline on both RMSE and Directional Accuracy
2. Mean reversion + contrarian strategy combination
3. Transfer entropy for causality analysis
4. Comprehensive evaluation framework

---

## License

For academic use only.
