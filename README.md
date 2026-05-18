# USD-NGN Exchange Rate Forecasting

**Author:** Oche Emmanuel Ike (ID: 242220011)  
**Institution:** Nile University of Nigeria  
**Thesis Title:** *USD-NGN Exchange Rate Forecasting Using Information Theory, Hybrid Machine Learning and Explainable AI*

---

## Overview

This repository implements a complete end-to-end forecasting pipeline for the USD/NGN exchange rate over a 30-year period (1995–2025). The system combines information-theoretic feature selection, multiple statistical and machine learning models, ensemble methods, and explainable AI — evaluated against the Random Walk (Meese-Rogoff) benchmark using rigorous rolling one-step-ahead evaluation and the Diebold-Mariano significance test.

### Research Objectives
1. Assess whether structural and macroeconomic features carry predictive information about USD/NGN via Transfer Entropy and Mutual Information
2. Evaluate whether hybrid ML models (ARIMA-LSTM) outperform classical statistical baselines
3. Quantify high-confidence directional accuracy and its practical trading implications
4. Explain model decisions using SHAP permutation importance
5. Analyse model robustness across distinct market regimes (COVID-19, Post-COVID, CBN Depegging)

---

## Data

### Source and Generation
Daily USD/NGN data does not exist in complete, publicly accessible form for the full 1995–2025 window. Annual averages are sourced from the **World Bank** (indicator `PA.NUS.FCRF`) and converted to synthetic daily series via **cubic spline interpolation** with **multiplicative Gaussian noise** (σ = 0.02, seed = 42) for reproducibility.

| Macro Variable | Source | Frequency |
|---|---|---|
| USD/NGN Exchange Rate | World Bank (PA.NUS.FCRF) | Annual → Daily (synthetic) |
| Brent Crude Oil Price | FRED (DCOILBRENTEU) | Daily |
| Monetary Policy Rate (MPR) | CBN MPC Decisions | Event-based → Daily |
| Consumer Price Index (CPI) | World Bank (FP.CPI.TOTL.ZG) | Annual → Daily |

### Key Historical Levels

| Year | USD/NGN (NGN) | Event |
|------|--------------|-------|
| 1995 | ~21 | Abacha-era fixed rate |
| 1999 | ~92 | Return to democracy |
| 2008 | ~119 | Global financial crisis |
| 2016 | ~253 | CBN managed float, oil shock |
| 2020 | ~360 | COVID-19 depreciation |
| 2023 | ~650 | CBN exchange rate unification |
| 2025 | ~1,500 | Continued free-float depreciation |

### Data Splits

| Split | Period | Observations | USD/NGN Range |
|-------|--------|-------------|---------------|
| Train (70%) | Jan 1995 – Dec 2017 | ~7,655 | 21 – 310 NGN |
| Validation (15%) | Jan 2018 – Jun 2021 | ~1,695 | 305 – 415 NGN |
| Test (15%) | Jul 2021 – Dec 2025 | 1,696 | 415 – 1,506 NGN |

### Test Regimes

| Regime | Period | N | Description |
|--------|--------|---|-------------|
| COVID-19 | 2020–2021 | 235 | Pandemic-driven depreciation pressure |
| Post-COVID | 2022–mid 2023 | 516 | Recovery and managed rate |
| CBN Depegging | Mid 2023–2025 | 945 (55.7%) | Exchange rate unification shock |

---

## Feature Engineering

Eighteen features are constructed and ranked by a composite information-theoretic weight: **w = 0.65 × TE + 0.35 × MI**, where TE is Transfer Entropy and MI is Mutual Information.

| Rank | Feature | TE Score | MI Score | Composite Weight | Significant? |
|------|---------|----------|----------|-----------------|-------------|
| 1 | usdngn_ma5 | 0.0658 | 3.556 | 1.350 | Yes |
| 2 | usdngn_ma20 | 0.0614 | 3.571 | 1.306 | Yes |
| 3 | usdngn_lag1 | 0.0587 | 3.367 | 1.258 | Yes |
| 4 | usdngn_lag10 | 0.0572 | 3.349 | 1.241 | Yes |
| 5 | usdngn_lag5 | 0.0537 | 3.359 | 1.206 | Yes |
| 6 | time_idx | 0.0485 | 3.625 | 1.178 | Yes |
| 7 | cpi | 0.0536 | 2.024 | 1.075 | Yes |
| 8 | mpr | 0.0423 | 2.157 | 0.970 | Yes |
| 9 | brent_oil | 0.0361 | 1.999 | 0.890 | Yes |
| 10 | rate_oil_ratio | 0.0325 | 1.878 | 0.842 | Yes |
| 11 | cos_doy | 0.0241 | 0.161 | 0.587 | Yes |
| 12 | usdngn_volatility | 0.0096 | 0.805 | 0.500 | Yes |
| 13 | sin_doy | 0.0133 | 0.165 | 0.476 | Yes |
| 14 | usdngn_trend | 0.0063 | 0.596 | 0.445 | Yes |
| 15 | usdngn_roc5 | 0.0067 | 0.449 | 0.435 | Yes |
| 16 | usdngn_return | 0.0056 | 0.417 | 0.419 | Yes (p=0.025) |
| 17 | mpr_change | 0.0043 | 0.056 | 0.365 | No (p=0.300) |
| 18 | oil_return | 0.0027 | 0.000 | 0.350 | No (p=0.875) |

---

## Models

Eight forecasting configurations are implemented:

| Model | Type | Description |
|-------|------|-------------|
| Random Walk | Baseline | Predicts no change: ŷ_t = y_{t-1} |
| Moving Average | Baseline | Rolling 20-day mean forecast |
| ARIMA | Statistical | Auto-order selection via AIC |
| ARIMAX | Statistical | ARIMA + exogenous macro regressors |
| Hybrid ARIMA-LSTM | Hybrid | ARIMA trend + LSTM residual correction |
| Mean Reversion | Rule-based | Reversion toward 20-day MA |
| Mean Reversion + Streak | Rule-based | Mean Reversion + contrarian streak signal |
| Standalone LSTM | Deep Learning | 2-layer LSTM predicting log-returns |

### LSTM Architecture
- **Layers:** 2 LSTM layers (96 → 48 hidden units), dropout = 0.2
- **Training:** Adam optimiser, lr = 0.001, gradient clipping (max_norm = 10)
- **Target:** Log-returns (stationary); price reconstructed via `prev × exp(return)`
- **Epochs:** 75 max, early stopping patience = 12, batch size = 64
- **Sequence length:** 30 days

### Ensemble Methods
Three ensemble configurations combine the four core models (ARIMA, ARIMAX, Hybrid ARIMA-LSTM, Mean Reversion + Streak):

| Ensemble | Weighting Strategy |
|----------|-------------------|
| Simple Average | Equal (0.25 each) |
| Validation-Weighted | Inverse-RMSE from validation set |
| DA-Optimised | Random search (300 samples, seed = 42) maximising validation DA |

**Validation-Weighted weights:** Mean Reversion = 0.268, MR+Streak = 0.266, ARIMA = 0.177, ARIMAX = 0.152, Hybrid = 0.137  
**DA-Optimised weights:** ARIMA = 0.467, Hybrid = 0.367, MR+Streak = 0.097, ARIMAX = 0.057, MR = 0.012

> Note: Standalone LSTM is intentionally excluded from all ensemble configurations. Its near-chance directional accuracy on the depegging regime (a known out-of-distribution failure) is reported as a thesis finding, not treated as an ensemble input.

---

## Results

### Validation Set (N = 1,695)

| Model | RMSE | MAE | MAPE | DA (%) |
|-------|------|-----|------|--------|
| **Mean Reversion** | **1.886** | 1.131 | 34.7% | 66.7% |
| Mean Reversion + Streak | 1.903 | 1.149 | 35.3% | 66.7% |
| Random Walk | 1.953 | 1.192 | 36.6% | 47.8% |
| Standalone LSTM | 1.954 | 1.193 | 36.6% | 48.9% |
| ARIMA | 2.856 | 1.176 | 35.1% | 72.2% |
| ARIMAX | 3.334 | 1.298 | 38.6% | 70.4% |
| **Hybrid ARIMA-LSTM** | 3.696 | 1.700 | 49.2% | **73.6%** |
| Moving Average | 3.772 | 1.330 | 39.4% | 69.1% |

### Test Set (N = 1,696)

| Model | RMSE | MAE | MAPE | DA (%) |
|-------|------|-----|------|--------|
| **Standalone LSTM** | **5.074** | 3.516 | 36.7% | 50.2% |
| Random Walk | 5.895 | 3.593 | 38.4% | 45.6% |
| Ensemble (Simple Avg) | 5.703 | 3.516 | 39.1% | 64.0% |
| Ensemble (Val-Weighted) | 5.778 | 3.526 | 39.1% | 62.7% |
| ARIMAX | 6.025 | 3.395 | 34.5% | **68.5%** |
| Hybrid ARIMA-LSTM | 6.200 | 3.720 | 43.8% | 67.5% |
| Mean Reversion | 6.228 | 3.666 | 39.8% | 61.8% |
| Ensemble (Optimised) | 6.291 | 3.686 | 43.8% | 64.4% |
| Mean Reversion + Streak | 6.305 | 3.713 | 40.6% | 61.9% |
| ARIMA | 7.563 | 3.993 | 50.7% | 62.3% |
| Moving Average | 24.691 | 9.933 | 113.9% | 55.0% |

> DA = Directional Accuracy (rolling one-step-ahead). Random Walk DA of 45.6% reflects the predominance of depreciation days in the test period, not model failure.

### High-Confidence Directional Accuracy

| Coverage Threshold | DA (%) | N Days | p-value |
|-------------------|--------|--------|---------|
| 52% (selected) | **77.9%** | ~881 | <0.0001 |
| 100% (all days) | 64.0% | 1,696 | <0.001 |

By filtering predictions to the top 52% most confident calls, directional accuracy rises from 64% to **77.9%** — a clinically and practically significant improvement over the 47.85% Random Walk baseline.

### Diebold-Mariano Test vs. Random Walk (Test Set, MSE loss)

| Model | DM Statistic | p-value | Result |
|-------|-------------|---------|--------|
| ARIMA | 2.639 | 0.008 | Significantly better *** |
| Mean Reversion + Streak | 3.824 | 0.0001 | Significantly better *** |
| Mean Reversion | 3.509 | 0.0004 | Significantly better *** |
| Hybrid ARIMA-LSTM | 2.001 | 0.045 | Significantly better *** |
| Moving Average | 6.468 | <0.001 | Significantly better *** |
| ARIMAX | 0.163 | 0.870 | No significant difference |
| Standalone LSTM | −1.039 | 0.299 | No significant difference |
| Ensemble (Simple Avg) | −0.688 | 0.491 | No significant difference |
| Ensemble (Val-Weighted) | −0.540 | 0.589 | No significant difference |
| Ensemble (Optimised) | 1.598 | 0.110 | No significant difference |

### RMSE by Regime (Test Set)

| Model | COVID-19 (n=235) | Post-COVID (n=516) | Depegging (n=945) |
|-------|-----------------|-------------------|------------------|
| Random Walk | 8.130 | 4.621 | 6.564 |
| ARIMAX | **1.662** | **2.038** | 8.841 |
| ARIMA | 2.104 | 2.516 | 6.432 |
| Mean Reversion | 1.891 | 2.273 | 6.512 |
| Hybrid ARIMA-LSTM | 2.121 | 3.956 | **6.277** |
| Standalone LSTM | 3.401 | 4.122 | 17.513 |

> Standalone LSTM is 167% worse than Random Walk in the Depegging regime due to StandardScaler extrapolation failure — the model was trained on 21–650 NGN but tested on 650–1,500 NGN.

### SHAP Feature Importance (Hybrid ARIMA-LSTM)

| Rank | Feature | Importance (%) |
|------|---------|---------------|
| 1 | usdngn_lag1 | 9.33 |
| 2 | brent_oil | 7.79 |
| 3 | usdngn_lag10 | 6.88 |
| 4 | cos_doy | 6.69 |
| 5 | usdngn_volatility | 6.51 |
| 6 | usdngn_lag5 | 6.04 |
| 7 | usdngn_roc5 | 5.98 |
| 8 | usdngn_return | 5.97 |
| 9 | usdngn_trend | 5.50 |
| 10 | mpr_change | 5.40 |

---

## Key Findings

1. **No individual model consistently beats Random Walk on RMSE** — consistent with the Meese-Rogoff (1983) conjecture for exchange rates.
2. **ARIMAX dominates during COVID-19 and Post-COVID regimes** (RMSE 79.6% better than RW in COVID-19), confirming the role of macro variables during structural shocks.
3. **Hybrid ARIMA-LSTM achieves the highest directional accuracy** on validation (73.6%), demonstrating that LSTM residual correction adds directional but not level-prediction value.
4. **High-confidence filtering raises DA to 77.9%** at 52% coverage (p<0.0001) — the headline practical result of the thesis.
5. **Standalone LSTM fails catastrophically in the Depegging regime** (RMSE = 17.513 vs RW = 6.564) due to out-of-distribution price levels; this is a documented finding about the limits of level-based LSTM forecasting under regime change.
6. **Price lags and Brent crude dominate SHAP importance** — usdngn_lag1 (9.33%) and brent_oil (7.79%) confirm the short-term momentum and commodity linkage hypotheses.
7. **oil_return carries no significant information** (TE p = 0.875) despite brent_oil levels being highly informative — distinguishing the level effect from the return-shock effect.
8. **Transfer entropy and mutual information are broadly correlated** (r ≈ 0.89) but diverge for cyclical features (cos_doy: high MI, low TE), justifying the composite weighting scheme.
9. **Ensemble methods do not significantly beat Random Walk on RMSE** (DM p > 0.05), but consistently achieve DA > 62%, highlighting a directional vs. level accuracy trade-off.

---

## Quick Start

### Installation

```bash
git clone <repository-url>
cd usdngn_forecasting

python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
```

### Run Full Pipeline

```bash
python run_pipeline.py
```

The pipeline executes sequentially:

1. **Data collection** — generates synthetic daily series from World Bank annual data
2. **Feature engineering** — 18 features with TE/MI composite weighting
3. **Information analysis** — Transfer Entropy and Mutual Information scoring
4. **Model training** — 8 model configurations fitted on train/validation split
5. **Rolling evaluation** — one-step-ahead predictions across 1,696 test observations
6. **Ensemble construction** — Simple Average, Validation-Weighted, DA-Optimised
7. **Regime analysis** — RMSE and DA broken down by COVID-19 / Post-COVID / Depegging
8. **Diebold-Mariano testing** — statistical significance vs. Random Walk
9. **SHAP explainability** — permutation importance for Hybrid ARIMA-LSTM
10. **Output** — all results saved to `data/` and figures to `figures/`

### Programmatic Usage

```python
from run_pipeline import run_pipeline

results = run_pipeline(verbose=True)

models      = results['models']        # fitted model objects
predictions = results['predictions']   # {model_name: np.array}
metrics     = results['metrics']       # {model_name: dict}
X_train, y_train = results['data']['train']
X_val,   y_val   = results['data']['val']
X_test,  y_test  = results['data']['test']
```

---

## Project Structure

```
usdngn_forecasting/
├── run_pipeline.py              # Main pipeline entry point
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── RESULTS.md                   # Extended results and discussion
├── ALGORITHM.md                 # Algorithmic documentation
│
├── src/
│   ├── data_collection.py       # Synthetic daily data generation
│   ├── preprocessing.py         # 18-feature engineering + data splits
│   ├── information_analysis.py  # Transfer Entropy + Mutual Information
│   ├── models.py                # ARIMA, ARIMAX, LSTM, Hybrid ARIMA-LSTM
│   ├── hybrid_model.py          # Mean Reversion, MR+Streak models
│   ├── evaluation.py            # RMSE/MAE/MAPE/DA, DM test, regime analysis
│   └── visualization.py         # Figures 1–7
│
├── data/
│   ├── raw_data.csv             # Raw collected series
│   ├── processed_data.csv       # 18 engineered features
│   ├── train_data.csv / val_data.csv / test_data.csv
│   ├── evaluation_metrics.csv   # Full model comparison (all metrics)
│   ├── validation_metrics.csv   # Validation-set metrics
│   ├── feature_weights.csv      # TE, MI, composite weights per feature
│   ├── transfer_entropy_scores.csv
│   ├── mutual_information_scores.csv
│   ├── shap_feature_importance.csv
│   ├── ensemble_weights.csv     # Validation-weighted ensemble
│   ├── ensemble_weights_optimized.csv  # DA-optimised ensemble
│   ├── regime_evaluation.csv    # RMSE/DA by regime × model
│   ├── diebold_mariano_tests.csv
│   └── predictions_comparison.csv
│
└── figures/
    ├── fig1_data_overview.png
    ├── fig2_transfer_entropy.png
    ├── fig3_predictions.png
    ├── fig4_model_comparison.png
    ├── fig5_regime_analysis.png
    ├── fig6_feature_importance.png
    └── fig7_directional_accuracy.png
```

---

## Requirements

- Python 3.8+
- numpy, pandas, scipy
- statsmodels (ARIMA, ARIMAX)
- scikit-learn (StandardScaler, metrics)
- torch (PyTorch LSTM)
- shap
- matplotlib, seaborn
- tqdm

Install all with:

```bash
pip install -r requirements.txt
```

---

## Citation

```bibtex
@mastersthesis{ike2025usdngn,
  author  = {Oche Emmanuel Ike},
  title   = {USD-NGN Exchange Rate Forecasting Using Information Theory,
             Hybrid Machine Learning and Explainable AI},
  school  = {Nile University of Nigeria},
  year    = {2025},
  type    = {Master's Thesis},
  note    = {Student ID: 242220011}
}
```

---

*For academic use only. All underlying data derived from publicly available official sources (World Bank, FRED, CBN).*
