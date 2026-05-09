# Chapter 3: METHODOLOGY

## 3.1 Research Design and Overview

This research employs a systematic four-stage mixed-method approach combining information-theoretic analysis with machine learning forecasting techniques. The research design integrates traditional econometric foundations with modern computational methods, guided by causality-aware principles to forecast USD-NGN exchange rates. The overall architecture follows:

**Stage 1: Data Collection and Preprocessing** — Systematic collection of historical exchange rates and macroeconomic indicators from verified sources, with appropriate frequency harmonization and feature engineering.

**Stage 2: Information Flow Analysis** — Computation of transfer entropy and mutual information metrics to identify causal relationships and directional dependencies among economic variables, resulting in information-theoretic feature weighting.

**Stage 3: Hybrid Model Development** — Implementation of multiple forecasting architectures ranging from traditional econometric models to deep learning approaches, culminating in an information-weighted hybrid ARIMA-LSTM framework with explainability integration.

**Stage 4: Comprehensive Evaluation** — Rigorous benchmarking against baseline models using multiple performance metrics and statistical significance testing via the Diebold-Mariano test.

This systematic approach ensures that each subsequent stage builds upon validated findings from the previous stage, maintaining methodological rigor throughout the research process.

---

## 3.2 Data Collection and Preprocessing

### 3.2.1 Data Sources and Variables

The research utilizes daily observations spanning 30 years (January 1, 1995 to December 31, 2024), encompassing multiple critical economic periods in Nigeria's recent history. Primary data sources are detailed in **Table 3.1**.

**Table 3.1: Data Sources and Primary Variables**

| Variable | Frequency | Source | Type | Period |
|----------|-----------|--------|------|--------|
| **USD-NGN Exchange Rate** | Daily | CBN Official Rates | Target | Jan 1995 – Dec 2024 |
| **Brent Crude Oil Price** | Daily | OPEC/World Bank/FRED | Predictor | Jan 1995 – Dec 2024 |
| **CBN Policy Interest Rate (MPR)** | Monthly (forward-filled) | CBN MPC Decisions | Predictor | Jan 1995 – Dec 2024 |
| **Consumer Price Index (CPI)** | Monthly (forward-filled) | National Bureau of Statistics | Predictor | Jan 1995 – Dec 2024 |

The study period captures six distinct economic regimes critical to understanding Nigerian exchange rate dynamics:

1. **Return to Democracy & Liberalization (1998-2000)**: Transition from fixed to managed float regime
2. **Oil Price Boom (2010-2014)**: Sustained high commodity prices and strong naira
3. **Oil Price Crash & Recession (2014-2016)**: Sharp commodity decline, foreign exchange crisis
4. **Recovery and Consolidation (2017-2019)**: Gradual economic stabilization
5. **COVID-19 Pandemic Impact (2020-2022)**: External shock effects on currency dynamics
6. **Post-Pandemic Depegging (2023-2024)**: Central Bank naira depegging, historic depreciation

This periodization ensures model robustness across varying economic conditions and policy regimes.

### 3.2.2 Data Alignment and Preprocessing

**Frequency Harmonization**: Monthly variables (MPR, CPI) were forward-filled to daily frequency, creating step functions that change only on announcement/release dates. This preserves the temporal structure of market information availability and prevents artificial data interpolation bias.

**Missing Data Treatment**: 
- Weekend and holiday gaps in exchange rate data were forward-filled
- Linear interpolation applied to isolated missing values (<3 consecutive days)
- Periods with >5 consecutive missing observations were excluded from analysis
- Final dataset: 10,958 observations with zero missing values

**Data Validation**: All data were cross-referenced against official sources (CBN Monetary Policy Committee reports, CBN exchange rate databases, FRED, OPEC). Validation confirmed <0.1% maximum deviation between collected values and official records.

### 3.2.3 Feature Engineering

From raw variables, 27 features were systematically engineered to capture different dimensions of exchange rate dynamics. **Table 3.2** presents the feature taxonomy.

**Table 3.2: Engineered Features by Category**

| Category | Features | Count | Equations |
|----------|----------|-------|-----------|
| **Returns** | USD-NGN returns, Oil returns | 2 | $r_t = \ln(y_t/y_{t-1})$ |
| **Moving Averages** | MA(5), MA(20), MA(60) for exchange rate | 3 | $MA_t^{(h)} = \frac{1}{h}\sum_{k=0}^{h-1} y_{t-k}$ |
| **Volatility** | Rolling std. dev. (20-day window) | 1 | $\sigma_t = \sqrt{\frac{1}{19}\sum_{k=0}^{19}(r_{t-k}-\bar{r})^2}$ |
| **Lagged Prices** | Lags 1-5 days for exchange rate | 5 | $y_{t-k}, k \in \{1,2,3,4,5\}$ |
| **Oil Linkage** | Rate/Oil ratio, Oil returns | 2 | $ratio_t = y_t / o_t$ |
| **Monetary Policy** | MPR level, MPR changes | 2 | $\Delta MPR_t = MPR_t - MPR_{t-1}$ |
| **Inflation Dynamics** | CPI level, CPI momentum | 2 | $momentum_t = \ln(CPI_t/CPI_{t-252})$ |
| **Rate of Change** | 5-day, 10-day ROC | 2 | $ROC_t^{(h)} = \frac{y_t - y_{t-h}}{y_{t-h}}$ |
| **Additional** | Oil-MPR interaction, Volatility-MA ratio | 2 | Computed interactions |
| **Total** | | **27** | |

**[TABLE 3.2 PLACEMENT NOTE: Insert feature engineering mathematics]**

Feature engineering prioritized both statistical and economic interpretability. Each feature captures distinct aspects: momentum indicators (moving averages), extreme events (volatility), structural relationships (oil ratio), and policy transmission (MPR effects).

### 3.2.4 Look-Ahead Bias Prevention

A critical methodological contribution involves the systematic elimination of look-ahead bias. All predictor variables (excluding naturally lagged features) were shifted forward by one period. This ensures:

- ARIMA models forecast $y_{t+1}$ using information available at time $t$
- No future information leaks into training processes
- Realistic evaluation reflecting actual deployment conditions

The data split structure (chronological 70/15/15) preserved time series ordering without shuffling.

---

## 3.3 Information Flow Analysis

### 3.3.1 Transfer Entropy Computation

Transfer entropy quantifies directed information flow from source variable $X$ to target variable $Y$, measuring how much knowledge of the past of $X$ reduces uncertainty about the future of $Y$.

**Mathematical Definition**:

$$TE(X \to Y) = \sum_{y_{t+1}, y_t^{(k)}, x_t^{(k)}} p(y_{t+1}, y_t^{(k)}, x_t^{(k)}) \log \frac{p(y_{t+1} | y_t^{(k)}, x_t^{(k)})}{p(y_{t+1} | y_t^{(k)})}$$

where:
- $y_{t+1}$ is the next state of target variable
- $y_t^{(k)}$ is the $k$-step history of target variable  
- $x_t^{(k)}$ is the $k$-step history of source variable
- $p(\cdot)$ denotes probability distribution

**Implementation Details**:

1. **Discretization**: Variables discretized into 6 equal-frequency bins (validated via cross-entropy minimization). This balances between information loss and statistical robustness.

2. **Statistical Testing**: Bootstrap significance testing with 1,000 resamples at $\alpha = 0.05$. For each feature, the null hypothesis (no directional information flow) is tested by shuffling the source variable and recomputing TE against the permutation distribution.

3. **Lag Selection**: $k=1$ lag selected based on preliminary autocorrelation analysis, capturing immediate dependencies while avoiding computational complexity.

4. **Rolling Windows**: Analysis conducted on 250-day rolling windows with 20-day step size to detect regime-dependent changes in information flows.

**Bivariate Pairs Analyzed**: 
- Oil prices → USD-NGN
- Monetary Policy Rate → USD-NGN  
- CPI → USD-NGN
- Reverse directions for bidirectional assessment

### 3.3.2 Mutual Information Analysis

Mutual information captures symmetric, nonlinear dependencies between variables:

$$MI(X; Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$$

Unlike transfer entropy (directional), MI measures total statistical dependence regardless of direction.

**Implementation**: Scikit-learn's `mutual_info_regression` with 10 nearest neighbors density estimator. This non-parametric approach effectively captures nonlinear relationships without assuming functional form.

### 3.3.3 Feature Weight Derivation

Transfer entropy and mutual information scores were combined into unified feature weights through a two-stage process:

**Stage 1 – Normalization**:
$$\widetilde{TE}_i = \frac{TE_i - \min(TE)}{\max(TE) - \min(TE) + \varepsilon}$$
$$\widetilde{MI}_i = \frac{MI_i - \min(MI)}{\max(MI) - \min(MI) + \varepsilon}$$

**Stage 2 – Combination with Significance Weighting**:
$$c_i = \alpha \cdot \widetilde{TE}_i + (1-\alpha) \cdot \widetilde{MI}_i$$

where $\alpha = 0.6$ (TE weighted more heavily as it captures causality)

$$\text{weight}_i = \min_w + (\max_w - \min_w) \cdot \text{normalize}(c_i) \cdot (1 - p_i)$$

where $p_i$ is the TE p-value, and bounds are $[\min_w, \max_w] = [0.35, 1.35]$.

Features with higher transfer entropy and mutual information receive larger weights, guiding the hybrid model to prioritize economically meaningful relationships.

---

## 3.4 Model Development

### 3.4.1 Benchmark Models

**Random Walk Model**: Predicts next exchange rate as current rate: $\hat{y}_{t+1|t} = y_t$. This is the reference standard in exchange rate literature, as the efficient market hypothesis suggests exchange rates follow martingales.

**Moving Average Model**: Forecasts based on 20-day rolling mean:
$$\hat{y}_{t+1|t} = \frac{1}{20}\sum_{k=0}^{19} y_{t-k}$$

A 20-day window was selected based on cross-validation over training set.

### 3.4.2 Econometric Models

**ARIMA(p,d,q)**: Autoregressive Integrated Moving Average fitted via automatic order selection using Akaike Information Criterion (AIC) with maximum orders $(5,2,5)$. This provides linear temporal structure baseline.

**ARIMAX(p,d,q)**: ARIMA extended with exogenous variables (oil prices, MPR, CPI). This tests whether macroeconomic variables improve forecasts within a classical econometric framework.

### 3.4.3 Deep Learning Models

**LSTM Architecture**:

```
Input Layer: 60-day sequences × 27 weighted features
↓
LSTM Layer 1: 128 units, tanh activation, 0.2 dropout
↓
LSTM Layer 2: 64 units, tanh activation, 0.2 dropout
↓
Dense Layer: 32 units, ReLU activation
↓
Output Layer: 1 neuron, linear activation
```

**Training Configuration**:
- Optimizer: Adam (lr=0.001, β₁=0.9, β₂=0.999)
- Loss: Mean Squared Error + L2 regularization (λ=0.01)
- Batch size: 32 sequences
- Early stopping: patience=20 epochs on validation loss
- Feature scaling: StandardScaler applied to all inputs

**GRU Model**: Similar architecture but with GRU units instead of LSTM cells, providing a computationally lighter alternative.

### 3.4.4 Hybrid ARIMA-LSTM Model with Information Weighting

The proposed hybrid model combines linear and nonlinear forecasting strengths through a three-stage architecture:

**Stage 1 – Linear Component**:
Fit ARIMA$(p,d,q)$ to the target series to extract level forecasts and residuals:
$$\hat{y}_{t+1|t}^{A} = f_{ARIMA}(y_{1:t})$$
$$e_t = y_t - \hat{y}_{t|t-1}^{A}$$

**Stage 2 – Information-Weighted Feature Matrix**:
Apply information-theoretic weights to exogenous features:
$$\tilde{x}_t^{(i)} = w_i \cdot x_t^{(i)}$$

where $w_i$ derived from transfer entropy/mutual information analysis.

**Stage 3 – Nonlinear Residual Modeling**:
Train LSTM on residual dynamics:
$$\mathbf{u}_t = [\tilde{x}_t; e_{t-5:t}]$$ (weighted features + residual history)
$$\hat{r}_{t+1|t} = f_{LSTM}(\mathbf{u}_{1:t})$$

**Final Hybrid Prediction**:
$$\hat{y}_{t+1|t} = \lambda(s_t) \cdot \hat{y}_{t+1|t}^{A} + (1-\lambda(s_t)) \cdot \hat{r}_{t+1|t}$$

where $\lambda(s_t)$ is regime-dependent blending coefficient:
- Calm regime: $\lambda = 0.8$ (trust ARIMA trend)
- Volatile regime: $\lambda = 0.5$ (balanced)
- Shock regime: $\lambda = 0.3$ (trust LSTM nonlinearity)

Regime classification based on 20-day rolling volatility thresholds.

### 3.4.5 Mean Reversion Models

**Mean Reversion Model**: Predicts reversion toward 20-day moving average with mean-reversion strength parameter $\kappa$:
$$\hat{y}_{t+1|t} = y_t + \kappa \cdot (MA_{20,t} - y_t)$$

**Mean Reversion + Streak Detection**: Enhanced variant incorporating trend persistence detection. If consecutive price movements in same direction exceed threshold, reduces reversion strength (allowing trends to persist).

---

## 3.5 Explainable AI Integration

### 3.5.1 SHAP Feature Importance

SHapley Additive exPlanations (SHAP) values were computed to provide interpretable feature importance. Since LSTMs are not natively tree-based, a gradient boosting surrogate model (XGBoost) was trained on LSTM input-output pairs for 1,000 holdout samples.

**SHAP Implementation**:
1. Train XGBoost surrogate capturing LSTM behavior
2. Compute SHAP values via TreeExplainer
3. Aggregate to global feature importance rankings
4. Extract local explanations for individual predictions

This provides:
- **Global insights**: Feature importance rankings across entire test set
- **Local insights**: Case-specific contributions explaining individual predictions
- **Regime-specific analysis**: Feature importance variation across market conditions

**[FIGURE PLACEMENT: Insert SHAP feature importance bar chart here]**

---

## 3.6 Benchmarking and Evaluation Framework

### 3.6.1 Benchmark Models Suite

Models evaluated across four categories:

1. **Statistical Baselines**: Random Walk, Moving Average
2. **Econometric Models**: ARIMA, ARIMAX
3. **Deep Learning Models**: LSTM, GRU
4. **Proposed Approaches**: Hybrid ARIMA-LSTM, Mean Reversion variants
5. **Ensemble Methods**: Simple average, validation-weighted, optimized weighting

### 3.6.2 Evaluation Metrics

**Accuracy Metrics** (lower is better):

$$RMSE = \sqrt{\frac{1}{n}\sum_{t=1}^{n}(y_t - \hat{y}_t)^2}$$

$$MAE = \frac{1}{n}\sum_{t=1}^{n}|y_t - \hat{y}_t|$$

$$MAPE = \frac{1}{n}\sum_{t=1}^{n}\left|\frac{y_t - \hat{y}_t}{y_t}\right| \times 100\%$$

**Directional Metric** (higher is better):

$$DA = \frac{1}{n}\sum_{t=1}^{n}\mathbb{1}\{\text{sign}(\hat{y}_{t+1} - y_t) = \text{sign}(y_{t+1} - y_t)\}$$

where $DA$ represents percentage of correct direction predictions.

### 3.6.3 Statistical Significance Testing

**Diebold-Mariano Test**: Pairwise comparison of forecast accuracy between models.

$$DM = \frac{\bar{d}}{\sqrt{\hat{\sigma}_d^2/n}} \sim N(0,1)$$

where $\bar{d}$ is mean loss differential and $\hat{\sigma}_d^2$ is heteroskedasticity-consistent variance estimate.

Critical region: reject null hypothesis (equal accuracy) if $|DM| > 1.96$ at $\alpha = 0.05$.

### 3.6.4 Evaluation Protocol

Rolling one-step-ahead forecast protocol prevents look-ahead bias and reflects real deployment conditions:

```python
for t in range(n_train, n_train + n_test):
    # Fit model on data up to time t
    model.fit(X_train:t, y_train:t)
    # Generate one-step-ahead forecast
    y_hat_t+1 = model.predict(X_t+1)
    # Store forecast and evaluate
    store_results(y_hat_t+1, y_t+1)
```

This rolling window approach ensures:
- No future information leakage
- Sequential updating of model parameters
- Realistic assessment of forecast performance

---

## 3.7 Implementation Tools and Environment

**Programming Language**: Python 3.10+

**Core Libraries**:
- Data handling: pandas, numpy
- Time series analysis: statsmodels (ARIMA, stationarity tests)
- Machine learning: scikit-learn, torch (LSTM/GRU)
- Information theory: custom transfer entropy implementation
- Explainability: shap, xgboost
- Evaluation: scikit-learn metrics, scipy.stats (Diebold-Mariano)
- Visualization: matplotlib, seaborn

**Computing Environment**:
- Development: Local workstation (16GB RAM, multi-core CPU)
- Execution: WSL2 with Python virtual environment
- Version control: Git for code management

---

## 3.8 Data Processing Pipeline Summary

**Table 3.3: Data Processing Pipeline Steps**

| Stage | Input | Processing | Output | Records |
|-------|-------|-----------|--------|---------|
| **Raw Collection** | CBN, OPEC, NBS APIs | Download & verify | raw_data.csv | 10,958 |
| **Preprocessing** | raw_data.csv | Handle missing, align frequency | processed_data.csv | 10,958 |
| **Feature Engineering** | processed_data.csv | Compute 27 features, shift predictors | engineered_features.csv | 10,937 |
| **Information Analysis** | engineered_features.csv | TE/MI computation, weight derivation | feature_weights.csv | 27 weights |
| **Train/Val/Test Split** | engineered_features.csv | Chronological 70/15/15 | train_data, val_data, test_data | 7,656 / 1,641 / 1,641 |
| **Model Training** | train_data | Fit all 12 model types | trained_models | 12 models |
| **Rolling Forecast** | val_data, test_data | One-step-ahead rolling | predictions | 1,641 + 1,641 |
| **Evaluation** | predictions, actuals | Compute metrics, DM test | evaluation_metrics.csv | 12 × 6 metrics |

---

## 3.9 Methodological Validation and Quality Assurance

**Stationarity Testing**: Augmented Dickey-Fuller and KPSS tests performed on key variables to validate preprocessing appropriateness.

**Out-of-Sample Validation**: All results reported on hold-out test set (15% of data) never used during model development or hyperparameter tuning.

**Cross-Validation**: Validation set (15%) used for early stopping in neural networks and ensemble weighting optimization.

**Reproducibility**: All random seeds fixed, computational environment documented, code version controlled.

This comprehensive methodology ensures rigorous, transparent, and reproducible research outcomes suitable for academic publication and practical application.

---

**[END OF CHAPTER 3]**
