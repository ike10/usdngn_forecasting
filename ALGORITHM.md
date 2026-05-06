# Project Algorithm

## Objective

Build a USD/NGN exchange-rate forecasting system that:

- uses information theory, especially transfer entropy, to measure which variables carry predictive information for USD/NGN,
- converts that information into feature weights for modeling,
- trains a hybrid ARIMA-LSTM forecasting model,
- adjusts prediction behavior during regime changes and large shocks,
- and evaluates whether the hybrid outperforms baseline models under a fair rolling one-step-ahead protocol.

---

## Notation

Let:

- \( t = 1, 2, \dots, T \) denote time index in days.
- \( y_t \) denote the USD/NGN exchange rate at time \( t \).
- \( z_t^{(j)} \) denote the \( j \)-th raw exogenous variable at time \( t \), such as oil, MPR, or CPI.
- \( x_t^{(i)} \) denote the \( i \)-th engineered feature at time \( t \).
- \( \mathbf{x}_t = [x_t^{(1)}, x_t^{(2)}, \dots, x_t^{(m)}]^\top \) denote the full engineered feature vector.
- \( \hat{y}_{t+1|t} \) denote the forecast of \( y_{t+1} \) made using information available up to time \( t \).
- \( \hat{y}^{A}_{t+1|t} \) denote the ARIMA forecast component.
- \( \hat{r}^{L}_{t+1|t} \) denote the LSTM-predicted residual correction.
- \( e_t \) denote ARIMA residual at time \( t \).
- \( w_i \) denote the information-theoretic weight of feature \( x^{(i)} \).
- \( \tilde{x}_t^{(i)} = w_i x_t^{(i)} \) denote the weighted feature.
- \( s_t \in \mathcal{S} \) denote the inferred market regime at time \( t \), where
  \( \mathcal{S} = \{\text{calm}, \text{trend}, \text{volatile}, \text{shock}\} \).
- \( \lambda(s_t) \) denote the regime-dependent residual blending coefficient.

Forecasting target:

\[
\hat{y}_{t+1|t} \approx f\left(y_{1:t}, \mathbf{x}_{1:t}\right)
\]

where \( f(\cdot) \) is approximated by the hybrid ARIMA-LSTM forecasting system.

---

## End-to-End Algorithm

### 1. Define the Forecasting Problem

Input:

- Daily USD/NGN exchange rate series
- Daily or interpolated macro-financial covariates:
  - Brent crude oil
  - Monetary Policy Rate (MPR)
  - CPI / inflation

Output:

- One-step-ahead forecast of USD/NGN
- Direction-of-change forecast
- Comparative performance against benchmark models

---

### 2. Collect and Construct Historical Data

1. Collect yearly verified anchor values for:
   - USD/NGN
   - Brent oil
   - MPR
   - CPI
2. Interpolate yearly anchors to daily frequency.
3. Add controlled noise where appropriate so the daily paths are not perfectly flat.
4. Preserve major exchange-rate transitions in the USD/NGN series as regime-sensitive daily behavior.
5. Combine all series into a single daily dataframe indexed by date.

Result:

- A synchronized historical dataset covering the full study window.

---

### 3. Preprocess the Data

1. Forward-fill and back-fill missing observations.
2. Engineer forecasting features from the raw variables.
3. Remove rows made invalid by rolling windows and lag operations.
4. Test key variables for stationarity using:
   - Augmented Dickey-Fuller
   - KPSS
5. Shift non-lag predictor columns by one step to remove look-ahead bias.

Result:

- A leakage-safe feature matrix aligned for forecasting.

---

### 4. Engineer Forecasting Features

The preprocessing stage creates features such as:

- log returns,
- moving averages,
- rolling volatility,
- lagged prices,
- rate-to-oil ratio,
- MPR changes,
- CPI momentum,
- trend features,
- short-horizon rate-of-change features.

Purpose:

- Represent both level behavior and directional behavior of the exchange rate.

Representative feature equations:

Log return of USD/NGN:

\[
r_t^{(y)} = \log\left(\frac{y_t}{y_{t-1}}\right)
\]

Log return of oil:

\[
r_t^{(o)} = \log\left(\frac{o_t}{o_{t-1}}\right)
\]

Moving average of variable \( v_t \) over a window of length \( h \):

\[
\text{MA}_t^{(h)}(v) = \frac{1}{h}\sum_{k=0}^{h-1} v_{t-k}
\]

Rolling volatility of exchange-rate returns over window \( h \):

\[
\sigma_t^{(h)} = \sqrt{\frac{1}{h-1}\sum_{k=0}^{h-1}\left(r_{t-k}^{(y)} - \bar{r}_t^{(y,h)}\right)^2}
\]

where

\[
\bar{r}_t^{(y,h)} = \frac{1}{h}\sum_{k=0}^{h-1} r_{t-k}^{(y)}
\]

Lag feature:

\[
x_t^{(\text{lag},\ell)} = y_{t-\ell}
\]

Rate-to-oil ratio:

\[
x_t^{(\text{ratio})} = \frac{y_t}{o_t}
\]

Trend relative to moving average:

\[
x_t^{(\text{trend},h)} = \frac{y_t - \text{MA}_t^{(h)}(y)}{\text{MA}_t^{(h)}(y)}
\]

---

### 5. Split the Dataset

Split the processed dataset chronologically into:

- training set: 70%
- validation set: 15%
- test set: 15%

Rules:

- No shuffling
- No future leakage
- Validation is used for model selection and weighting
- Test is used only for final out-of-sample evaluation

---

### 6. Run Information-Theoretic Feature Analysis

This is the feature-selection and weighting stage.

1. For each candidate predictor feature, align it against the target series with a one-step forecast lag.
2. Compute transfer entropy from feature to USD/NGN:
   - `TE(feature -> usdngn)`
3. Estimate statistical significance by bootstrapping against shuffled versions of the source feature.
4. Compute mutual information between the same feature and the target.
5. Normalize TE and MI scores.
6. Combine them into a single feature-importance score:

   `combined_score = alpha * normalized_TE + (1 - alpha) * normalized_MI`

7. Apply a significance penalty or bonus using the TE p-value.
8. Convert the final combined score into a bounded feature weight.

Outputs:

- transfer entropy scores,
- mutual information scores,
- feature weights for model training.

Purpose:

- Features carrying more directional and informational influence on USD/NGN receive larger modeling weight.

Formal transfer entropy from feature \( x^{(i)} \) to target \( y \):

\[
TE_{x^{(i)} \to y}
=
\sum p(y_{t+1}, y_t^{(k)}, x_t^{(i,k)})
\log
\frac{
    p(y_{t+1} \mid y_t^{(k)}, x_t^{(i,k)})
}{
    p(y_{t+1} \mid y_t^{(k)})
}
\]

where:

- \( y_t^{(k)} \) denotes the target history of order \( k \),
- \( x_t^{(i,k)} \) denotes the source-feature history of order \( k \),
- probabilities are estimated after discretization.

Mutual information between feature \( x^{(i)} \) and target \( y \):

\[
MI(x^{(i)}; y)
=
\sum_{x,y} p(x,y)\log\frac{p(x,y)}{p(x)p(y)}
\]

Normalization and fusion:

\[
\widetilde{TE}_i = \frac{TE_i - \min_j TE_j}{\max_j TE_j - \min_j TE_j + \varepsilon}
\]

\[
\widetilde{MI}_i = \frac{MI_i - \min_j MI_j}{\max_j MI_j - \min_j MI_j + \varepsilon}
\]

\[
c_i = \alpha \widetilde{TE}_i + (1-\alpha)\widetilde{MI}_i
\]

If \( p_i \) is the TE significance p-value for feature \( i \), then the significance-adjusted score is:

\[
c_i^\star = c_i \cdot g(p_i)
\]

where \( g(p_i) \) is a monotone decreasing significance function such as \( g(p_i)=1-p_i \) after clipping.

Final bounded feature weight:

\[
w_i = w_{\min} + (w_{\max} - w_{\min}) \cdot \text{Norm}(c_i^\star)
\]

---

### 7. Prepare Model Inputs

1. Select the final feature subset used by the forecasting models.
2. Extract:
   - `X_train`, `X_val`, `X_test`
   - `y_train`, `y_val`, `y_test`
3. Replace NaNs and invalid numeric values in model matrices.
4. Build a feature-weight vector in the exact order of the model input columns.

Result:

- A train/validation/test input structure ready for forecasting models.

To avoid look-ahead bias, predictors are lagged:

\[
\mathbf{x}_t^{\text{model}} = \mathbf{x}_{t-1}
\]

for non-lag features, so that predicting \( y_t \) only uses information known by time \( t-1 \).

---

### 8. Train Baseline Models

Train benchmark models for fair comparison:

1. Random Walk
   - Forecast next value as the last observed value.
2. Moving Average
   - Forecast next value from the recent rolling mean.
3. ARIMA
   - Search for a reasonable `(p, d, q)` order.
   - Fit on training USD/NGN series.

Purpose:

- Establish whether the hybrid actually adds value over standard forecasting baselines.

---

### 9. Train the Hybrid ARIMA-LSTM Model

The hybrid model has three main layers.

#### 9.1 ARIMA Layer

1. Fit ARIMA on the training target series.
2. Generate fitted residuals:

   `residual_t = actual_t - arima_prediction_t`

Purpose:

- ARIMA captures linear temporal structure and level dynamics.

Formally, let the differenced ARIMA representation be:

\[
\Phi(B)(1-B)^d y_t = \Theta(B)\epsilon_t
\]

where:

- \( B \) is the backshift operator,
- \( \Phi(B) \) is the autoregressive polynomial,
- \( \Theta(B) \) is the moving-average polynomial,
- \( d \) is the differencing order,
- \( \epsilon_t \) is white noise.

The ARIMA one-step-ahead forecast is:

\[
\hat{y}^{A}_{t+1|t} = E[y_{t+1} \mid y_{1:t}]
\]

Residual:

\[
e_t = y_t - \hat{y}^{A}_{t|t-1}
\]

#### 9.2 Information-Weighted LSTM Layer

1. Multiply each feature column by its information-theoretic weight.
2. Build LSTM input using:
   - weighted features
   - ARIMA residual history
3. Train the LSTM to predict the ARIMA residual component rather than the raw level.

Purpose:

- LSTM learns nonlinear residual structure not captured by ARIMA.

Weighted features:

\[
\tilde{x}_t^{(i)} = w_i x_t^{(i)}
\]

Weighted feature vector:

\[
\tilde{\mathbf{x}}_t = [\tilde{x}_t^{(1)}, \tilde{x}_t^{(2)}, \dots, \tilde{x}_t^{(m)}]^\top
\]

The LSTM receives a sequence of weighted features and residual history:

\[
\mathbf{u}_t = [\tilde{\mathbf{x}}_t^\top, e_t]^\top
\]

For sequence length \( L \), the input sample is:

\[
\mathbf{U}_{t-L+1:t} = (\mathbf{u}_{t-L+1}, \dots, \mathbf{u}_t)
\]

The LSTM then estimates:

\[
\hat{r}^{L}_{t+1|t} = f_{\text{LSTM}}(\mathbf{U}_{t-L+1:t})
\]

#### 9.3 Directional Ensemble Layer

1. Build direction-specific features such as:
   - direction lags,
   - up-day ratios,
   - streak length,
   - momentum,
   - volatility,
   - RSI,
   - position relative to moving averages.
2. Train a classifier ensemble to predict:
   - up move,
   - down move.
3. Use the validation set to assign model weights and choose a classification threshold.

Purpose:

- Improve directional accuracy beyond what raw level forecasting alone can provide.

Direction label:

\[
d_{t+1} =
\begin{cases}
1, & \text{if } y_{t+1} > y_t \\
0, & \text{otherwise}
\end{cases}
\]

If classifier \( q \) produces probability \( \pi_{t+1}^{(q)} = P(d_{t+1}=1 \mid \mathcal{I}_t) \), then the ensemble probability is:

\[
\pi_{t+1}^{(ens)} = \sum_{q=1}^{Q} \omega_q \pi_{t+1}^{(q)}
\]

subject to:

\[
\omega_q \ge 0, \qquad \sum_{q=1}^{Q}\omega_q = 1
\]

Directional decision:

\[
\hat{d}_{t+1} =
\begin{cases}
1, & \text{if } \pi_{t+1}^{(ens)} \ge \tau \\
0, & \text{if } \pi_{t+1}^{(ens)} < \tau
\end{cases}
\]

where \( \tau \) is a validation-optimized threshold.

---

### 10. Hybrid Forecast Combination Rule

For each one-step forecast:

1. ARIMA produces a base level forecast.
2. LSTM produces a residual correction.
3. The final level forecast is formed by combining:

   `hybrid_forecast = arima_forecast + regime_adjusted_weight * lstm_residual_prediction`

4. Directional ensemble output is used to supplement directional inference when available.

Purpose:

- Combine linear structure, nonlinear correction, and directional classification.

The base hybrid forecast is:

\[
\hat{y}_{t+1|t}^{(H)} = \hat{y}^{A}_{t+1|t} + \lambda(s_t)\hat{r}^{L}_{t+1|t}
\]

If a shock adjustment term \( \delta_{t+1}^{(shock)} \) is triggered, then:

\[
\hat{y}_{t+1|t} = \hat{y}^{A}_{t+1|t} + \lambda(s_t)\hat{r}^{L}_{t+1|t} + \delta_{t+1}^{(shock)}
\]

---

### 11. Detect Regimes and Shock Conditions

Before combining forecasts, inspect recent target behavior.

The regime detector uses recent return behavior to classify the market as:

- calm,
- trend,
- volatile,
- shock.

Signals used include:

- short-window volatility,
- long-window volatility,
- recent momentum,
- shock score based on return size relative to baseline volatility.

Purpose:

- Exchange-rate behavior is not stable across all periods, especially during devaluation and policy shifts.

Daily return:

\[
q_t = \frac{y_t - y_{t-1}}{|y_{t-1}| + \varepsilon}
\]

Short-horizon volatility over window \( h_s \):

\[
v_t^{(s)} = \text{Std}(q_{t-h_s+1}, \dots, q_t)
\]

Long-horizon volatility over window \( h_\ell \):

\[
v_t^{(\ell)} = \text{Std}(q_{t-h_\ell+1}, \dots, q_t)
\]

Momentum proxy over window \( h_m \):

\[
m_t = \frac{1}{h_m}\sum_{k=0}^{h_m-1} q_{t-k}
\]

Shock score:

\[
\kappa_t = \frac{|q_t|}{\max(v_t^{(\ell)}, v_{\text{train}}, \varepsilon)}
\]

The regime is then assigned by threshold rules:

\[
s_t =
\begin{cases}
\text{shock}, & \kappa_t \ge \theta_{shock} \\
\text{volatile}, & v_t^{(s)} > \theta_v v_t^{(\ell)} \\
\text{trend}, & |m_t| > \theta_m v_t^{(\ell)} \\
\text{calm}, & \text{otherwise}
\end{cases}
\]

---

### 12. Regime-Aware Prediction Adjustment

After regime detection:

1. Use a different residual-combination strength depending on the regime.
2. If the system detects a shock regime:
   - compute an additional shock adjustment from recent return intensity,
   - push the forecast to react more realistically to abrupt moves.

Purpose:

- Prevent the model from behaving as if all periods are statistically identical.

Shock adjustment may be written as:

\[
\delta_{t+1}^{(shock)} =
\eta \cdot \min\left(\bar{\kappa}, \frac{|q_t|}{\max(v_t^{(\ell)}, v_{\text{train}}, \varepsilon)}\right)\cdot q_t \cdot y_t
\]

where:

- \( \eta \) is a shock-sensitivity coefficient,
- \( \bar{\kappa} \) is an upper cap on shock intensity.

---

### 13. Rolling One-Step-Ahead Forecasting

For evaluation on validation and test:

1. Forecast one step ahead only.
2. After each prediction, update the history with the actual observed value.
3. Recompute the next prediction using the updated history.

Why this matters:

- This is the correct way to test a forecasting system intended for real sequential use.
- It avoids misleading multi-step flattening behavior.

Formally:

\[
\hat{y}_{t+1|t} = f_t(y_{1:t}, \mathbf{x}_{1:t})
\]

then after observing \( y_{t+1} \), update the information set and forecast:

\[
\hat{y}_{t+2|t+1} = f_{t+1}(y_{1:t+1}, \mathbf{x}_{1:t+1})
\]

---

### 14. Build Model Ensembles

Construct ensemble forecasts from selected models:

1. Simple average ensemble
2. Validation-weighted ensemble

For the validation-weighted ensemble:

1. Evaluate candidate member models on the validation set.
2. Compute inverse-RMSE weights.
3. Normalize the weights.
4. Apply them to test predictions.

Purpose:

- Reduce single-model fragility and improve robustness.

For validation-weighted ensemble with candidate models \( m=1,\dots,M \):

\[
\alpha_m^\star = \frac{1 / RMSE_m^{(val)}}{\sum_{j=1}^{M} 1 / RMSE_j^{(val)}}
\]

and the ensemble forecast is:

\[
\hat{y}_{t+1|t}^{(ens)} = \sum_{m=1}^{M} \alpha_m^\star \hat{y}_{t+1|t}^{(m)}
\]

---

### 15. Evaluate Forecast Accuracy

Evaluate every model on the test set using rolling one-step predictions.

Metrics:

- RMSE
- MAE
- MAPE
- Directional Accuracy
- Sample size

Directional Accuracy rule:

- Compare the sign of the predicted move against the sign of the actual move relative to the last known value.

Purpose:

- Measure both magnitude accuracy and market-direction usefulness.

Let \( N \) be the number of forecasted observations. Then:

Root Mean Squared Error:

\[
RMSE = \sqrt{\frac{1}{N}\sum_{t=1}^{N}(y_t - \hat{y}_t)^2}
\]

Mean Absolute Error:

\[
MAE = \frac{1}{N}\sum_{t=1}^{N}|y_t - \hat{y}_t|
\]

Mean Absolute Percentage Error:

\[
MAPE = \frac{100}{N}\sum_{t=1}^{N}\left|\frac{y_t - \hat{y}_t}{y_t}\right|
\]

For one-step directional accuracy, define:

\[
a_t =
\begin{cases}
1, & \text{if } y_t > y_{t-1} \\
0, & \text{otherwise}
\end{cases}
\qquad
\hat{a}_t =
\begin{cases}
1, & \text{if } \hat{y}_t > y_{t-1} \\
0, & \text{otherwise}
\end{cases}
\]

Then:

\[
DA = \frac{100}{N}\sum_{t=1}^{N}\mathbf{1}(a_t = \hat{a}_t)
\]

---

### 16. Evaluate by Market Regime

Partition the test period into predefined economic or policy regimes such as:

- pre-crisis,
- oil crisis,
- recovery,
- COVID-19,
- post-COVID,
- depegging / post-float.

For each regime:

1. Subset predictions and actual values by date range.
2. Recompute evaluation metrics.

Purpose:

- Determine whether the hybrid is robust across structurally different periods.

---

### 17. Compare Against Benchmarks

For every model:

1. Compare against Random Walk.
2. Identify whether the model beats the baseline on:
   - RMSE
   - Directional Accuracy
3. Optionally apply Diebold-Mariano testing for statistical comparison of forecast errors.

Success condition:

- A strong hybrid should outperform the Random Walk and other baselines consistently, not only in isolated metrics.

For statistical comparison of forecast errors between models 1 and 2, define loss differential:

\[
d_t = L(e_t^{(1)}) - L(e_t^{(2)})
\]

where \( L(\cdot) \) may be squared error or absolute error. The Diebold-Mariano test evaluates:

\[
H_0: E[d_t] = 0
\]

against the alternative that one model has lower expected forecast loss.

---

## Compact Pseudocode

```text
load raw yearly anchor data
interpolate to daily series
merge usdngn, oil, mpr, cpi

preprocess data
engineer features
shift predictors by 1 step
split into train, val, test

run transfer entropy analysis on train
run mutual information analysis on train
combine TE and MI into feature weights

train Random Walk
train Moving Average
train ARIMA

fit ARIMA on y_train
compute ARIMA residuals
weight X_train using TE/MI weights
train LSTM on weighted features + residual history
train directional classifier ensemble

for each forecast step:
    detect current regime
    get ARIMA one-step forecast
    get LSTM residual forecast
    combine forecasts using regime-dependent weight
    if shock regime:
        apply shock adjustment
    store level prediction
    update history with actual observation

evaluate all models using RMSE, MAE, MAPE, DA
evaluate by regime
compare hybrid vs benchmarks
save metrics, predictions, and feature-weight outputs
```

---

## Current Implementation Status

The codebase now implements these core ideas in broad structure:

- data generation and preprocessing,
- information-theoretic feature weighting,
- ARIMA-LSTM hybrid forecasting,
- directional ensemble logic,
- regime-aware forecast adjustment,
- benchmark comparison,
- regime-level evaluation.

What still remains a research-quality improvement area:

- stronger external benchmark models,
- more formal regime-switching methods,
- faster full-pipeline runtime,
- deeper statistical validation of outperformance claims.
