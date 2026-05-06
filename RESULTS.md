# Project Results

## Executive Summary

This project was evaluated on a rolling one-step-ahead USD/NGN forecasting task using data from `1995-01-01` to `2024-12-31`, with a `70/15/15` chronological split.

The final result is:

- the **overall forecasting system is a partial success**,
- the **best-performing model is the validation-weighted ensemble**,
- the **core ARIMA-LSTM hybrid is not yet the best model in the system**.

Why this is a partial success:

- Success criterion 1: beat Random Walk on both RMSE and Directional Accuracy.
- Success criterion 2: do so out-of-sample on the test period.

The project **does meet that criterion at the ensemble level**, but **does not meet it with the ARIMA-LSTM hybrid alone**.

---

## Final Test Results

Source: `data/evaluation_metrics.csv`

| Model | RMSE | MAE | MAPE | DA |
|---|---:|---:|---:|---:|
| Random Walk | 23.1434 | 7.0164 | 0.9746 | 50.64% |
| Moving Average | 58.0821 | 12.2979 | 1.5361 | 71.97% |
| ARIMA | 23.6137 | 6.8111 | 0.9243 | 68.19% |
| Hybrid ARIMA-LSTM | 23.6769 | 6.7762 | 0.9151 | 68.01% |
| Mean Reversion | 23.1316 | 6.7925 | 0.9281 | 66.12% |
| Mean Reversion + Streak | 23.1443 | 6.7679 | 0.9235 | 66.12% |
| Ensemble (Simple Avg) | 23.0486 | 6.6814 | 0.9129 | 66.42% |
| **Ensemble (Val-Weighted)** | **23.0471** | **6.6793** | **0.9127** | **66.48%** |
| Ensemble (Optimized) | 23.3602 | 6.7311 | 0.9119 | 68.07% |

### Best Model

The best overall model is:

- **Ensemble (Val-Weighted)** for lowest RMSE and lowest MAE.

Improvement over Random Walk:

- RMSE improvement: about **0.42%**
- MAE improvement: about **4.81%**
- Directional Accuracy improvement: about **+15.84 percentage points**

### Hybrid Model Verdict

The ARIMA-LSTM hybrid achieved:

- RMSE: `23.6769`
- DA: `68.01%`

Interpretation:

- It is materially better than Random Walk on **directional accuracy**.
- It is worse than Random Walk on **RMSE**.
- Therefore it **does not satisfy the main success condition by itself**.

---

## Success Verdict

### Was the project successful?

**Yes, partially.**

### What counts as success here?

The project objective was not only to build a model, but to build a forecasting system that:

- uses macro-financial information and engineered features,
- improves forecasting quality,
- and beats strong baselines.

### What succeeded

- The system successfully integrated:
  - transfer entropy,
  - mutual information,
  - feature engineering,
  - ARIMA,
  - LSTM residual modeling,
  - directional classification,
  - regime-aware adjustments,
  - and ensemble forecasting.
- The system produced models that beat Random Walk on both RMSE and DA.
- The **validation-weighted ensemble** and **simple average ensemble** both beat Random Walk on the final test set.

### What did not fully succeed

- The **ARIMA-LSTM hybrid was not the final winner**.
- The **directional ensemble inside the hybrid was not statistically significant overall**.
- The system’s best gains are still modest on RMSE, especially during the most severe depegging regime.

### Formal conclusion

- If the claim is: "the whole forecasting framework beats Random Walk," then the answer is **yes**.
- If the claim is: "the information-theory-driven ARIMA-LSTM hybrid is the best model in the project," then the answer is **no, not yet**.

---

## Regime-Level Findings

Source: `data/regime_evaluation.csv`

The test set is hardest during the **Depegging** regime, which covers the post-float and sharp depreciation period.

### COVID-19 Regime

- Best RMSE: **Hybrid ARIMA-LSTM** at `6.1912`
- Random Walk RMSE: `6.4603`

Interpretation:

- In a moderately unstable regime, the hybrid captures local nonlinear structure well.

### Post-COVID Regime

- Best RMSE: **Ensemble (Optimized)** at `11.2182`
- Hybrid ARIMA-LSTM RMSE: `11.2374`
- Random Walk RMSE: `11.4749`

Interpretation:

- The hybrid remains competitive here, but the best performance comes from combining models.

### Depegging Regime

- Best RMSE: **Ensemble (Val-Weighted)** at `36.7934`
- Random Walk RMSE: `36.8656`
- Hybrid ARIMA-LSTM RMSE: `37.9175`

Interpretation:

- This is the structural-break regime.
- The hybrid loses accuracy here.
- Model averaging is more robust than relying on one architecture.

### Regime Summary

- The hybrid performs relatively better in smoother or moderately unstable regimes.
- The ensemble performs best in the most difficult structural-break regime.
- This suggests the project’s real strength is **model combination**, not yet a single dominant hybrid.

---

## Information-Theoretic Feature Results

Source: `data/feature_weights.csv`

| Feature | Weight | TE Score | MI Score | p-value |
|---|---:|---:|---:|---:|
| `usdngn_ma20` | 1.35 | 0.1697 | 2.2932 | 0.00 |
| `usdngn_ma5` | 1.35 | 0.1687 | 2.2931 | 0.00 |
| `cpi` | 1.30 | 0.1735 | 1.8895 | 0.00 |
| `mpr` | 1.25 | 0.1484 | 2.1212 | 0.00 |
| `rate_oil_ratio` | 1.23 | 0.1654 | 1.6835 | 0.00 |
| `brent_oil` | 1.09 | 0.1206 | 1.9084 | 0.00 |
| `mpr_change` | 0.38 | 0.0080 | 0.0333 | 0.00 |
| `usdngn_volatility` | 0.35 | 0.0059 | 0.1082 | 0.20 |
| `oil_return` | 0.35 | 0.0061 | 0.0000 | 0.65 |

### Interpretation of the Feature Weights

The feature weights should be interpreted as **predictive influence inside this forecasting setup**, not as absolute causal proof.

#### 1. `usdngn_ma5` and `usdngn_ma20`

These are the strongest features in the system.

Meaning:

- The recent history of the exchange rate itself carries the most usable information.
- USD/NGN is strongly path-dependent.
- Short- and medium-term local price level matter more than isolated daily shocks.

Economic interpretation:

- Exchange-rate dynamics are persistent.
- Traders, policymakers, and market participants react to recent exchange-rate levels.
- Mean-reversion and trend effects both matter.

#### 2. `cpi`

This is one of the strongest macro variables.

Meaning:

- Inflation conditions carry strong predictive information for the exchange rate.

Economic interpretation:

- Higher inflation weakens purchasing power.
- Persistent domestic inflation tends to increase pressure on the naira.
- That usually pushes the USD/NGN rate upward over time.

#### 3. `mpr`

This also has strong predictive weight.

Meaning:

- Monetary policy stance matters materially for exchange-rate forecasting.

Economic interpretation:

- A higher policy rate can support the naira by tightening liquidity and raising domestic yields.
- But in the Nigerian context, MPR hikes also often happen during FX stress.
- So MPR is partly a stabilizer and partly a stress-response signal.

This means the relation is not purely:

- "higher MPR means stronger naira"

It is more accurately:

- "MPR contains information about both policy defense and macro stress."

#### 4. `rate_oil_ratio`

This engineered feature is highly informative.

Meaning:

- The interaction between USD/NGN and oil is more informative than oil alone.

Economic interpretation:

- Nigeria is oil-dependent.
- Oil influences export receipts, dollar liquidity, fiscal space, and FX reserve pressure.
- A high USD/NGN-to-oil ratio can signal that the naira is weak relative to the oil backdrop, which is useful for identifying imbalance.

#### 5. `brent_oil`

Oil level remains important, but less than the exchange-rate moving averages and inflation-policy block.

Economic interpretation:

- Higher oil prices can improve FX inflows and reduce pressure on the naira.
- But the pass-through from oil to the official exchange rate is not immediate or one-to-one.
- That is why oil matters, but not as strongly as direct exchange-rate memory and inflation/policy variables.

#### 6. `mpr_change`

This has low weight.

Meaning:

- Daily or short-horizon changes in MPR are not as useful as the policy-rate level itself.

Economic interpretation:

- MPR changes are discrete and infrequent.
- For a daily forecasting problem, the level of policy stance carries more information than the day-to-day change series.

#### 7. `usdngn_volatility`

This has low-to-moderate predictive value in the current setup.

Meaning:

- Volatility helps identify uncertainty and regime state, but does not by itself strongly determine the next level forecast.

Economic interpretation:

- Volatility is a context variable.
- It matters for how the model should behave, not necessarily as a strong direct predictor of tomorrow’s exchange-rate level.

#### 8. `oil_return`

This is the weakest feature in the current weighting scheme.

Meaning:

- Daily oil shocks do not transmit cleanly enough into next-step USD/NGN forecasts in this dataset.

Economic interpretation:

- Oil’s effect is likely slower, filtered through reserves, expectations, official pricing rules, and policy response.
- Oil level matters more than immediate daily oil return noise.

---

## How the Engineered Variables Affect USD/NGN

### Exchange-Rate Memory Features

Examples:

- `usdngn_ma5`
- `usdngn_ma20`

Effect:

- Capture persistence, local trend, and equilibrium zone.
- Help the system recognize when the current exchange rate is extended relative to recent history.

Why it matters:

- USD/NGN does not move as a memoryless process.
- Recent price structure strongly influences next-step behavior.

### Ratio Features

Example:

- `rate_oil_ratio`

Effect:

- Captures whether the exchange rate is out of line with oil conditions.

Why it matters:

- Raw macro variables often become more predictive when expressed as relationships instead of isolated levels.

### Policy Features

Examples:

- `mpr`
- `mpr_change`

Effect:

- Represent monetary tightening, policy defense, and macro response to inflation or FX pressure.

Why it matters:

- Exchange rates respond not only to fundamentals, but to policy reaction functions.

### Inflation Features

Example:

- `cpi`

Effect:

- Encodes macroeconomic erosion of domestic currency strength.

Why it matters:

- Inflation is one of the clearest long-horizon drivers of exchange-rate pressure.

### Risk and Instability Features

Example:

- `usdngn_volatility`

Effect:

- Helps the model identify unstable conditions and regime shifts.

Why it matters:

- Even if volatility does not dominate next-step levels, it matters for regime-aware forecasting.

---

## Model Interpretation

### Why the ensemble wins

The ensemble works best because it combines:

- ARIMA’s linear short-memory structure,
- the hybrid’s nonlinear residual correction,
- and the mean-reversion family’s robustness during unstable periods.

That combination reduces model-specific error.

### Why the hybrid does not yet win

The hybrid likely underperforms in the hardest regime because:

- structural breaks are too abrupt for its residual-learning structure,
- the directional ensemble remains weak overall,
- and official-rate dynamics can shift faster than the hybrid recalibrates.

### Important technical insight

The project’s strongest output is currently:

- **a robust forecasting framework with ensemble gains**

not yet:

- **a single dominant ARIMA-LSTM hybrid model**

---

## Final Conclusion

This project should be reported as a **qualified success**.

The defensible claim is:

- the project built a valid, information-theory-informed USD/NGN forecasting framework,
- the framework produced out-of-sample models that beat Random Walk on both RMSE and Directional Accuracy,
- and the best final system is the **validation-weighted ensemble**.

The claim that should **not** be made yet is:

- "the ARIMA-LSTM hybrid is the best model in the project."

The evidence does not support that statement on the current test run.

---

## Recommended Thesis Framing

The most accurate framing for the thesis is:

- transfer entropy and mutual information improved feature prioritization,
- the hybrid model added useful nonlinear structure,
- regime-aware logic improved robustness,
- but the strongest empirical result came from **ensemble forecasting**, not from the hybrid alone.

That is a stronger and more defensible academic position than overstating the hybrid’s dominance.
