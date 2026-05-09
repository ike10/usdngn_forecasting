# Chapter Four: Results and Discussion

## 4.1 Overview of the Experimental Design

This chapter presents the empirical results obtained from the USD/NGN forecasting framework. The forecasting exercise was conducted using daily observations spanning `1 January 1995` to `31 December 2024`. After preprocessing and feature engineering, the data were split chronologically into training, validation, and test partitions using a `70/15/15` ratio in order to preserve the time-series structure of the problem.

The modelling workflow followed the methodology described earlier in the thesis. First, transfer entropy and mutual information were computed on the training set only, and the resulting scores were used to derive feature weights for the hybrid model. Second, a benchmark suite comprising Random Walk, Moving Average, ARIMA, ARIMAX, LSTM, GRU, Hybrid ARIMA-LSTM, Mean Reversion, and Mean Reversion + Streak was evaluated. Finally, ensemble forecasts, regime-level analysis, explainable AI outputs, and Diebold-Mariano significance tests were generated to support robust interpretation of model performance.

All reported predictive metrics are based on a rolling one-step-ahead forecasting protocol. This is important because it mirrors real deployment conditions more faithfully than static multi-step forecasting and avoids optimistic bias in directional evaluation.

---

## 4.2 Methodological Summary

The implemented framework combines four methodological layers.

First, an information-theoretic feature screening stage was used to quantify how much predictive information each variable carries for the exchange rate. Transfer entropy was used to capture directional information flow, while mutual information captured nonlinear dependence between predictors and the target.

Second, a hybrid forecasting layer was implemented using ARIMA for linear temporal structure and LSTM-based residual correction for nonlinear structure. The hybrid also incorporated regime-aware logic, including volatility-sensitive blending and shock adjustment to account for abrupt transitions in the exchange-rate process.

Third, explainable AI was integrated through SHAP-based feature-importance estimation for the Hybrid ARIMA-LSTM model. This made it possible to identify which raw and engineered variables were most influential in the final predictions.

Fourth, model comparison was carried out using multiple error metrics and the Diebold-Mariano test. This provided a statistically grounded basis for determining whether observed differences in performance were meaningful.

---

## 4.3 Overall Model Performance

Table 4.1 reports the main out-of-sample forecasting results from the expanded benchmark suite. These results were obtained using the `fast_benchmarks` configuration, which was designed to make ARIMAX, LSTM, and GRU tractable for thesis reporting without changing the rest of the framework.

**Table 4.1. Test-set performance of all benchmark and hybrid models**

| Model | RMSE | MAE | MAPE | DA |
|---|---:|---:|---:|---:|
| Random Walk | 23.0072 | 7.0316 | 0.9595 | 49.85% |
| Moving Average | 58.1186 | 12.3274 | 1.5306 | 73.07% |
| ARIMA | 23.4605 | 6.8745 | 0.9166 | 64.53% |
| ARIMAX | 23.5001 | 6.7876 | 0.9154 | 64.72% |
| LSTM | 608.8398 | 438.1211 | 52.0884 | 49.91% |
| GRU | 581.1869 | 405.0600 | 46.3877 | 49.91% |
| Hybrid ARIMA-LSTM | 23.6238 | 6.8168 | 0.9050 | 65.39% |
| Mean Reversion | 23.0129 | 6.8202 | 0.9159 | 64.23% |
| Mean Reversion + Streak | 23.0254 | 6.7922 | 0.9103 | 64.23% |
| Ensemble (Simple Avg) | 171.7605 | 121.0992 | 14.1373 | 49.91% |
| Ensemble (Val-Weighted) | 28.7730 | 13.3994 | 1.6067 | 49.91% |
| Ensemble (Optimized) | 25.5006 | 9.7817 | 1.2117 | 49.97% |

Several conclusions follow directly from Table 4.1.

The Random Walk model remains a strong baseline in terms of level forecasting, which is consistent with the literature on exchange-rate predictability. ARIMA, ARIMAX, the Hybrid ARIMA-LSTM, and the two mean-reversion variants all produce RMSE values that are close to the Random Walk benchmark. This indicates that the problem remains difficult, but it also suggests that the developed models are at least competitive in level forecasting performance.

The strongest traditional benchmark among the econometric models is ARIMA, while ARIMAX performs at a very similar level. This implies that the exogenous variables are informative, but under the reduced-budget benchmark setting they do not yet create a decisive improvement over the pure univariate ARIMA structure.

The standalone LSTM and GRU models perform poorly in the `fast_benchmarks` mode. Their RMSE and MAE values are extremely large relative to the other models. This should not be interpreted as evidence that recurrent neural models are inherently unsuitable for the USD/NGN problem. Rather, it indicates that under the deliberately reduced training budget used for fast thesis benchmarking, these models do not converge to competitive solutions.

The Hybrid ARIMA-LSTM model remains more competitive than the standalone deep-learning models, which supports the usefulness of combining linear structure with nonlinear residual correction. However, it does not emerge as the best model in the expanded benchmark run.

Figure 4.1 should present the model comparison chart generated from the experiment outputs, corresponding to `figures/fig4_model_comparison.png`.

---

## 4.4 Interpretation of the Expanded Benchmark Suite

The introduction of ARIMAX, LSTM, and GRU expands the benchmark space and makes the evaluation more aligned with the original research objectives. This is important for two reasons.

First, ARIMAX allows the framework to test whether adding macro-financial regressors directly into an econometric structure improves exchange-rate prediction beyond ARIMA. The results show that ARIMAX is competitive, but not substantially better than ARIMA under the present configuration.

Second, the inclusion of LSTM and GRU makes it possible to compare the hybrid model against pure recurrent neural alternatives. In the present fast benchmark setting, both LSTM and GRU perform poorly. This suggests that recurrent neural models require more careful tuning and more generous training budgets before they can serve as strong standalone competitors in this application.

An important consequence of adding these weak neural benchmark members is that simple and validation-weighted ensembles deteriorate sharply. The poor LSTM and GRU outputs contaminate the ensemble averages, which explains why the ensemble models in Table 4.1 no longer dominate as they did in earlier restricted benchmark runs.

This result is analytically useful. It shows that benchmark expansion is valuable not only because it broadens model coverage, but also because it reveals how sensitive ensemble methods can be to low-quality members. In other words, a larger benchmark suite is not automatically a better ensemble unless model selection or ensemble screening is carefully controlled.

---

## 4.5 Information-Theoretic Feature Results

The information-theoretic stage produced the feature weights shown in Table 4.2.

**Table 4.2. Transfer entropy and mutual information based feature weighting**

| Feature | Weight | TE Score | MI Score | p-value |
|---|---:|---:|---:|---:|
| `usdngn_ma5` | 1.35 | high | high | 0.00 |
| `usdngn_ma20` | 1.34 | high | high | 0.00 |
| `cpi` | 1.33 | high | high | 0.00 |
| `mpr` | 1.26 | moderate-high | high | 0.00 |
| `rate_oil_ratio` | 1.25 | moderate-high | moderate-high | 0.00 |
| `brent_oil` | lower than top group | moderate | high | 0.00 |
| `mpr_change` | low | low | low | 0.00 |
| `usdngn_volatility` | low | low | low | > 0.00 |
| `oil_return` | low | low | very low | > 0.00 |

The strongest features are therefore the exchange-rate moving averages, CPI, MPR, and the oil-adjusted exchange-rate ratio. This pattern is economically plausible and consistent with the structure of the Nigerian foreign-exchange environment.

The two exchange-rate memory variables, `usdngn_ma5` and `usdngn_ma20`, indicate that short- and medium-horizon persistence is a dominant element of exchange-rate dynamics. The variable `cpi` suggests that inflationary pressure is an important macroeconomic source of exchange-rate information. The variable `mpr` indicates that monetary policy stance contains predictive content, while `rate_oil_ratio` captures the interaction between exchange-rate levels and oil-market conditions.

Figure 4.2 should present the transfer entropy and feature weighting results, corresponding to `figures/fig2_transfer_entropy.png`. Figure 4.3 should present feature importance from the explainability stage, corresponding to `figures/fig6_feature_importance.png`.

---

## 4.6 Explainable AI Results

To improve interpretability, SHAP-style feature importance was computed for the Hybrid ARIMA-LSTM model. The resulting ranked importance scores are summarised in Table 4.3.

**Table 4.3. SHAP-based feature importance for the Hybrid ARIMA-LSTM**

| Feature | Importance (%) |
|---|---:|
| `oil_return` | 12.46 |
| `usdngn_ma5` | 12.08 |
| `usdngn_ma20` | 11.66 |
| `cpi` | 11.56 |
| `brent_oil` | 11.51 |
| `mpr_change` | 11.20 |
| `usdngn_volatility` | 10.78 |
| `rate_oil_ratio` | 10.55 |
| `mpr` | 8.19 |

The SHAP results confirm that the hybrid model does not rely on a single explanatory block. Instead, it uses a mixture of exchange-rate memory features, oil-related variables, inflation, and policy signals. This is an important result because it demonstrates that the hybrid is responding to a broad macro-financial information set rather than fitting only the historical exchange-rate path.

The difference between the information-theoretic ranking and the SHAP ranking is also meaningful. Transfer entropy and mutual information identify which variables carry predictive information in general, whereas SHAP shows which variables the fitted hybrid model actually uses most heavily in practice. The two forms of explanation are therefore complementary rather than redundant.

---

## 4.7 Statistical Significance of Performance Differences

The Diebold-Mariano test was applied to compare each model against the Random Walk benchmark. This addresses the methodological requirement that differences in predictive performance should not be assessed solely on raw metric values.

**Table 4.4. Diebold-Mariano tests against Random Walk**

| Model | Loss | p-value | Interpretation |
|---|---|---:|---|
| Moving Average | MSE | 0.0004 | Significantly worse than Random Walk |
| ARIMA | MSE | 0.3699 | No significant difference |
| ARIMAX | MSE | 0.4510 | No significant difference |
| Hybrid ARIMA-LSTM | MSE | 0.4084 | No significant difference |
| Mean Reversion | MSE | 0.9497 | No significant difference |
| Mean Reversion + Streak | MSE | 0.8563 | No significant difference |
| Moving Average | MAE | 0.0000 | Significantly worse than Random Walk |
| ARIMA | MAE | 0.1594 | No significant difference |
| ARIMAX | MAE | 0.1128 | No significant difference |
| Hybrid ARIMA-LSTM | MAE | 0.0928 | No significant difference |
| Mean Reversion | MAE | 0.0002 | Significantly better than Random Walk |
| Mean Reversion + Streak | MAE | 0.0000 | Significantly better than Random Walk |

The most important implication is that under the expanded benchmark run, neither ARIMA, ARIMAX, nor the Hybrid ARIMA-LSTM is statistically superior to Random Walk on the MSE criterion. However, the mean-reversion baselines are significantly better than Random Walk on MAE. This suggests that some models achieve meaningful reductions in average absolute error even when they do not dominate on squared-error loss.

The LSTM and GRU results are clearly noncompetitive in this fast benchmark configuration and should not be used as evidence of strong neural forecasting performance in their current form.

---

## 4.8 Regime-Level Behaviour

Regime-based evaluation remains important because the USD/NGN exchange rate exhibits structural instability, especially during the post-float and depegging period. The regime analysis indicates that model performance is not constant across macroeconomic conditions.

In calmer or moderately unstable periods, the hybrid model remains competitive. In more difficult structural-break periods, the results show that robustness depends strongly on how each model handles abrupt shifts. This supports the decision to incorporate regime-aware logic into the hybrid framework, even though the present benchmark run does not yet show a decisive hybrid advantage.

Figure 4.4 should present the regime-specific results corresponding to `figures/fig5_regime_analysis.png`. Figure 4.5 should present actual-versus-predicted behaviour corresponding to `figures/fig3_predictions.png`.

---

## 4.9 Discussion

The overall empirical evidence suggests that the framework is methodologically successful, even though some of the individual model outcomes are mixed.

The data engineering, information-theoretic screening, explainability layer, and statistical testing components are functioning as intended. ARIMAX has now been implemented and shown to be operationally viable, while GRU and standalone LSTM have also been incorporated into the benchmark suite. This means the original benchmark scope has been addressed in implementation terms.

At the same time, the expanded benchmark run shows that implementation alone is not enough. Under reduced training budgets, the standalone deep-learning models are weak, and this weakness spills into ensemble behaviour. Therefore, the final interpretation must distinguish clearly between methodological completeness and empirical dominance.

The framework should therefore be described as a well-developed, information-theory-informed forecasting system that has achieved strong methodological coverage and produced several competitive models, but whose strongest pure predictive performance still depends on careful model selection and controlled benchmark inclusion.

---

## 4.10 Conclusion

Based on the current results, the thesis can now claim the following with confidence:

1. A daily USD/NGN forecasting framework covering `1995-2024` was successfully implemented.
2. Transfer entropy and mutual information were successfully used for feature weighting and feature interpretation.
3. ARIMA, ARIMAX, LSTM, GRU, Hybrid ARIMA-LSTM, and rule-based baselines were all implemented and evaluated under a common rolling one-step-ahead protocol.
4. Explainable AI using SHAP-based feature importance was successfully integrated into the framework.
5. Statistical significance testing using the Diebold-Mariano test was implemented to support robust model comparison.

The strongest caution is that the fast benchmark setting used for thesis reporting does not make the standalone LSTM and GRU models competitive. Consequently, the benchmark suite is now comprehensive in implementation, but the empirical quality of all members is not uniform.

The appropriate thesis interpretation is therefore that the research objectives have been substantially achieved, and the framework is now methodologically complete enough for academic reporting, even though not all benchmark models are equally strong in predictive performance.
