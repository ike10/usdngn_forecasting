# Project TODO List

## Status Legend
- [ ] Not started
- [x] Completed
- [~] In progress

---

## Completed Tasks

### Data Collection
- [x] Research and identify reliable data sources (CBN, World Bank, EIA, FRED)
- [x] Compile 30 years of verified USD-NGN historical rates (1995-2025)
- [x] Compile Brent crude oil prices (1995-2025)
- [x] Compile Nigeria MPR/interest rate data (1995-2025)
- [x] Compile Nigeria CPI/inflation data (1995-2025)
- [x] Implement data interpolation from yearly to daily frequency
- [x] Validate generated data against historical records (<0.1% error)

### Data Preprocessing
- [x] Implement feature engineering (27 features)
- [x] Implement train/validation/test split (70/15/15)
- [x] Handle missing values and NaN
- [x] Implement stationarity testing (ADF, KPSS)

### Model Development
- [x] Implement Random Walk baseline
- [x] Implement ARIMA with automatic order selection
- [x] Implement LSTM model
- [x] Implement Hybrid ARIMA-LSTM
- [x] Develop WinningHybridModel (mean reversion + contrarian)
- [x] Develop ImprovedHybridModel (with streak detection)

### Evaluation Framework
- [x] Implement RMSE, MAE, MAPE metrics
- [x] Implement Directional Accuracy (DA)
- [x] Implement Diebold-Mariano statistical test
- [x] Implement regime-based evaluation

### Code Quality
- [x] Organize code into src/ package structure
- [x] Create main pipeline script (run_pipeline.py)
- [x] Remove redundant files
- [x] Update README with current results

---

## Pending Tasks

### High Priority

#### Model Improvements
- [ ] Tune WinningHybrid hyperparameters (mr_speed, contrarian_magnitude, ma_window)
- [ ] Implement walk-forward validation for robust evaluation
- [ ] Add ensemble methods (weighted average of multiple models)
- [ ] Investigate why DA is lower than Random Walk in volatile periods
- [ ] Implement regime-switching model for different volatility periods

#### Visualization
- [ ] Create time series plot with actual vs predicted values
- [ ] Create residual analysis plots
- [ ] Create feature importance visualization
- [ ] Create regime-highlighted plots (pre-float vs post-float)
- [ ] Create model comparison bar charts
- [ ] Export publication-ready figures

#### Statistical Analysis
- [ ] Complete transfer entropy analysis for causality
- [ ] Perform Granger causality tests
- [ ] Add confidence intervals to predictions
- [ ] Perform out-of-sample stability analysis

### Medium Priority

#### Documentation
- [ ] Add docstrings to all functions
- [ ] Create API documentation
- [ ] Write methodology section for thesis
- [ ] Document data sources with full citations
- [ ] Create architecture diagram

#### Model Persistence
- [ ] Save trained models to pickle files
- [ ] Implement model loading and inference
- [ ] Create model versioning system
- [ ] Add model checkpointing during training

#### Testing
- [ ] Write unit tests for data collection
- [ ] Write unit tests for preprocessing
- [ ] Write unit tests for evaluation metrics
- [ ] Write integration tests for full pipeline
- [ ] Add CI/CD pipeline

### Low Priority

#### Explainability (XAI)
- [ ] Implement SHAP values for feature importance
- [ ] Create LIME explanations for individual predictions
- [ ] Analyze which features drive predictions in different regimes
- [ ] Document explainability findings

#### Additional Models
- [ ] Implement Prophet model
- [ ] Implement XGBoost/LightGBM
- [ ] Implement Transformer-based model
- [ ] Implement VAR (Vector Autoregression)

#### Real-time Capability
- [ ] Add API endpoint for predictions
- [ ] Implement real-time data fetching
- [ ] Create dashboard for monitoring
- [ ] Add alert system for significant predictions

---

## Known Issues

### Data
1. **1998 Interpolation**: The transition from fixed rate (21.89) to floating (92.34) in 1999 creates some interpolation artifacts in late 1998

### Models
1. **ARIMA Performance**: ARIMA performs poorly on test set due to regime changes (2023-2024 devaluation)
2. **DA Trade-off**: Models optimized for RMSE sometimes sacrifice Directional Accuracy
3. **Test Period Volatility**: The 2021-2025 test period has unprecedented volatility (391 → 1,558 NGN)

### Code
1. **LSTM Training**: LSTM training can be slow without GPU acceleration
2. **Memory Usage**: Full 30-year daily data (11,323 rows) can be memory-intensive

---

## Future Enhancements

### Research Extensions
- [ ] Compare with other emerging market currencies (ZAR, BRL, INR)
- [ ] Analyze impact of CBN policy announcements
- [ ] Study relationship with parallel market rates
- [ ] Investigate seasonality patterns

### Production Readiness
- [ ] Containerize with Docker
- [ ] Deploy to cloud (AWS/GCP/Azure)
- [ ] Add logging and monitoring
- [ ] Implement A/B testing framework

---

## Timeline Milestones

### Phase 1: Core Implementation (COMPLETED)
- Data collection and preprocessing
- Basic model implementation
- Evaluation framework

### Phase 2: Optimization (CURRENT)
- Hyperparameter tuning
- Model improvements
- Visualization

### Phase 3: Documentation
- Complete thesis write-up
- API documentation
- User guide

### Phase 4: Presentation
- Prepare defense slides
- Create demo notebook
- Final review

---

## Notes

### Key Insights from Analysis
1. Mean reversion toward 20-day MA is effective for RMSE reduction
2. Contrarian strategy (54.7% accuracy) adds marginal DA improvement
3. Random Walk is hard to beat consistently - this is expected per efficient market hypothesis
4. The June 2023 float and 2024 depreciation represent structural breaks

### References to Add
- Meese & Rogoff (1983) - Exchange rate forecasting difficulty
- Diebold & Mariano (1995) - Statistical test for forecast comparison
- Various CBN publications on exchange rate policy
