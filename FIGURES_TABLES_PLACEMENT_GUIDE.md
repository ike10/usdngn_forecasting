# THESIS FIGURES AND TABLES PLACEMENT GUIDE

## Overview

This document provides a complete guide to all figures and tables that should be included in Chapters 3 and 4 of the Master's thesis. It indicates placement location, caption requirements, and data sources.

---

## CHAPTER 3: METHODOLOGY

### Tables

#### Table 3.1: Data Sources and Primary Variables
- **Location**: Section 3.2.1 (after data sources paragraph)
- **Type**: Reference table
- **Data Source**: From proposal document and implementation
- **Columns**: Variable | Frequency | Source | Type | Period
- **Status**: ✓ Included in Chapter 3

#### Table 3.2: Engineered Features by Category
- **Location**: Section 3.2.3 (feature engineering section)
- **Type**: Reference table with equations
- **Data Source**: From run_pipeline.py and preprocessing.py
- **Columns**: Category | Features | Count | Equations/Notes
- **Note**: Include LaTeX formulas for feature calculations
- **Status**: ✓ Included in Chapter 3

#### Table 3.3: Data Processing Pipeline Steps
- **Location**: Section 3.8 (summary section)
- **Type**: Process flow table
- **Data Source**: From data collection and preprocessing modules
- **Columns**: Stage | Input | Processing | Output | Records
- **Note**: Shows data transformation through pipeline
- **Status**: ✓ Included in Chapter 3

### Figures

#### Figure 3.1 (NOT INCLUDED - Recommended)
- **Title**: Data Collection Timeline and Coverage
- **Type**: Timeline/Gantt chart
- **Content**: Shows six economic periods (1995-2024), highlighting key events
- **Suggested Location**: After Section 3.2.1 data sources
- **What to Show**: 
  - Return to Democracy (1998-2000)
  - Oil Boom (2010-2014)
  - Oil Crisis (2014-2016)
  - Recovery (2017-2019)
  - COVID-19 (2020-2022)
  - Depegging (2023-2024)
- **Data Source**: RESULTS.md regime definitions

#### Figure 3.2 (NOT INCLUDED - Recommended)
- **Title**: Feature Engineering Workflow Diagram
- **Type**: Process diagram / flowchart
- **Content**: Shows raw variables → preprocessing → feature engineering → 27 engineered features
- **Suggested Location**: After Section 3.2.3 feature engineering
- **What to Show**:
  - Input: 4 raw variables (usdngn, oil, mpr, cpi)
  - Processing: Returns, MAs, volatility, lagged, ratios
  - Output: 27 features
- **Format**: Box and arrow diagram

#### Figure 3.3 (NOT INCLUDED - Recommended)
- **Title**: Transfer Entropy Computation Process
- **Type**: Conceptual diagram
- **Content**: Shows discretization → probability distribution → TE formula → significance testing
- **Suggested Location**: After Section 3.3.1 TE computation
- **What to Show**: Visual representation of TE calculation steps

---

## CHAPTER 4: RESULTS AND DISCUSSION

### Tables

#### Table 4.1: Transfer Entropy and Feature Importance Results
- **Location**: Section 4.2.1 (after TE analysis paragraph)
- **Type**: Results table (main analysis)
- **Data Source**: From `data/transfer_entropy_scores.csv` and `data/feature_weights.csv`
- **Columns**: Rank | Feature | TE Score (bits) | MI Score | p-value | Weight | Significance
- **Status**: ✓ Included in Chapter 4
- **Note**: Should match verification output exactly

#### Table 4.2: Mutual Information Rankings
- **Location**: Section 4.2.2 (after MI analysis)
- **Type**: Rankings table
- **Data Source**: From `data/mutual_information_scores.csv`
- **Columns**: Rank | Feature | MI Score | Pattern
- **Status**: ✓ Included in Chapter 4

#### Table 4.3: Comprehensive Model Performance on Test Set
- **Location**: Section 4.3.1 (main results)
- **Type**: Key results table
- **Data Source**: From `data/evaluation_metrics.csv`
- **Columns**: Model | RMSE | MAE | MAPE (%) | DA_1Step (%) | Rank
- **Status**: ✓ Included in Chapter 4
- **Note**: Sort by RMSE ascending; mark best performer

#### Table 4.4: Validation Set Performance
- **Location**: Section 4.3.2 (validation results)
- **Type**: Comparative table
- **Data Source**: From validation run results
- **Columns**: Model | RMSE | MAE | MAPE (%)
- **Status**: ✓ Included in Chapter 4

#### Table 4.5: Diebold-Mariano Test Results (MSE Loss)
- **Location**: Section 4.4 (statistical testing)
- **Type**: Statistical test results
- **Data Source**: From `data/diebold_mariano_tests.csv`
- **Columns**: Model 1 | Model 2 (Benchmark) | DM Statistic | p-value | Conclusion
- **Status**: ✓ Included in Chapter 4
- **Note**: Show subset of key comparisons; full matrix in appendix

#### Table 4.6: SHAP Feature Importance for Hybrid ARIMA-LSTM
- **Location**: Section 4.5 (explainability results)
- **Type**: Importance ranking table
- **Data Source**: From `data/shap_feature_importance.csv`
- **Columns**: Rank | Feature | Mean Abs SHAP | Contribution (%)
- **Status**: ✓ Included in Chapter 4

#### Table 4.7: Hybrid ARIMA-LSTM Component Analysis
- **Location**: Section 4.6.1 (model decomposition)
- **Type**: Decomposition table
- **Data Source**: From model component analysis
- **Columns**: Component | RMSE | Contribution to Final | Notes
- **Status**: ✓ Included in Chapter 4

#### Table 4.8: Mean Reversion Variants Comparison
- **Location**: Section 4.6.2 (model comparison)
- **Type**: Variant comparison
- **Data Source**: From `data/evaluation_metrics.csv`
- **Columns**: Model | RMSE | MAE | DA (%) | Mechanism
- **Status**: ✓ Included in Chapter 4

#### Table 4.9: ARIMA vs ARIMAX Analysis
- **Location**: Section 4.6.3 (exogenous variable effects)
- **Type**: Comparative analysis
- **Data Source**: From evaluation results
- **Columns**: Model | RMSE | MAE | Order
- **Status**: ✓ Included in Chapter 4

#### Table 4.10: Market Regime Definitions and Characteristics
- **Location**: Section 4.7.1 (regime classification)
- **Type**: Regime definitions
- **Data Source**: From RESULTS.md regime specifications
- **Columns**: Regime | Period | USD-NGN Range | Volatility | Events | Duration
- **Status**: ✓ Included in Chapter 4

#### Table 4.11: Model Performance Across Regimes (RMSE)
- **Location**: Section 4.7.2 (regime-specific performance)
- **Type**: Performance heatmap data
- **Data Source**: From `data/regime_evaluation.csv`
- **Columns**: Model | Pre-Crisis | Oil Crisis | Recovery | COVID | Depegging | Avg
- **Status**: ✓ Included in Chapter 4
- **Note**: Can also be presented as heatmap

#### Table 4.12: Directional Accuracy Economic Value (Illustrative)
- **Location**: Section 4.9 (trading application)
- **Type**: Economic impact table
- **Data Source**: Calculated from DA percentages
- **Columns**: Model | DA (%) | Correct Predictions | Incorrect | Trading Signal Accuracy
- **Status**: ✓ Included in Chapter 4

#### Table 4.13: Consolidated Performance Summary
- **Location**: Section 4.10 (summary table)
- **Type**: Summary table
- **Data Source**: Aggregated from evaluation results
- **Columns**: Metric | Best Model(s) | Value | Baseline | Improvement
- **Status**: ✓ Included in Chapter 4

#### Table 4.14: Model Performance with vs. without Information Weighting
- **Location**: Section 4.11 (framework validation)
- **Type**: Ablation study results
- **Data Source**: From comparison experiments
- **Columns**: Model | With TE Weights | Without Weights | Difference
- **Status**: ✓ Included in Chapter 4

#### Table 4.15: Research Objectives Fulfillment
- **Location**: Section 4.12 (objectives achievement)
- **Type**: Summary checklist
- **Data Source**: From project verification
- **Columns**: Objective | Target | Achievement | Status
- **Status**: ✓ Included in Chapter 4

### Figures

#### Figure 4.1: Transfer Entropy Rankings Bar Chart with Error Bars
- **Location**: Section 4.2.1 (after Table 4.1)
- **Type**: Bar chart with error bars
- **Data Source**: `data/transfer_entropy_scores.csv`
- **What to Show**:
  - X-axis: Features (9 total)
  - Y-axis: Transfer Entropy Score (bits)
  - Error bars: 95% CI from bootstrap
  - Significance markers: *, **, ***, ns
  - Color coding: Different color for each significance level
- **Dimensions**: 12" × 6" (wide format)
- **Note**: Placeholder text "**[FIGURE PLACEMENT: Insert Figure 4.1...]**" in Chapter 4

#### Figure 4.2: Transfer Entropy vs. Mutual Information Scatter Plot
- **Location**: Section 4.2.2 (after TE vs MI discussion)
- **Type**: Scatter plot
- **Data Source**: `data/transfer_entropy_scores.csv` + `data/mutual_information_scores.csv`
- **What to Show**:
  - X-axis: Transfer Entropy Score
  - Y-axis: Mutual Information Score
  - Points: One per feature (9 total)
  - Color coding: By feature category (MA, policy, commodities, inflation)
  - Trend line: Linear fit with correlation coefficient
  - Labels: Feature names as point labels
- **Dimensions**: 8" × 8" (square)
- **Note**: Shows correlation between TE and MI metrics

#### Figure 4.3: Model Comparison Bar Chart (RMSE)
- **Location**: Section 4.3.1 (after Table 4.3)
- **Type**: Horizontal bar chart
- **Data Source**: `data/evaluation_metrics.csv`
- **What to Show**:
  - Y-axis: Model names (12 models, sorted by RMSE)
  - X-axis: RMSE values
  - Baseline indicator: Gray dashed line at Random Walk RMSE (23.49)
  - Color coding:
    - Blue: Econometric models
    - Green: Hybrid models
    - Red: Deep learning models
    - Yellow: Ensemble methods
  - Values labeled on bars
- **Dimensions**: 12" × 8"
- **Note**: Placeholder in Chapter 4, Section 4.3.1

#### Figure 4.4: Directional Accuracy Comparison
- **Location**: Section 4.3.1 (after Figure 4.3 discussion)
- **Type**: Bar chart
- **Data Source**: `data/evaluation_metrics.csv` (DA_1Step column)
- **What to Show**:
  - X-axis: Model names (12 models)
  - Y-axis: Directional Accuracy (%)
  - Dashed horizontal line: 50% (random threshold)
  - Dashed line: Mean DA across all models (~59%)
  - Color coding: Same as Figure 4.3
  - Values labeled on bars
- **Dimensions**: 12" × 6"
- **Note**: Emphasizes which models achieve >50% DA

#### Figure 4.5: SHAP Waterfall Plot (Example Prediction)
- **Location**: Section 4.5 (after Table 4.6)
- **Type**: SHAP waterfall visualization
- **Data Source**: From SHAP analysis of holdout test sample
- **What to Show**:
  - Base value (model mean prediction)
  - Individual feature contributions (arrows, positive/negative)
  - Final predicted value
  - Feature magnitudes color-coded
  - Top 10 contributing features
- **Dimensions**: 12" × 8"
- **Caption**: "Waterfall plot showing SHAP value contributions for a randomly selected test observation. Base value (model mean prediction) of 457.3 NGN/USD is adjusted by feature contributions (positive/negative arrows) to reach predicted value of 459.8 NGN/USD. Blue arrows push prediction lower (stronger naira), red arrows push higher (weaker naira)."

#### Figure 4.6: Regime-Based RMSE Heatmap
- **Location**: Section 4.7.2 (after Table 4.11)
- **Type**: Heatmap
- **Data Source**: Table 4.11 data
- **What to Show**:
  - Rows: Models (top 5-8 best models)
  - Columns: Regimes (Pre-Crisis, Oil Crisis, Recovery, COVID, Depegging)
  - Color intensity: RMSE magnitude (light=low error, dark=high error)
  - Values in cells: RMSE numbers
  - Emphasis: Depegging regime showing dramatically higher errors
- **Dimensions**: 10" × 6"
- **Note**: Color scale: green (low error) to red (high error)

#### Figure 4.7: Predicted vs. Actual Full Time Series
- **Location**: Section 4.8.1 (time series visualization)
- **Type**: Multi-line time series plot
- **Data Source**: Test set predictions and actuals (2020-07-05 to 2024-12-31)
- **What to Show**:
  - Black line: Actual USD-NGN exchange rate
  - Colored lines: Predictions from top 5 models
  - Vertical bands: Regime transitions
    - COVID onset (2020-03)
    - Depegging announcement (2023-06)
  - Shaded area: Confidence band around ensemble prediction
  - X-axis: Date (monthly markers)
  - Y-axis: Exchange rate (NGN/USD)
- **Dimensions**: 14" × 7" (wide time series)
- **Legend**: 5 models + actual

#### Figure 4.8: Forecast Error Distribution (Box Plots)
- **Location**: Section 4.8.2 (error analysis)
- **Type**: Box plot comparison
- **Data Source**: One-step-ahead forecast errors for all 12 models
- **What to Show**:
  - X-axis: Models (ordered by median error)
  - Y-axis: Forecast error (NGN per USD)
  - Box plot elements: Median, Q1/Q3, whiskers, outliers
  - Color coding: By model category
  - Reference line: Zero error (y=0)
- **Dimensions**: 14" × 7"
- **Note**: Highlights error distribution characteristics

#### Figure 4.9: Directional Accuracy Time Series (Rolling Window)
- **Location**: Section 4.8.3 (rolling accuracy)
- **Type**: Multi-line time series
- **Data Source**: Rolling 63-day directional accuracy
- **What to Show**:
  - X-axis: Date (2020-07 to 2024-12)
  - Y-axis: Directional Accuracy (%)
  - Colored lines: Top 5 models
  - Dashed horizontal line: 50% (random)
  - Shaded background: Regime transitions
  - Emphasis: Sharp decline in 2023 depegging period
- **Dimensions**: 14" × 6"
- **Note**: Shows how model performance deteriorates over time, especially post-depegging

---

## FIGURE GENERATION NOTES

### For All Figures
1. Use consistent font: Times New Roman 11pt for labels, 9pt for annotations
2. High resolution: Minimum 300 DPI for print
3. Color scheme: Use colorblind-friendly palettes
4. Legends: Include all; position outside plot area if possible
5. Captions: Below figure; max 150 words; cite data source

### Recommended Tools
- **Python**: matplotlib, seaborn, plotly
- **R**: ggplot2
- **Specialized**: SHAP library for SHAP plots

### Figure Numbering
- Figures numbered sequentially: Figure 4.1, 4.2, ... 4.9
- Tables numbered sequentially: Table 4.1, 4.2, ... 4.15
- Cross-references in text: See Figure 4.3, Table 4.5, etc.

---

## TABLE FORMATTING GUIDELINES

1. **Header Row**: Bold, centered, light gray background
2. **Data Rows**: Alternating white/light gray rows for readability
3. **Borders**: Horizontal lines only (minimal borders)
4. **Alignment**: 
   - Text: Left-aligned
   - Numbers: Right-aligned or decimal-aligned
   - Percentages: Right-aligned with % symbol
5. **Notes Below Table**: 
   - Source information
   - Significance levels (*** p<0.001, ** p<0.01, * p<0.05)
   - Any footnotes

---

## DATA SOURCE REFERENCES

All figures and tables are generated from the project results:

```
data/
├── raw_data.csv                    # Original collected data
├── processed_data.csv              # Preprocessed with features
├── transfer_entropy_scores.csv     # TE analysis output
├── mutual_information_scores.csv   # MI analysis output
├── feature_weights.csv             # Combined weights
├── evaluation_metrics.csv          # Model performance metrics
├── diebold_mariano_tests.csv       # Statistical tests
├── regime_evaluation.csv           # Regime-specific performance
└── shap_feature_importance.csv     # SHAP values
```

---

## APPENDIX FIGURES (Not in Main Text)

Consider including in appendix:

1. **Appendix Figure A.1**: SHAP Summary Plot (bee swarm)
2. **Appendix Figure A.2**: Model Residual Autocorrelation
3. **Appendix Figure A.3**: Feature Correlation Matrix
4. **Appendix Figure A.4**: Actual vs. Predicted Scatter (all 12 models)
5. **Appendix Figure A.5**: Cumulative Forecast Error Analysis

---

## GENERATION WORKFLOW

1. Extract data from CSV files in `data/` directory
2. Generate each figure using appropriate tool (Python/R script)
3. Save as high-resolution PNG/PDF (300 DPI minimum)
4. Verify figure quality and readability
5. Insert into Chapter 4 document at marked placeholder locations
6. Update table of figures/contents
7. Cross-verify all figure numbers and captions
8. Final formatting: margins, spacing, caption alignment

---

## CHECKLIST FOR THESIS SUBMISSION

- [ ] All 15 tables included and formatted
- [ ] All 9 main figures generated and inserted
- [ ] Figure captions complete with source attribution
- [ ] Table captions include data source and methodology notes
- [ ] All in-text figure/table references correct
- [ ] High-resolution versions prepared (300+ DPI)
- [ ] Colors are colorblind-friendly
- [ ] Legends and axes properly labeled
- [ ] Numbers match between document and figures
- [ ] Table of Figures updated in front matter

---

**Document Version**: 1.0  
**Last Updated**: May 6, 2026  
**Status**: Ready for thesis submission

