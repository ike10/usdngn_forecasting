# MASTER'S THESIS STANDARDS AND WRITING GUIDELINES

## Document Purpose

This guide ensures that Chapters 3 and 4 of the USD-NGN Exchange Rate Forecasting thesis comply with academic thesis standards, maintain professional presentation, and follow proper research communication protocols.

---

## PART 1: THESIS STRUCTURE STANDARDS

### Chapter Organization

**Proper Chapter Structure:**
```
Chapter 3: METHODOLOGY
3.1 Research Design and Overview
3.2 [Main Subsection]
    3.2.1 [Sub-subsection]
    3.2.2 [Sub-subsection]
3.3 [Main Subsection]
    ...
```

**Section Numbering:**
- Chapter level: 3.X (single digit)
- First subsection: 3.X.1 (two digits)
- Sub-subsections: 3.X.Y.1 (three digits max)
- Avoid going beyond three levels; reorganize if needed

### Chapter Length Guidelines

- **Chapter 3 (Methodology)**: 10,000-15,000 words
  - Adequate detail for reproducibility
  - Mathematics, algorithms, pseudocode
  - Implementation specifics
  
- **Chapter 4 (Results)**: 8,000-12,000 words
  - Comprehensive results presentation
  - Multiple perspectives on findings
  - Interpretation and discussion integrated

---

## PART 2: ACADEMIC WRITING STANDARDS

### Writing Style and Tone

1. **Passive Voice (Preferred in Methods)**
   - ✓ "Data were collected from CBN official rates"
   - ✗ "We collected data from CBN official rates"
   - Rationale: Emphasizes method/process over researcher

2. **Active Voice (Acceptable in Results/Discussion)**
   - ✓ "The hybrid model achieved RMSE of 24.50"
   - Also acceptable: "Hybrid model achieved RMSE of 24.50"
   - Rationale: Clearer, more engaging for findings

3. **Formal Academic Register**
   - Use: "The analysis reveals...", "Results demonstrate...", "Findings suggest..."
   - Avoid: "This is cool because...", "As we can see...", "Actually, ..."
   - Maintain: Professional, objective tone throughout

4. **Third Person**
   - ✓ "The research examines..."
   - ✗ "I examine..." or "We examine..." (unless discussing collaborative work)

### Paragraph Structure

**Optimal Paragraph Pattern:**
1. **Topic Sentence**: Introduces main idea
2. **Evidence/Details**: Supporting data, equations, citations
3. **Analysis**: Interpretation of evidence
4. **Transition/Conclusion**: Link to next paragraph

**Paragraph Length**: 150-300 words (4-8 sentences typically)
- Too short (<100 words): Seems incomplete
- Too long (>400 words): Consider splitting

### Sentence Construction

**Academic Quality:**
- Vary sentence length (mix short and long sentences)
- Use subordinate clauses for relationships: "Because the Random Walk baseline is difficult to beat, improved performance is noteworthy."
- Maintain parallel structure: "The framework combines [A], [B], and [C]"

**Common Errors to Avoid:**
- ✗ Run-on sentences (>40 words)
- ✗ Comma splices (use semicolons or split sentences)
- ✗ Ambiguous pronouns (always clarify what "it" or "this" refers to)
- ✗ Dangling modifiers: "After completing the analysis, conclusions were drawn" → "After completing the analysis, we drew conclusions"

---

## PART 3: MATHEMATICAL AND TECHNICAL PRESENTATION

### Equation Formatting

**Standard Requirements:**
1. Display important equations on separate lines (not inline)
2. Number equations: Equation 3.1, 3.2, etc. within each chapter
3. Define all variables in text before/after equation
4. Use consistent notation throughout

**Example:**
```
Transfer entropy from variable X to Y is defined as:

    TE(X → Y) = Σ p(y_{t+1}, y_t^{(k)}, x_t^{(k)}) log [p(y_{t+1} | y_t^{(k)}, x_t^{(k)}) / p(y_{t+1} | y_t^{(k)})]   (Eq. 3.1)

where y_{t+1} is the next state of the target variable, y_t^{(k)} is the k-step history of the target, 
and x_t^{(k)} is the k-step history of the source variable.
```

### Algorithm/Pseudocode

**When to Include:**
- Complex multi-step procedures
- Critical for reproducibility
- Novel methodological contribution

**Format:**
```
Algorithm 3.1: Hybrid ARIMA-LSTM Training

Input: Training data X_train, y_train; Information weights W
Output: Trained hybrid model M

1. Fit ARIMA(p,d,q) to y_train
2. Extract ARIMA residuals: e = y_train - ŷ_train^ARIMA
3. Weight features: X̃ = X_train * W
4. Combine: U = [X̃; e_history]
5. Train LSTM on U to predict residuals
6. Store ARIMA and LSTM components
7. Return M = {ARIMA, LSTM, W}
```

### Inline Code and Technical Terms

**Proper Formatting:**
- Software/package names: Python, PyTorch, scikit-learn (standard capitalization)
- File names: `run_pipeline.py`, `data/evaluation_metrics.csv` (monospace)
- Variable names: x, y, RMSE, MAE (consistent)
- Model names: Random Walk (capitalize), ARIMA, LSTM (establish convention, maintain)

---

## PART 4: FIGURE AND TABLE STANDARDS

### Figure Requirements

**Physical Specifications:**
- Resolution: 300 DPI minimum (600 DPI preferred for print)
- File format: PNG, PDF, or EPS (not JPG for technical figures)
- Size: Readable at 8-10 inches width
- Font: Consistent with thesis body text (Times 11pt, Arial 10pt)

**Figure Quality Checklist:**
- ✓ Axis labels include units (e.g., "RMSE (NGN/USD)")
- ✓ Legend is present and clearly identifies all elements
- ✓ Color scheme is colorblind-friendly (no red/green only)
- ✓ Gridlines are subtle (light gray, not dominant)
- ✓ Data points/bars are clearly visible (not too small)
- ✓ Title is descriptive but concise
- ✓ Source/data attribution included in caption

**Figure Captions:**
- Location: Below figure
- Format: "Figure X.Y: [Descriptive Title]. [Explanation/interpretation]. [Data source if applicable]"
- Length: 50-150 words
- Content: Should be understandable without reading surrounding text
- Example:
  ```
  Figure 4.1: Transfer Entropy Scores for Economic Variables. Transfer entropy measures 
  directional information flow from each predictor to USD-NGN exchange rate. Scores range 
  from 0 (no information) to maximum bits transferred. Error bars represent 95% bootstrap 
  confidence intervals (n=1,000 resamples). Statistical significance: *** p<0.001, ** p<0.01, 
  * p<0.05. Data source: Transfer entropy analysis on training set (1995-2016).
  ```

### Table Requirements

**Format Standards:**
- Header row: Clearly distinguished (bold, shading, or borders)
- Borders: Minimal (horizontal lines only, no vertical gridlines)
- Alignment:
  - Text: Left-aligned
  - Numbers: Right-aligned or decimal-aligned
  - Column headers: Centered
- Row height: Adequate spacing (18pt minimum)
- Font: Same as body text; header can be bold

**Table Captions:**
- Location: Above table (opposite of figures)
- Format: "Table X.Y: [Descriptive Title]. [Brief description]."
- Footnotes: Below table
- Example footnotes:
  ```
  Table 4.3: Comprehensive Model Performance on Test Set (2020-07-05 to 2024-12-31).
  
  Note: RMSE, MAE measured in NGN per USD. MAPE in percentage. DA (Directional Accuracy) 
  represents one-step-ahead forecast accuracy as percentage. DM Test denotes Diebold-Mariano 
  statistic comparison vs. Random Walk baseline. Values sorted by RMSE ascending.
  
  *** p<0.001; ** p<0.01; * p<0.05; ns = not significant
  
  Source: Evaluation on hold-out test set; 1,641 observations (2020-07-05 to 2024-12-31).
  ```

### Figure/Table Integration

**Placement Rules:**
1. First reference in text before insertion (not after)
2. Reference format: "As shown in Table 4.3..." or "Figure 4.1 displays..."
3. Never place figure/table so first reference is on different page
4. Position immediately after paragraph containing first reference
5. Leave adequate space (at least 0.5 inch before/after)

**Referencing:**
- ✓ "See Table 4.3 for complete results"
- ✓ "As demonstrated in Figure 4.1"
- ✗ "In the following table..." (don't reference location, use number)
- ✗ "Look at the figure below" (use specific figure number)

---

## PART 5: CITATION AND REFERENCE STANDARDS

### In-Text Citations

**Author-Date Format (Recommended for Theses):**

**Single Author:**
- First citation: (Smith 2020)
- Narrative: Smith (2020) demonstrates...

**Multiple Authors:**
- 2-3 authors: (Smith, Jones, and Brown 2020) or Smith et al. (2020)
- 4+ authors: (Smith et al. 2020)

**Multiple Works:**
- (Smith 2019; Jones 2020; Brown 2021)
- Listed chronologically

**Direct Quote:**
- (Smith 2020, p. 45) - Always include page number for quotes
- Example: "Exchange rates are notoriously difficult to forecast" (Smith 2020, p. 12).

### Reference List Format

**Book:**
```
Smith, J. (2020). Exchange Rate Forecasting in Emerging Markets. Oxford University Press.
```

**Journal Article:**
```
Jones, K., & Brown, L. (2019). Transfer entropy in financial forecasting. Journal of Finance, 45(3), 234-256. https://doi.org/10.xxxx/xxx
```

**Conference Paper:**
```
White, M. (2021). Machine learning approaches to currency prediction. In Proceedings of the International Conference on Financial Forecasting (pp. 123-145). IEEE.
```

**Thesis/Dissertation:**
```
Green, R. (2018). Advanced forecasting methods for emerging markets [Master's thesis, University of Lagos].
```

**Online/Website:**
```
Central Bank of Nigeria. (2024). Exchange rate data. Retrieved from https://www.cbn.gov.ng/rates/
```

---

## PART 6: LANGUAGE AND CLARITY STANDARDS

### Academic Vocabulary

**Preferred Terms:**
- ✓ "Investigate", "examine", "analyze", "evaluate"
- ✗ "Check out", "look at", "see if", "try to"

- ✓ "Demonstrates", "reveals", "indicates", "suggests"
- ✗ "Shows us", "is like", "appears to be"

- ✓ "Subsequently", "consequently", "therefore", "thus"
- ✗ "Then", "so", "like"

### Pronoun Clarity

**Ambiguous:**
> "The model improved performance. This was unexpected."
> *(What does "this" refer to? The improvement? The model? The method?)*

**Clear:**
> "The model improved performance, which was unexpected because..."
> "The performance improvement was unexpected because..."

**Rule:** If you use "this", "that", "it", or "they", always follow with a noun:
- ✓ "This result indicates..."
- ✓ "That finding suggests..."
- ✗ "This indicates..." (unclear what "this" is)

### Hedging and Certainty

**Appropriate Hedging (for results that are suggestive, not conclusive):**
- "Results suggest that..."
- "Evidence indicates that..."
- "Findings propose that..."
- "Analysis reveals a tendency toward..."

**Strong Claims (for validated, statistically significant results):**
- "Results demonstrate that..."
- "Analysis confirms that..."
- "Evidence establishes that..."
- "Findings prove that..." (use rarely; requires p<0.001)

---

## PART 7: COMMON THESIS ERRORS AND CORRECTIONS

### Error 1: Tense Inconsistency
- ✗ "The model uses ARIMA to forecast the trend and captured residuals"
- ✓ "The model uses ARIMA to forecast the trend and captures residuals"
- Rule: Keep tense consistent within sentences

### Error 2: Orphaned References
- ✗ Figure 4.1 appears with no prior mention in text
- ✓ "Performance comparison (Figure 4.1) reveals..."
- Rule: Always introduce figures/tables before insertion

### Error 3: Undefined Abbreviations
- ✗ "The RMSE is minimized using DM testing"
- ✓ "The Root Mean Square Error (RMSE) is minimized using Diebold-Mariano (DM) testing"
- Rule: Define all abbreviations on first use (including in figure captions)

### Error 4: Passive Voice Overuse
- ✗ "It was found that results were obtained by testing models"
- ✓ "Model testing revealed that results were obtained"
- Rule: Use passive when appropriate, but balance with active voice

### Error 5: Missing Context for Equations
- ✗ "TE(X→Y) = Σ p(y_{t+1}, y_t^{(k)}, x_t^{(k)}) log[...]"
- ✓ "Transfer entropy from X to Y is calculated as: TE(X→Y) = ... where [define terms]"
- Rule: Always explain equations, don't assume readers understand notation

### Error 6: Unsupported Claims
- ✗ "Machine learning revolutionizes exchange rate forecasting"
- ✓ "While machine learning has shown promise in financial forecasting, improvements over baseline models remain modest (Smith 2020)"
- Rule: Support claims with evidence or citations

---

## PART 8: CHAPTER-SPECIFIC GUIDANCE

### Chapter 3: Methodology Standards

**Essential Components:**
1. ✓ Research design overview (1-2 pages)
2. ✓ Data description with provenance (sources, dates, coverage)
3. ✓ Mathematical/theoretical foundations for each method
4. ✓ Implementation details (hyperparameters, configurations)
5. ✓ Evaluation framework with multiple metrics
6. ✓ Reproducibility information (software, versions, random seeds)

**Avoid:**
- ✗ Results in methodology chapter
- ✗ Results of preliminary analyses (save for results chapter)
- ✗ Excessive detail on standard algorithms (reference original papers instead)
- ✗ Implementation code listings (pseudocode acceptable; full code in appendix)

### Chapter 4: Results and Discussion Standards

**Essential Components:**
1. ✓ Main results clearly stated with tables/figures
2. ✓ Statistical significance testing results
3. ✓ Interpretation of results in context
4. ✓ Comparison with baseline/related work
5. ✓ Discussion of limitations and boundary conditions
6. ✓ Implications for theory and practice

**Avoid:**
- ✗ Methodology discussion (belongs in Chapter 3)
- ✗ Unexplained numerical results (interpret all major findings)
- ✗ Cherry-picked results (present complete picture, including negative findings)
- ✗ Over-interpretation (stay within data constraints)

**Structure Pattern:**
```
4.1 Overview of experimental setup (context)
4.2 Specific finding 1 [Table/Figure]
4.3 Specific finding 2 [Table/Figure]
4.4 Cross-finding analysis (synthesis)
4.5 Comparison with baselines
4.6 Limitations discussion
4.7 Implications and contributions
```

---

## PART 9: EDITING CHECKLIST

### Before Final Submission

#### Content Verification
- [ ] All tables contain complete data with proper formatting
- [ ] All figures have captions with source attribution
- [ ] All equations are numbered and explained
- [ ] All abbreviations defined on first use
- [ ] All figures/tables referenced in text before insertion
- [ ] No orphaned figure/table fragments
- [ ] No placeholder text remains (e.g., "[FIGURE HERE]")

#### Style Verification
- [ ] Consistent tense throughout (primarily past for methods, mix for results)
- [ ] Consistent terminology (not switching between "exchange rate" and "currency")
- [ ] Consistent voice (primary past passive or present active)
- [ ] No contractions (use "cannot" not "can't")
- [ ] No colloquialisms ("nice", "cool", "basically")
- [ ] Paragraphs average 4-8 sentences
- [ ] Section transitions are smooth

#### Technical Verification
- [ ] All mathematical notation is consistent (define once, use always)
- [ ] All references are cited (no statement should float without source)
- [ ] Numbers are formatted consistently (0.05 vs .05 vs 5%)
- [ ] Units are included with all quantities (NGN/USD, RMSE, %, etc.)

#### Formatting Verification
- [ ] Consistent heading styles (Chapter, Section, Subsection)
- [ ] Consistent spacing (1.5 or double-spaced body)
- [ ] Margins correct (typically 1" all sides)
- [ ] Page numbers present
- [ ] Headers/footers properly formatted
- [ ] Table of Contents accurate
- [ ] Bibliography complete and properly formatted

#### Citation Verification
- [ ] All in-text citations match bibliography
- [ ] No citations missing page numbers for direct quotes
- [ ] Bibliography entries are complete (author, date, title, source)
- [ ] Bibliography alphabetized by author
- [ ] Consistent citation format throughout

---

## PART 10: EXAMPLE PASSAGES (GOOD vs POOR)

### Example 1: Results Presentation

**POOR:**
> "The results show that the model works pretty well. The RMSE is 24.50, which is better than we expected. This is really important for the field because it demonstrates that ML can help with exchange rate prediction."

**GOOD:**
> "The hybrid ARIMA-LSTM model achieved a test set RMSE of 24.50 NGN/USD, compared to 23.49 for the Random Walk baseline. While this represents a 4.3% increase in error, the directional accuracy improvement is more significant, with the hybrid model achieving 65.39% versus 49.85% for the baseline. This directional accuracy advantage is economically meaningful for trading applications, representing 280 additional correct predictions over the 1,641 test observations."

**Improvements:**
- Removes informal language ("pretty well", "really important")
- Specifies units and comparison baseline
- Quantifies results in multiple ways
- Adds interpretation

### Example 2: Methodology Explanation

**POOR:**
> "We use transfer entropy to find which variables are important. Transfer entropy is calculated using probabilities and information theory. We then weight the features based on their transfer entropy scores."

**GOOD:**
> "Transfer entropy was computed for each predictor variable to measure directional information flow to the USD-NGN exchange rate. Transfer entropy from variable X to Y is defined as TE(X→Y) = Σ p(y_{t+1}, y_t^{(k)}, x_t^{(k)}) log[p(y_{t+1}|y_t^{(k)}, x_t^{(k)})/p(y_{t+1}|y_t^{(k)})], where y_{t+1} is the next state of the target variable. Statistical significance was assessed via bootstrap resampling (1,000 iterations, α=0.05). Feature weights were derived by combining normalized transfer entropy and mutual information scores: weight_i = α·TE_i + (1-α)·MI_i, with α=0.6."

**Improvements:**
- Includes mathematical definition
- Specifies statistical testing approach
- Explains weight derivation formula
- Maintains passive voice (appropriate for methods)
- Includes all necessary parameters

### Example 3: Finding Interpretation

**POOR:**
> "Table 4.1 shows that exchange rate memory features are most important. This makes sense because exchange rates probably follow patterns."

**GOOD:**
> "Transfer entropy analysis (Table 4.1) identifies exchange rate momentum features (usdngn_ma5, usdngn_ma20) as the strongest predictors, with transfer entropy scores of 0.1715 and 0.1668 bits respectively (both p<0.001). This dominance of autoregressive features reflects the well-documented persistence in exchange rate dynamics and supports the inclusion of lagged dependent variables in traditional time series models such as ARIMA. The TE scores for these features exceed macroeconomic predictors by 2-3%, suggesting that recent exchange rate movements provide more information than fundamentals for one-step-ahead forecasting."

**Improvements:**
- Provides specific values and statistical significance
- References relevant literature/theory
- Compares relative importance (exchange rate vs macro)
- Connects findings to model implications

---

## CONCLUSION: QUALITY ASSURANCE

**Final Quality Checklist Before Submission:**

- [ ] Read entire chapter aloud (catches awkward phrasing)
- [ ] Check every figure/table is referenced
- [ ] Verify mathematical notation is consistent
- [ ] Confirm all abbreviations are defined
- [ ] Ensure all colors are printer-friendly
- [ ] Validate all numbers and statistics
- [ ] Cross-check bibliography with citations
- [ ] Format headings consistently
- [ ] Proof for spelling/grammar (use automated tools)
- [ ] Have external reader review (peer/advisor)

---

**Document Version**: 1.0  
**Applicable to**: Chapters 3 & 4 of USD-NGN Forecasting Thesis  
**Last Updated**: May 6, 2026  
**Status**: FINAL - Ready for Thesis Writing

