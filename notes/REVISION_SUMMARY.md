# Summary of Revisions to Results & Discussion

## Key Changes Made:

### 1. **Corrected Main Finding**
- **Original claim**: Location encoders help recover spatial coefficients
- **Actual finding**: Location encoders DEGRADE amplitude recovery by 34.5% compared to raw lat/lon
- The baseline "none" encoder (using only coordinates) achieves **0.801 OLS slope** vs. **0.525** for location encoders

### 2. **Added Quantitative Performance Table** (Table 1)
- Shows all 12 encoders ranked by Pearson correlation
- Displays mean ± std across 10 repetitions
- Highlights that "none" baseline is the TOP performer for amplitude recovery
- All methods achieve similar predictive performance (R² ≈ 0.98)

### 3. **Enhanced Figure Captions**
- Added context about what the visualizations show
- Clarified that "none" is the baseline using raw coordinates
- Emphasized consistent color scales for comparison

### 4. **Completely Rewrote Discussion** with key insights:

#### Main Findings:
- **Shape correlation**: All encoders achieve r > 0.96 (good pattern recovery)
- **Amplitude recovery**: Raw coordinates achieve 80% vs. 53% for encoders
- **Predictive performance**: Nearly identical across all methods (R² ≈ 0.98)

#### Mechanistic Explanations:
1. **High-dimensional entanglement**: Location encoders (12+ dims) distribute spatial information across many features, making SHAP attribution difficult
2. **Non-linear transformations**: Complex Fourier features obscure direct spatial interpretability
3. **Dimensionality curse**: More degrees of freedom dilute the coefficient signal

#### Practical Implications:
- For **prediction tasks**: Location encoders remain valuable
- For **explainability/interpretability**: Use raw coordinates
- Trade-off between predictive power and interpretability
- Important consideration for environmental modeling, urban planning, epidemiology

### 5. **Maintained Writing Style**
- Kept technical precision
- Used quantitative evidence throughout
- Referenced both attached papers appropriately
- Maintained academic tone consistent with original

### 6. **New Insights Added**
- The paradox: better encoders → worse explainability
- All methods learn spatial patterns, but embeddings hide them
- Future directions: hybrid approaches, amplitude calibration
- Validation framework for real-world applications

## What to Check Next:

1. **Verify figure files exist**: comparison_xgb_all_coefficients.pdf and comparison_mlp_all_coefficients.pdf
2. **Check if MLP results** follow same pattern (I only analyzed XGBoost in detail)
3. **Confirm citation keys** match your bibliography (li_geoshapley_2024, mai_review_2022)
4. **Review table formatting** for your LaTeX template
5. **Consider adding**: Statistical significance tests comparing none vs. encoders
