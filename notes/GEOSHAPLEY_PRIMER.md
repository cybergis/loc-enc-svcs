# GeoShapley & Spatial Effects: Presentation Primer

## 1. THE PROBLEM: Black-Box Models Don't Tell You About Space

### Traditional ML Models (XGBoost, Neural Networks):
- ✅ Make good predictions: `y_pred = f(X1, X2, location_features)`
- ❌ Don't explain spatial patterns
- ❌ Don't separate location effects from feature effects
- ❌ Can't answer: "How does X1's effect vary spatially?"

### What We Want:
```
y = β₀(location) + β₁(location) × X₁ + β₂(location) × X₂
                    ↓                      ↓
            Spatially-varying coefficient (SVC)
            = how much X₁ matters at each location
```

---

## 2. SHAPLEY VALUES: Why They Work

### Core Idea (Coalition Game Theory):
Imagine a boardroom of features voting on the prediction:

```
Prediction = 100

Question: How much did "location" contribute?

- Remove location feature:    Prediction drops to 60 (contribution = 40)
- Remove X1:                  Prediction drops to 75 (contribution = 25)  
- Remove X2:                  Prediction drops to 82 (contribution = 18)

But order matters! Removing location AFTER X1 might give different answer.

Shapley Value = Average contribution across ALL possible removal orders
              = Fair credit assignment for each feature
```

### Why It's Good for Spatial:
- **Theoretically sound**: Based on cooperative game theory
- **Additive**: Contributions sum to final prediction
- **Fair**: Each feature gets credit proportional to actual impact

---

## 3. GEOSHAPLEY: Adding Geography

### Standard SHAP:
```
Features: [X₁, X₂, lon, lat]
Shapley decomposition:
  SHAP(X₁) + SHAP(X₂) + SHAP(lon) + SHAP(lat) = prediction
```

**Problem**: Treats longitude/latitude like any other feature. Misses spatial structure.

### GeoShapley (Li et al., 2024):
**Decomposes into THREE components:**

```
Prediction = Primary Effect + Geographic Effect + Interaction Effect
           = SHAP(X₁, X₂)  + SHAP(location)    + SHAP(X₁×location, X₂×location)

Where:
  Primary Effect:      How much X₁ and X₂ matter (average across space)
  Geographic Effect:   Pure location effect (b₀(location))
  Interaction Effect:  How location modifies X₁'s and X₂'s effects
                      (this gives you spatially-varying coefficients!)
```

### Mathematical Decomposition:
```
f(X, L) = μ + Primary(X) + Geo(L) + Spatial_Interaction(X, L)

Where:
  μ                    = base prediction (constant)
  Primary(X)           = effect of features alone
  Geo(L)              = effect of location alone  
  Spatial_Interaction = how location and features interact
```

---

## 4. EXTRACTING SPATIALLY-VARYING COEFFICIENTS

### From GeoShapley, you get:

**For each location i:**

```
β₀(Lᵢ) = Geo(Lᵢ)
         ↓
         Pure location effect
         (intercept that varies by location)

β₁(Lᵢ) = ∂Spatial_Interaction/∂X₁|Lᵢ
         ↓
         How much X₁ matters at location i
         (varies because of location)

β₂(Lᵢ) = ∂Spatial_Interaction/∂X₂|Lᵢ
         ↓
         How much X₂ matters at location i
```

### Interpretation:
- **High β₁(north)**: X₁ has strong effect in northern locations
- **Low β₁(south)**: X₁ has weak effect in southern locations
- **Spatial pattern in β₁**: Shows geographical heterogeneity

---

## 5. YOUR EXPERIMENTAL WORKFLOW

### Step 1: Generate Synthetic Data
```python
# TRUE data generating process (ground truth)
true_b0 = parabolic surface (intercept varies spatially)
true_b1 = gradient surface   (X₁'s effect varies spatially, range [1-5])
true_b2 = gradient surface   (X₂'s effect varies spatially, range [1-5])

# Generate outcome
y = true_b0 + true_b1 × X₁ + true_b2 × X₂ + noise
```

### Step 2: Train Black-Box Model
```python
# Model DOESN'T know about true SVCs
model = XGBoost(X₁, X₂, [embeddings/coordinates], y)

# Model learns: "High X₁ predicts high y in some locations"
# But doesn't explicitly learn β₁, β₂
```

### Step 3: Extract SVCs with GeoShapley
```python
explainer = GeoShapleyExplainer(model, background_data)
rslt = explainer.explain(data_to_explain)

# Extract spatial effects
β₀_estimated = rslt.base_value + rslt.geo
β₁_estimated = rslt.get_svc(col=[X1_idx], coef_type="gwr")
β₂_estimated = rslt.get_svc(col=[X2_idx], coef_type="gwr")
```

### Step 4: Evaluate Recovery
```python
# Compare estimated to true
Pearson_r = correlation(true_b1, estimated_b1)
             ↓
             HIGH (0.8-0.9) = Good pattern recovery
             LOW (0.1-0.3)  = Poor pattern recovery

Amplitude_ratio = range(estimated_b1) / range(true_b1)
                ↓
                HIGH (0.9-1.0) = Good magnitude recovery  
                LOW (0.06-0.07) = Poor magnitude recovery (YOUR CASE)
```

---

## 6. WHY YOUR RESULTS SHOW GOOD PATTERN BUT LOW AMPLITUDE

### Your Metrics:
| Metric | Value | Meaning |
|--------|-------|---------|
| Pearson r | 0.85 | ✅ Model captures WHERE coefficients are high/low |
| Moran's I | 0.98 | ✅ Model preserves spatial smoothness |
| Test R² | 0.91 | ✅ Model predicts y well |
| Amplitude ratio | 0.07 | ❌ Model underestimates magnitude of spatial variation |

### Why This Happens:

**Model learns spatial PATTERN but underestimates SCALE:**

```
True β₁:      [1, 2, 3, 4, 5]   (range = 4.0)
Estimated β₁: [2.8, 2.9, 3.0, 3.1, 3.2]  (range = 0.4)

Correlation: HIGH (both increase together)
Ratio: LOW (estimated is 10× smaller)
```

### Root Causes:

1. **Feature Standardization**
   - StandardScaler normalizes X₁, X₂ to mean=0, std=1
   - Model learns in compressed space
   - SHAP values extracted in compressed space

2. **Tiny Geographic Coordinate Scale**
   - Regional coords: ~0.1° longitude/latitude
   - Standardized coords: ~0.0001
   - Spatial interactions numerically weak
   - Model can detect direction but not magnitude

3. **Model Regularization**
   - XGBoost max_depth, min_child_weight
   - MLP L2 penalty (alpha)
   - Intentionally suppress large variations to prevent overfitting

---

## 7. LOCATION EMBEDDINGS: What They Add

### Without Embeddings (Baseline):
```
Features: [X₁, X₂, lon, lat]

Model learns: "When lon increases and X₁ is high, y is high"
Problem: Only 2D spatial signal, linear representation
```

### With Location Embeddings:
```
Features: [X₁, X₂, emb_0, emb_1, emb_2, emb_3]

Where embeddings are learned from coordinates:
  emb_0 = sin(lon) + cos(lat)    (captures circular patterns)
  emb_1 = sin(2×lon) + cos(2×lat) (captures finer patterns)
  emb_2 = ...                     (multi-scale spatial structure)
  emb_3 = ...

Model learns: "When these 4 embedding dimensions combine AND X₁ is high, y is high"
Benefit: Multi-scale spatial signal, richer representation
```

### Your Results:
- **Baseline (raw coords)**: Amplitude ratio = 0.058
- **With embeddings**: Amplitude ratio = 0.073  
- **Improvement**: 26% better amplitude recovery

Why? Embeddings create numerically larger spatial features → stronger gradients during training → better spatial interaction learning

---

## 8. KEY COMPONENTS FOR PRESENTATION

### Component 1: Data Generation
**What**: Synthetic grid with known spatial effects
**Why**: Can compare model's recovered effects to ground truth
**Key point**: "We know the right answer, so we can measure if GeoShapley works"

### Component 2: Black-Box Model
**What**: XGBoost or MLP trained on features + spatial representations
**Why**: Doesn't explicitly model SVCs, must be extracted
**Key point**: "Model learns to predict well, but doesn't tell us HOW location matters"

### Component 3: GeoShapley Extraction
**What**: Decomposes predictions into geographic + feature + interaction effects
**Why**: Reveals spatial patterns the model learned implicitly
**Key point**: "Like opening the black box to see what the model learned about space"

### Component 4: Evaluation Metrics
**What**: Compare estimated to true spatial effects
**Why**: Quantify how well the extraction works
**Metrics**:
- **Pearson r**: Can we detect patterns? (shapes match?)
- **Amplitude ratio**: Can we estimate magnitudes? (sizes match?)
- **Moran's I**: Is spatial autocorrelation preserved? (smoothness correct?)

---

## 9. SPATIAL VARYING COEFFICIENTS (SVCs) EXPLAINED

### Concept:
In traditional regression: `y = β₀ + β₁×X₁ + β₂×X₂`
- β₁ = constant slope for X₁ everywhere

In spatially-varying regression: `y = β₀(L) + β₁(L)×X₁ + β₂(L)×X₂`
- β₁(L) = different slope for X₁ at each location L
- Example: X₁'s effect in urban areas ≠ X₁'s effect in rural areas

### Interpretation:
```
Location A: β₁(A) = 3.5  →  "X₁ strongly predicts y here"
Location B: β₁(B) = 1.2  →  "X₁ weakly predicts y here"

Spatial variation in β₁ = evidence of geographic heterogeneity
                        = location matters for how X₁ works
```

### How GeoShapley Recovers SVCs:
```
GeoShapley says:
  "This location's prediction can be attributed to:
   - Primary effect of X₁: +2.5 (constant everywhere)
   - Interaction of X₁ with location: +0.8 (THIS location adds +0.8)
   Total SVC for X₁ at this location: 2.5 + 0.8 = 3.3"

Compare across locations:
  Location A: 2.5 + 0.8 = 3.3
  Location B: 2.5 + 0.2 = 2.7
  Location C: 2.5 - 0.3 = 2.2

Pattern: Varies spatially ✓
Magnitude: Recovers true range ✓ (or ✗ if coordinate scale is too small)
```

---

## 10. PRESENTATION NARRATIVE

### Slide Structure:

1. **Problem**: Black-box models don't explain spatial heterogeneity
   - Traditional SHAP treats location as just another feature
   - Can't distinguish "what varies spatially" from "what's constant"

2. **Solution**: GeoShapley spatial decomposition
   - Primary effects (features alone)
   - Geographic effects (location alone)
   - Interaction effects (how features vary by location)
   - Yields: Spatially-varying coefficients β₁(L), β₂(L)

3. **Method**: Your experimental workflow
   - Generate data with known SVCs
   - Train model (model doesn't see the SVCs)
   - Extract SVCs with GeoShapley
   - Compare to ground truth

4. **Results**: Pattern vs Magnitude trade-off
   - ✅ Models capture WHERE effects change (Pearson r = 0.85)
   - ✅ Models predict well (Test R² = 0.91)
   - ❌ Models underestimate HOW MUCH effects change (Amplitude ratio = 0.07)
   - **Key insight**: Coordinate scale matters for amplitude, not pattern

5. **Innovation**: Location embeddings improve recovery
   - Embeddings: 26% better amplitude recovery than raw coordinates
   - Why: Multi-scale spatial representations create stronger interactions
   - Finding: Even imperfect embeddings beat raw coordinates

6. **Implication**: GeoShapley reveals spatial processes, but coordinate representation matters
   - **For practitioners**: Choose embedding carefully if amplitude matters
   - **For theory**: Reveals limits of black-box methods for quantifying spatial effects
   - **For future work**: Better coordinate representations could improve recovery

---

## 11. QUICK REFERENCE: YOUR EXPERIMENT SPECS

| Component | Your Setup |
|-----------|-----------|
| **Grid size** | 25×25 = 625 points |
| **Coordinate system** | Regional (geographic lon/lat) |
| **Coord range** | ~0.1° (~11 km) |
| **True coefficients** | b₁, b₂ range [1-5] |
| **ML models** | XGBoost, MLP |
| **Spatial features** | Raw coords vs 11 embeddings |
| **Evaluation metric (pattern)** | Pearson r (results: 0.75-0.85) |
| **Evaluation metric (magnitude)** | Amplitude ratio (results: 0.06-0.07) |
| **Key finding** | Geographic coords create weak spatial signal for amplitude recovery |

---

## 12. FOR YOUR SLIDES: One-Slide Summary

**GeoShapley: From Predictions to Spatial Effects**

```
Black-box model learns:
  "y = f(X₁, X₂, location)"
  Implicitly: "X₁ matters more in region A"

GeoShapley extracts:
  β₀(L) = location-specific intercept
  β₁(L) = location-specific slope for X₁
  β₂(L) = location-specific slope for X₂

Evaluation:
  ✅ Captures spatial patterns (Pearson r ≈ 0.85)
  ✅ Good predictions (Test R² ≈ 0.91)
  ❌ Underestimates magnitudes (Amplitude ratio ≈ 0.07)
  
Why: Geographic coordinate scale affects magnitude learning,
     but not pattern detection.

Solution: Location embeddings improve by 26%.
```

