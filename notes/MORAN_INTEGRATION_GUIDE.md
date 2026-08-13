# Integration Guide: Location Encoders + Moran Eigenvectors

This guide shows how to combine location encoders with the Moran eigenvector approach from the Machine Learning Moran Eigenvector paper.

## Background

**US Counties Study** (from `US_all.ipynb`):
- Uses Moran eigenvectors from spatial weight matrices (exponential, queen)
- Feature selection via LassoCV or LassoLarsIC
- Creates interaction terms: moran × X1, moran × X2
- Standardizes all features: (X - mean) / std

**Our Location Encoder Study**:
- Uses 11 different location encoders (Space2Vec, NeRF, etc.)
- Extracts spatial effects with GeoShapley
- Focuses on amplitude recovery of spatially-varying coefficients

**Combined Approach**:
- Location encoders capture fine-grained spatial patterns
- Moran eigenvectors capture spatial autocorrelation structure
- Together: More complete spatial representation

## Architecture Comparison

### US Counties Paper (Current)
```
X1, X2 → [Standardize] → 
Moran Eigenvectors (from W_exp or W_queen) → [Lasso Selection] →
Interactions: moran × X1, moran × X2 →
[Standardize all] →
FLAML AutoML (XGB/LightGBM/RF, 30min) →
Predictions
```

### Location Encoder Study (Current)
```
X1, X2 → [Standardize] →
Coordinates → [Location Encoder] → Embeddings (12-128 dim) →
[Concatenate: X1, X2, embeddings] →
[Standardize all] →
MLP or XGBoost →
Predictions →
GeoShapley → Spatial Effects
```

### **Proposed Combined Approach**
```
X1, X2 → [Standardize] →
Coordinates → [Location Encoder] → Embeddings (12-128 dim) →
Coordinates → [Spatial Weights] → Moran Eigenvectors → [Lasso Selection] →
Interactions: selected_moran × X1, selected_moran × X2 →
[Concatenate: X1, X2, embeddings, selected_moran, interactions] →
[Standardize all] →
MLP or XGBoost →
Predictions →
GeoShapley → Spatial Effects
```

## Implementation Steps

### Step 1: Generate Moran Eigenvectors

```python
import pysal as ps
from pysal.explore.esda.moran import Moran_Local
import numpy as np
from scipy.linalg import eigh
from sklearn.linear_model import LassoCV, LassoLarsIC

def generate_moran_eigenvectors(counties_gdf, weight_type='exp', threshold=100):
    """
    Generate Moran eigenvectors from spatial weight matrix.
    
    Args:
        counties_gdf: GeoDataFrame with county geometries
        weight_type: 'exp' (exponential distance) or 'queen' (adjacency)
        threshold: Distance threshold for exponential weights (km)
        
    Returns:
        DataFrame with Moran eigenvectors
    """
    # Create spatial weights
    if weight_type == 'queen':
        w = ps.lib.weights.Queen.from_dataframe(counties_gdf)
    elif weight_type == 'exp':
        # Exponential distance weights
        centroids = counties_gdf.geometry.centroid
        coords = np.array([[c.x, c.y] for c in centroids])
        w = ps.lib.weights.DistanceBand.from_array(
            coords, 
            threshold=threshold,
            binary=False
        )
        # Apply exponential decay
        for key in w.neighbors:
            distances = w.neighbors[key]
            w.weights[key] = [np.exp(-d / threshold) for d in distances]
    
    # Row-normalize
    w.transform = 'r'
    
    # Get weight matrix
    W = w.full()[0]
    
    # Compute Moran eigenvectors (eigenvectors of spatial weight matrix)
    # Use symmetric version: 0.5 * (W + W.T)
    W_sym = 0.5 * (W + W.T)
    eigenvalues, eigenvectors = eigh(W_sym)
    
    # Sort by eigenvalue magnitude (most important first)
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Create DataFrame
    n_counties = len(counties_gdf)
    moran_df = pd.DataFrame(
        eigenvectors,
        columns=[f'moran_{weight_type}_{i}' for i in range(n_counties)]
    )
    
    return moran_df, eigenvalues


def select_moran_features(X, y, moran_df, method='lasso_cv', max_features=50):
    """
    Select most important Moran eigenvectors using Lasso.
    
    Args:
        X: Feature matrix (X1, X2)
        y: Target variable
        moran_df: DataFrame with all Moran eigenvectors
        method: 'lasso_cv' or 'lasso_bic'
        max_features: Maximum number of Moran features to select
        
    Returns:
        Selected Moran eigenvectors DataFrame
    """
    # Combine X and Moran features
    X_combined = pd.concat([X, moran_df.iloc[:, :max_features]], axis=1)
    
    if method == 'lasso_cv':
        lasso = LassoCV(cv=5, random_state=222, n_jobs=-1)
        lasso.fit(X_combined, y)
        
        # Get selected features
        coefs = lasso.coef_
        selected_idx = np.where(np.abs(coefs) > 1e-5)[0]
        
    elif method == 'lasso_bic':
        lasso = LassoLarsIC(criterion='bic')
        lasso.fit(X_combined, y)
        
        coefs = lasso.coef_
        selected_idx = np.where(np.abs(coefs) > 1e-5)[0]
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Get Moran feature names (exclude X1, X2)
    moran_cols = [col for col in X_combined.columns if 'moran' in col]
    selected_moran = [col for col in moran_cols if X_combined.columns.get_loc(col) in selected_idx]
    
    print(f"  Selected {len(selected_moran)} Moran eigenvectors (method: {method})")
    
    return moran_df[selected_moran]


def create_moran_interactions(X, selected_moran_df):
    """
    Create interaction terms: moran × X1, moran × X2
    
    Args:
        X: Feature matrix with X1, X2
        selected_moran_df: Selected Moran eigenvectors
        
    Returns:
        DataFrame with interaction terms
    """
    interactions = {}
    
    for moran_col in selected_moran_df.columns:
        # moran × X1
        interactions[f'{moran_col}_X1'] = selected_moran_df[moran_col] * X['X1']
        # moran × X2
        interactions[f'{moran_col}_X2'] = selected_moran_df[moran_col] * X['X2']
    
    interactions_df = pd.DataFrame(interactions)
    
    print(f"  Created {len(interactions_df.columns)} interaction terms")
    
    return interactions_df
```

### Step 2: Modify countiesRun.py

Add this section after generating location embeddings (around line 200):

```python
# NEW: Generate Moran eigenvectors
print(f"\n[NEW] Generating Moran eigenvectors...")

# Generate for both weight types
moran_exp, eigenvalues_exp = generate_moran_eigenvectors(
    counties_gdf, weight_type='exp', threshold=100
)
moran_queen, eigenvalues_queen = generate_moran_eigenvectors(
    counties_gdf, weight_type='queen'
)

# Select important features
X_base = pd.DataFrame({'X1': X1, 'X2': X2})
selected_moran_exp = select_moran_features(
    X_base, y, moran_exp, method='lasso_cv', max_features=50
)
selected_moran_queen = select_moran_features(
    X_base, y, moran_queen, method='lasso_cv', max_features=50
)

# Create interactions
interactions_exp = create_moran_interactions(X_base, selected_moran_exp)
interactions_queen = create_moran_interactions(X_base, selected_moran_queen)

# Combine all features
print(f"\n[3/6] Preparing ML features (with Moran)...")
X_features = pd.DataFrame({'X1': X1, 'X2': X2})

if embeddings.shape[1] > 0:
    X_embeddings = pd.DataFrame(
        embeddings,
        columns=[f'emb_{i}' for i in range(embeddings.shape[1])]
    )
else:
    X_embeddings = pd.DataFrame()

# Combine: X1, X2, embeddings, moran eigenvectors, interactions
X_ml_features = pd.concat([
    X_features,
    X_embeddings,
    selected_moran_exp,
    selected_moran_queen,
    interactions_exp,
    interactions_queen
], axis=1)

print(f"  ✓ Total features: {len(X_ml_features.columns)}")
print(f"    - Base: 2 (X1, X2)")
print(f"    - Embeddings: {embeddings.shape[1]}")
print(f"    - Moran (exp): {len(selected_moran_exp.columns)}")
print(f"    - Moran (queen): {len(selected_moran_queen.columns)}")
print(f"    - Interactions (exp): {len(interactions_exp.columns)}")
print(f"    - Interactions (queen): {len(interactions_queen.columns)}")

# Continue with StandardScaler as before...
```

### Step 3: Create Combined Experiment Script

```python
# countiesRun_with_moran.py
# Full implementation combining location encoders + Moran eigenvectors

# ... (copy countiesRun.py and add Moran functionality)
```

## Experimental Design

### Ablation Study

Compare 5 configurations:

1. **Baseline**: X1, X2 only
2. **Location Encoders Only**: X1, X2 + embeddings
3. **Moran Only**: X1, X2 + selected Moran eigenvectors + interactions
4. **Combined (no interactions)**: X1, X2 + embeddings + selected Moran eigenvectors
5. **Combined (full)**: X1, X2 + embeddings + selected Moran eigenvectors + interactions

### Expected Results

| Configuration | Expected OLS Slope | Rationale |
|---------------|-------------------|-----------|
| Baseline | ~0.1 | No spatial information |
| Location Encoders | ~0.4 | Fine-grained spatial patterns |
| Moran Only | ~0.3 | Spatial autocorrelation structure |
| Combined (no interactions) | ~0.5 | Complementary information |
| **Combined (full)** | **~0.6-0.7** | **Best: captures both patterns + interactions** |

## Questions to Answer

1. **Do location encoders and Moran eigenvectors capture different aspects of space?**
   - Check feature importance: Are both types used by model?
   - Check correlation: Are embeddings and Moran eigenvectors orthogonal?

2. **Does the combined approach improve amplitude recovery?**
   - Compare OLS slopes across configurations
   - Check amplitude_range_ratio metric

3. **Which weight matrix (exp vs queen) works better with location encoders?**
   - Exponential: Continuous distance decay
   - Queen: Discrete adjacency

4. **Do interaction terms (moran × X) help when embeddings are present?**
   - Ablation: Combined with vs without interactions
   - May be redundant if embeddings already capture spatial variation

## Advantages of Combined Approach

1. **Complementary Information**:
   - Location encoders: Learn from coordinates directly
   - Moran eigenvectors: Learn from spatial weight matrix structure

2. **Robustness**:
   - If embeddings fail to capture global patterns, Moran eigenvectors provide backup
   - If Moran eigenvectors miss fine-grained variation, embeddings fill gap

3. **Interpretability**:
   - Moran eigenvectors: Clear spatial autocorrelation interpretation
   - GeoShapley: Extract location-specific effects

4. **Paper Strength**:
   - Shows awareness of existing spatial ML literature
   - Demonstrates comprehensive evaluation approach
   - Provides ablation study showing value of each component

## Implementation Checklist

- [ ] Install PySAL: `conda install -c conda-forge pysal`
- [ ] Implement `generate_moran_eigenvectors()` function
- [ ] Implement `select_moran_features()` function
- [ ] Implement `create_moran_interactions()` function
- [ ] Modify `countiesRun.py` to include Moran features
- [ ] Create ablation experiment script
- [ ] Update visualization to show Moran feature importance
- [ ] Run experiments for all 5 configurations
- [ ] Aggregate results and compare metrics
- [ ] Write up findings in paper

## References

- Machine Learning Moran Eigenvector Spatial Filtering (your US_all.ipynb notebooks)
- GeoShapley: Valuing Property Locations in Predictive Modeling
- Space2Vec and other location encoder papers
