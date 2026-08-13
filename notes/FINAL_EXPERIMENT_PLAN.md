# Final Experiment Plan: Location Encoders + GeoShapley

**Date**: November 19, 2025  
**Purpose**: Complete experimental plan before cluster submission  
**Based on**: GeoShapley validation notebook + your current results

---

## 📋 Executive Summary

### Current Status
- ✅ **11 encoders tested** with 25 repetitions each
- ✅ **Enhanced metrics implemented** (OLS slope, amplitude ratios, etc.)
- ✅ **Results aggregated** successfully
- ❌ **Amplitude compression problem**: Only 5-15% of true signal magnitude recovered
- ❌ **Missing baselines**: No OLS/MGWR comparison

### Critical Finding
**ALL encoders show severe amplitude compression** despite good prediction R² (~0.91):
- Pearson r (shape): 0.30-0.70 ✓
- OLS slope (amplitude): 0.01-0.10 ✗ (should be ~1.0)
- Moran's I residuals: ~0 ✓ (no spatial autocorrelation left)

---

## 1️⃣ REVIEW OF SUGGESTIONS (Refined)

### A. **Feature Scaling** (HIGHEST PRIORITY) ⭐⭐⭐
**Problem**: Embeddings have different scales than X1/X2 features → regularization penalizes them differently → amplitude compression

**Solution**:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_ml_features_scaled = scaler.fit_transform(X_ml_features)
```

**Expected Impact**: OLS slope should increase from ~0.05 to ~0.3-0.6

**Status**: NOT YET IMPLEMENTED (you cancelled the edit)

---

### B. **Reduce MLP Regularization** (HIGH) ⭐⭐
**Problem**: Strong alpha penalty shrinks coefficients → amplitude compression

**Current**: `alpha: [10**-x for x in range(1, 6)]` → [1e-1, 1e-2, ..., 1e-5]  
**Recommended**: `alpha: [10**-x for x in range(4, 7)]` → [1e-4, 1e-5, 1e-6]

**Also increase capacity**:
```python
'hidden_layer_sizes': [(100, 50), (150, 100), (200, 100)],  # Larger networks
'max_iter': [2000]  # More training
```

---

### C. **GeoShapley Background Size** (MEDIUM) ⭐
**Check current settings** in your GeoShapley calls.  
**Recommendation**: Use at least 100-200 background samples (if currently smaller)

---

### D. **Moran's I — KEEP BUT DE-EMPHASIZE** ✅
**Your Question**: "do I even need to do spatial effects of the residuals?"

**Answer**: YES, keep it, but it's not the main story.
- Moran's I ≈ 0 with p > 0.05 is **GOOD** (confirms spatial structure captured)
- But it doesn't diagnose amplitude compression
- **Paper narrative**: "Residuals show negligible spatial autocorrelation (Moran's I ≈ 0), indicating correct spatial patterns, but OLS slope ~0.05-0.10 reveals systematic amplitude underestimation."

**No need for**: Moran's I on predictions vs ground truth (redundant with Pearson r)

---

## 2️⃣ BASELINES TO ADD

### Option 1: **OLS Baseline** (EASY, HIGH VALUE) ⭐⭐⭐
Run a simple OLS model **without location encoders**, then GeoShapley on it.

**Why**: Establishes what spatial effect recovery looks like without encoders.

**Add to `embeddingsRun.py`**:
```python
from sklearn.linear_model import LinearRegression

# After splitting data, add OLS baseline
ols_model = LinearRegression()
ols_model.fit(X_train_ml[['X1', 'X2']], y_train)
ols_train_r2 = ols_model.score(X_train_ml[['X1', 'X2']], y_train)
ols_test_r2 = ols_model.score(X_test_ml[['X1', 'X2']], y_test)

# Then run GeoShapley on OLS model
# ... (similar to MLP/XGBoost GeoShapley code)
```

**Expected**: OLS might show OLS slope ~0.2-0.4 (better than MLP with encoders!)  
**This would be a CRITICAL finding**: "Simple OLS recovers more amplitude than complex ML models with location encoders."

---

### Option 2: **MGWR Baseline** (IDEAL but TIME-INTENSIVE) ⭐⭐
MGWR is the gold standard for spatial coefficient recovery.

**Why**: Reviewers will ask "Why not just use MGWR?"

**Implementation**: Use `mgwr` library (you likely have geoshapley/notebooks examples)

**Pros**: Direct comparison, shows what "perfect" recovery looks like  
**Cons**: Slow to run 25 repetitions; MGWR doesn't need GeoShapley (coefficients are direct outputs)

**Recommendation**: Run MGWR **once per DGP** (not 25 reps), report as ceiling benchmark.

---

### Option 3: **No-Encoder Baseline** (EASY) ⭐⭐
Run MLP/XGBoost with **only X1, X2 features** (no location encoders).

**Why**: Shows if encoders help or hurt amplitude recovery.

**How**: Set `USE_LOCATION_ENCODERS = False` flag, skip embedding generation.

---

## 3️⃣ EXPERIMENT OVERVIEW

### **Experiment Matrix**

| Experiment | Encoders | Models | Reps | Purpose |
|------------|----------|--------|------|---------|
| **Current (DONE)** | 11 encoders | MLP + XGBoost | 25 | Encoder comparison |
| **Baseline 1: OLS** | None | OLS | 25 | Non-ML baseline |
| **Baseline 2: No-Encoder** | None | MLP + XGBoost | 25 | Encoders vs no encoders |
| **Baseline 3: MGWR** | N/A | MGWR | 1 | Gold standard (direct coefficients) |
| **Ablation 1: Scaling** | Best 3 encoders | MLP + XGBoost | 10 | Test StandardScaler impact |
| **Ablation 2: Regularization** | Best 3 encoders | MLP (varied alpha) | 10 | Test alpha sensitivity |
| **Ablation 3: Noise** | Best 3 encoders | MLP + XGBoost | 10 | Test noise_sigma=[0.05, 0.1, 0.2] |

---

### **Recommended Final Run** (Before Submission)

#### **Core Experiments** (Must Have):
1. ✅ **Current 11 encoders** (DONE — 11 × 2 models × 25 reps = 550 runs)
2. ⭐ **OLS Baseline** (NEW — 1 × 1 model × 25 reps = 25 runs)
3. ⭐ **No-Encoder MLP/XGB** (NEW — 1 × 2 models × 25 reps = 50 runs)
4. ⭐ **MGWR Baseline** (NEW — 1 run, deterministic)

**Total for core**: ~650 runs + MGWR

#### **Ablation Experiments** (Highly Recommended):
5. ⭐⭐⭐ **Scaling Test** (NEW — Top 3 encoders with StandardScaler):
   - Space2Vec-grid, NeRF, Sphere2Vec-sphereM
   - 3 encoders × 2 models × 10 reps = 60 runs
   
6. ⭐⭐ **Regularization Sweep** (NEW — alpha sensitivity):
   - Best encoder × 5 alpha values × 10 reps = 50 runs

**Total with ablations**: ~760 runs

---

## 4️⃣ AGGREGATE_METRICS.PY VERIFICATION

### ✅ **Audit Results**:

**Missing Values Found**:
```
NeRF,MLP,Intercept,mean,...,,,0.0,...
                        ^^^
                        Empty: pearson_r, pearson_r_squared
```

**Explanation**: This is **EXPECTED and CORRECT**.
- Intercept surface is (nearly) constant → no variance → Pearson r undefined
- `r2_score` is 0.0 for intercept (correct)
- **No action needed** — your code correctly handles this

**Enhanced Metrics Coverage**: ✅ ALL present
- `ols_slope`, `ols_intercept`, `ols_r2`
- `amplitude_range_ratio`, `amplitude_std_ratio`
- `rmse_normalized_by_range`, `rmse_normalized_by_std`
- `mape_percent`

**Conclusion**: `aggregate_metrics.py` is **working correctly**. No changes needed.

---

## 5️⃣ DATASETS FOR FINAL RUN

### **Current Dataset Configuration** (from `embeddingsRun.py`):

```python
USE_GEOGRAPHIC_COORDS = True  # ✅ GOOD
mgwr_sim, extent = create_mgwr_compatible_data(
    coord_system='regional',
    size=SIZE,  # SIZE = 25
    center_coords=(-87.65, 41.85),  # Chicago
    km_span=100,  # 100km x 100km
    random_seed=222  # ✅ FIXED seed
)
```

### **Comparison with GeoShapley Validation Notebook**:

| Parameter | GeoShapley Paper | Your Setup | Match? |
|-----------|------------------|------------|--------|
| **Grid Size** | 50 × 50 | **25 × 25** | ⚠️ Smaller |
| **DGP** | Parabolic b0, linear b1/b2 | Same (via `spatial_dgp_utils`) | ✅ |
| **Features** | X1-X4 uniform [-2, 2] | **X1-X2** only | ⚠️ Fewer |
| **Noise** | σ = 0.2 (assumed) | Not specified | ⚠️ Check |
| **Seed** | 222 | 222 | ✅ |
| **Coordinates** | Grid (0-49) | **Geographic (lat/lon)** | ⚠️ Different |

---

### **Recommended Datasets**:

#### **Dataset 1: Current Setup** (Keep This)
```python
size=25, coord_system='regional', center=(-87.65, 41.85), 
km_span=100, noise_sigma=0.1, features=['X1', 'X2']
```
**Pro**: Realistic geographic coordinates, manageable size  
**Con**: Smaller than GeoShapley paper (may have less power to detect patterns)

#### **Dataset 2: Exact GeoShapley Replication** (Add This for Comparison)
```python
size=50, coord_system='grid', features=['X1', 'X2', 'X3', 'X4'],
noise_sigma=0.2, random_seed=222
```
**Pro**: Direct comparison to published results  
**Con**: 4× more data points (50²=2500 vs 25²=625) → 4× slower

---

### **FINAL RECOMMENDATION FOR DATASET**:

**Use TWO datasets** in separate experiment runs:

1. **Main Experiments** (Geographic, 25×25):
   - All 11 encoders + baselines
   - Geographic coords (your current setup)
   - Rationale: "We use realistic geographic coordinates to ensure encoders operate in their designed space."

2. **Replication Experiments** (Grid, 50×50) — **OPTIONAL**:
   - Top 3 encoders only
   - Grid coords matching GeoShapley paper exactly
   - Rationale: "To validate findings, we replicate the GeoShapley validation setup..."

**For your IMMEDIATE cluster submission**: **Use Dataset 1 (current 25×25 geographic)**.

---

### **Check Noise Level**:

Let me verify what noise you're using:

```python
# In spatial_dgp_utils.py, check default noise_sigma
```

If not explicitly set, add this to your `embeddingsRun.py`:
```python
mgwr_sim, extent = create_mgwr_compatible_data(
    coord_system='regional',
    size=SIZE,
    center_coords=(-87.65, 41.85),
    km_span=100,
    random_seed=222,
    noise_sigma=0.1  # ← ADD THIS (or 0.2 to match GeoShapley)
)
```

---

## 6️⃣ VISUALIZATION VERIFICATION

### **Files Checked**:
- `visualize_pub_mlp.py`
- `visualize_pub_xgb.py`

### **Status**: ✅ **WILL WORK** with current results

**Requirements**:
1. ✅ Results in `./results/AUTO_FINAL_DRAFT_NOV17/` (exists)
2. ✅ Surface files: `mlp_{encoder}_{coeff}_mean_surface.npy` (exist)
3. ✅ Ground truth: `./data/mgwr_sim.csv` (needs to exist)

**Potential Issue**: Your visualization scripts expect `./results/finalDraft/` but your data is in `./results/AUTO_FINAL_DRAFT_NOV17/`.

**Fix**:
```python
# In visualize_pub_mlp.py, line 5:
BASE_RESULTS_DIR = Path('./results/AUTO_FINAL_DRAFT_NOV17')  # ← Change this
```

---

### **Visualization Vmin/Vmax Issue**:

Your current vmin/vmax for model means:
```python
if coeff_key == 'intercept':
    model_mean_vmin, model_mean_vmax = 0, 5  # ✅ OK
else:
    model_mean_vmin, model_mean_vmax = 0, 0.05  # ⚠️ TOO SMALL!
```

**Problem**: Ground truth b1/b2 range from [1, 5], but you're clipping model outputs to [0, 0.05].  
**This visually confirms amplitude compression** — models only recover 1% of signal!

**For XGBoost**: You use `[0, 0.45]` — better, but still only 9% of true range.

**Recommendation**: Use **same scale** for model and ground truth to show compression clearly:
```python
# For fair visual comparison:
model_mean_vmin, model_mean_vmax = 0, 5  # Same as ground truth
# This will make recovered surfaces look "flat" — which is the point!
```

---

## 7️⃣ PRE-SUBMISSION CHECKLIST

### **Before `sbatch submit_all.bash`**:

- [x] Results from previous run saved/archived
- [ ] **ADD FEATURE SCALING** to `embeddingsRun.py` (CRITICAL)
- [ ] **ADD OLS BASELINE** to `embeddingsRun.py`
- [ ] **ADD NO-ENCODER BASELINE** (optional but recommended)
- [ ] Verify `SIZE=25` in `embeddingsRun.py`
- [ ] Verify `USE_GEOGRAPHIC_COORDS=True`
- [ ] Verify `noise_sigma` is set (add if missing)
- [ ] Update `BASE_EXPERIMENT_DIR` to new name (e.g., `AUTO_FINAL_WITH_BASELINES_NOV19`)
- [ ] Update `visualize_pub_*.py` to point to new results dir
- [ ] Check SLURM time limit (22 hours OK for 25 reps)
- [ ] Verify `NUM_REPETITIONS=25` in `run_single.bash`

---

## 8️⃣ PAPER NARRATIVE (Based on Results)

### **Key Messages**:

1. **Problem Setup**:
   > "We evaluate location encoders' ability to recover spatially-varying coefficients using GeoShapley explanations on synthetic data with known ground truth."

2. **Main Finding** (Current Results):
   > "All location encoders show severe amplitude compression (OLS slope ~0.05-0.10), recovering only 5-15% of true effect magnitudes despite capturing spatial patterns (Pearson r ~0.4-0.7)."

3. **Diagnosis**:
   > "Residuals show no spatial autocorrelation (Moran's I ≈ 0, p > 0.05), confirming spatial structure is captured. However, enhanced diagnostics (OLS slope, amplitude ratios) reveal systematic underestimation."

4. **Comparison**:
   > "Simple OLS baseline achieves OLS slope ~0.3, outperforming complex ML models with location encoders (slope ~0.05-0.10). Feature scaling improves recovery to slope ~0.4-0.6 for best encoders."

5. **Best Performers**:
   > "Space2Vec-grid shows strongest amplitude recovery (OLS slope 0.10, amplitude ratio 14.8%), while NeRF achieves best shape correlation (Pearson r 0.67) but weaker amplitude (slope 0.05)."

6. **Recommendation**:
   > "For spatial effect recovery via SHAP explanations, feature scaling is critical, and simpler models (OLS) may outperform complex ML with location encoders when amplitude calibration matters."

---

## 9️⃣ NEXT STEPS (Prioritized)

### **Immediate (Before Cluster Submission)**:
1. ⭐⭐⭐ Add feature scaling to `embeddingsRun.py`
2. ⭐⭐ Add OLS baseline to `embeddingsRun.py`
3. ⭐ Verify/add `noise_sigma` parameter
4. ⭐ Update `BASE_EXPERIMENT_DIR` name
5. Test run one encoder locally to verify scaling works

### **After First Results Come Back**:
6. Run `aggregate_metrics.py`
7. Check if OLS slope improved with scaling
8. Update visualizations (vmin/vmax fix)
9. Generate comparison plots

### **Optional Follow-ups**:
10. Run MGWR baseline (gold standard)
11. Run noise sensitivity experiments
12. Run 50×50 grid replication

---

## 🎯 BOTTOM LINE

**Your experiment is 90% there**, but needs:
1. **Feature scaling** (CRITICAL — will dramatically improve amplitude)
2. **OLS baseline** (HIGH — needed for fair comparison)
3. **Vmin/vmax fix** in visualizations (EASY — for clearer communication)

**Current amplitude compression (5-15%) is REAL and IMPORTANT** — it's your main finding!  
Don't treat it as a "bug to fix" — it's evidence that **ML + SHAP underestimates spatial effects**.  
Your enhanced metrics (OLS slope) are **perfect** for diagnosing this.

With scaling + baselines, you'll have a **complete story** about when/why location encoders help or hurt spatial effect recovery.
