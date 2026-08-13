# PRE-SUBMISSION CHECKLIST
## Location Encoder + GeoShapley Experiments

**Date**: November 19, 2025  
**Cluster**: SLURM (11 parallel jobs, 22h each)

---

## ✅ MANDATORY BEFORE SUBMISSION

### 1. Code Changes (embeddingsRun.py)

- [ ] **Archive current results**:
  ```bash
  cd /u/dkiv2/group_dkiv2/active/effectsExplainableEmbeddings
  mv results/AUTO_FINAL_DRAFT_NOV17 results/ARCHIVE_NOV18_no_scaling
  ```

- [ ] **Update experiment directory**:
  ```python
  # Line ~28 in run_single.bash:
  RESULTS_DIR_BASE="${PROJECT_DIR}/results/FINAL_WITH_SCALING_NOV19"
  ```

- [ ] **ADD FEATURE SCALING** (lines ~260):
  ```python
  from sklearn.preprocessing import StandardScaler
  
  X_ml_features = pd.concat([X_features, X_embeddings], axis=1)
  ml_feature_names = X_ml_features.columns.tolist()
  
  # ← ADD THIS BLOCK:
  scaler = StandardScaler()
  X_ml_features_scaled = pd.DataFrame(
      scaler.fit_transform(X_ml_features),
      columns=X_ml_features.columns,
      index=X_ml_features.index
  )
  print(f"  ✓ Applied StandardScaler (mean=0, std=1)")
  
  X_for_geoshapley = pd.concat([X_ml_features_scaled, original_coords], axis=1)
  ```

- [ ] **Reduce MLP regularization** (optional but recommended):
  ```python
  param_dist = {
      'hidden_layer_sizes': [(100, 50), (150, 100), (200, 100)],  # Larger
      'activation': ['relu'],
      'solver': ['adam'],
      'alpha': [10**-x for x in range(4, 7)],  # Weaker: 1e-4 to 1e-6
      'learning_rate_init': [0.001, 0.0005],
      'max_iter': [2000]
  }
  ```

---

### 2. Verification Checks

- [ ] **Test scaling locally**:
  ```bash
  python embeddingsRun.py --encoder_index 8 --num_repetitions 1 \
      --base_experiment_dir ./results/test_scaling
  # Should see: "✓ Applied StandardScaler" in output
  # Check OLS slope in results (should be > 0.1)
  ```

- [ ] **Check SLURM config**:
  ```bash
  cat run_single.bash | grep -E "time|mem|cpus"
  # Should show: --time=22:00:00, --mem=192g, --cpus-per-task=24
  ```

- [ ] **Verify dataset parameters**:
  ```python
  # In embeddingsRun.py, check line ~95:
  # size=25, coord_system='regional', noise_std defaults to 0.1 ✓
  ```

---

## 🔄 OPTIONAL (Recommended)

### 3. Add OLS Baseline

Create `baselineRun.py` (copy from embeddingsRun.py, modify):
```python
from sklearn.linear_model import LinearRegression

# Replace MLP training with:
ols_model = LinearRegression()
ols_model.fit(X_train_ml[['X1', 'X2']], y_train)

# Then run GeoShapley on ols_model (same as MLP code)
```

Add to `submit_all.bash`:
```bash
# After existing submissions:
sbatch --job-name=baseline_ols baselineRun.bash
```

---

### 4. Update Visualizations

Fix directory paths:
```python
# visualize_pub_mlp.py, line 5:
BASE_RESULTS_DIR = Path('./results/FINAL_WITH_SCALING_NOV19')

# visualize_pub_mlp.py, line ~69:
# Change vmax to show amplitude compression clearly:
model_mean_vmin, model_mean_vmax = 0, 5  # Same scale as ground truth
```

---

## 📊 POST-SUBMISSION MONITORING

### 5. Check Job Status

```bash
# Monitor running jobs:
squeue -u dkiv2

# Check output logs:
tail -f slurm_logs/encoder_out_*_8.txt  # NeRF example

# Check for errors:
grep -i error slurm_logs/encoder_err_*.txt
```

### 6. Expected Completion

- **11 jobs × 25 reps × ~15 min/rep** = ~6-8 hours total (parallel)
- Each job: ~4-6 hours
- Watch for: "Applied StandardScaler" in logs (confirms fix)

---

## 📈 POST-RUN ANALYSIS

### 7. Aggregate Results

```bash
cd /u/dkiv2/group_dkiv2/active/effectsExplainableEmbeddings
python aggregate_metrics.py

# Check for improvements:
python -c "
import pandas as pd
df = pd.read_csv('results/FINAL_WITH_SCALING_NOV19/all_encoders_spatial_metrics_mean_summary.csv')
svc_x1 = df[(df['spatial_effect']=='SVC_X1_Smooth') & (df['model']=='MLP')]
print('OLS Slope Summary:')
print(svc_x1[['encoder', 'ols_slope', 'amplitude_range_ratio', 'pearson_r']].to_string())
"
```

**Expected improvements with scaling**:
| Metric | Before (no scaling) | After (with scaling) | Target |
|--------|---------------------|----------------------|--------|
| OLS Slope | 0.05-0.10 | **0.3-0.6** | ~1.0 |
| Amplitude Ratio | 5-15% | **30-60%** | ~100% |
| Pearson r | 0.4-0.7 | 0.5-0.8 | >0.7 |

---

### 8. Generate Plots

```bash
# Update path in scripts first, then:
python visualize_pub_mlp.py
python visualize_pub_xgb.py

# Check outputs:
ls -lh comparison_*.pdf
```

---

## 🚨 TROUBLESHOOTING

### If jobs fail:

**Error: "ImportError: StandardScaler"**
- Fix: Add `from sklearn.preprocessing import StandardScaler` at top of file

**Error: Keycolumn mismatch after scaling**
- Fix: Use `pd.DataFrame(...)` to preserve column names and index

**Error: OLS slope still ~0.05**
- Check: `grep "StandardScaler" slurm_logs/*.txt` to confirm scaling applied
- Check: Verify regularization alpha was reduced

**Amplitude barely improves**:
- Next: Try increasing model capacity further (300, 200 hidden units)
- Next: Try post-hoc calibration (multiply SHAP outputs by 1/ols_slope)

---

## 💡 QUICK WIN CHECKLIST

Must do (20 minutes):
1. ✅ Add StandardScaler (5 lines of code)
2. ✅ Update results directory name
3. ✅ Test locally with 1 rep
4. ✅ Submit jobs

Should do (1 hour):
5. ✅ Add OLS baseline
6. ✅ Reduce MLP alpha
7. ✅ Update visualizations

Nice to have (later):
8. ⭐ Run MGWR baseline
9. ⭐ Noise sensitivity experiments
10. ⭐ 50×50 grid replication

---

## 📝 FOR YOUR PAPER

### Key findings to report (after new run):

1. **Amplitude compression is real**: 
   - "Without feature scaling: OLS slope ~0.05-0.10 (5-10% recovery)"
   - "With feature scaling: OLS slope ~0.3-0.6 (30-60% recovery)"
   - "MGWR baseline: OLS slope ~0.9 (gold standard)"

2. **Best encoder** (will likely change with scaling):
   - Current: Space2Vec-grid (slope 0.10)
   - Expected: NeRF or Space2Vec-grid (slope ~0.5 with scaling)

3. **Diagnostic innovation**:
   - "Traditional metrics (Pearson r, MSE) insufficient for spatial effect recovery"
   - "OLS slope and amplitude ratios essential for calibration assessment"
   - "Moran's I confirms spatial structure captured despite amplitude compression"

---

## 🎯 SUCCESS CRITERIA

After this run, you should see:
- ✅ OLS slope increases 3-6× (from ~0.05 to ~0.3-0.6)
- ✅ Amplitude ratio increases proportionally
- ✅ Pearson r stays similar or improves slightly
- ✅ Model R² stays high (~0.91)
- ✅ Clear ranking of encoders by amplitude recovery

**If OLS slope < 0.2 after scaling**: Problem is elsewhere (SHAP hyperparameters, model capacity, or fundamental SHAP limitation)

**If OLS slope > 0.5 after scaling**: SUCCESS! You've demonstrated feature scaling is critical for spatial effect recovery.

---

**BOTTOM LINE**: Add StandardScaler, submit jobs, expect 3-6× improvement in amplitude recovery. This is your main contribution!
