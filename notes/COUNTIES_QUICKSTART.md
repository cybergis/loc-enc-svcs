# US Counties Location Encoder Experiments - Quick Start

## What We Just Created

A complete framework for running location encoder experiments on US county-level data, integrating:
- **11 location encoders** (Space2Vec, NeRF, Sphere2Vec, etc.)
- **Synthetic county datasets** with spatially-varying coefficients
- **GeoShapley** for spatial effect extraction
- **Feature scaling** improvements from grid experiments
- **Choropleth visualizations** for county-level results
- **Integration guide** for Moran eigenvector approach

## Files Created

1. **`us_counties_dgp.py`** (264 lines)
   - Generate synthetic data for ~3000 US counties
   - 4 DGP types: gradient, quadratic, latitude, distance
   - Auto-downloads county shapefiles or creates synthetic grid

2. **`countiesRun.py`** (385 lines)
   - County-level experiment runner
   - Includes feature scaling (StandardScaler)
   - Reduced MLP regularization (alpha: 1e-4 to 1e-6)
   - All 11 encoders supported

3. **`visualize_counties.py`** (243 lines)
   - 3-panel choropleth maps: True | Mean | Std Dev
   - Multi-encoder comparison figures
   - Publication-quality outputs

4. **`run_counties.bash`** (45 lines)
   - SLURM batch script for 11 encoders
   - Array job with 25 reps per encoder
   - 4 hours, 32GB RAM per job

5. **`COUNTIES_README.md`** (267 lines)
   - Complete usage documentation
   - Quick start examples
   - Troubleshooting guide

6. **`MORAN_INTEGRATION_GUIDE.md`** (334 lines)
   - How to combine location encoders + Moran eigenvectors
   - Implementation code snippets
   - Ablation study design

## Quick Test (5 minutes)

```bash
# 1. Test data generation
python us_counties_dgp.py

# 2. Test single experiment (1 encoder, 1 rep)
python countiesRun.py \
    --encoder_index 0 \
    --model_type MLP \
    --num_repetitions 1 \
    --output_dir ./results/test_counties

# 3. Check output
ls results/test_counties/
```

Expected output:
```
space2vec_rbf_MLP_rep0_metrics.csv
space2vec_rbf_MLP_rep0_spatial_effects.csv
```

## Full Experiment (2-3 hours on cluster)

```bash
# Submit all 11 encoders × 25 reps = 275 experiments
sbatch run_counties.bash ./results/counties_gradient_nov17 MLP 25

# Monitor progress
squeue -u $USER

# After completion, visualize
python visualize_counties.py \
    --results_dir ./results/counties_gradient_nov17 \
    --encoders space2vec_rbf space2vec_grid tile_ffn \
    --coefficient b1 \
    --output_dir ./county_figs
```

## What to Expect

### Metrics Comparison: Grid vs Counties

| Metric | Grid (25×25) | Counties (~3000) | Notes |
|--------|-------------|------------------|-------|
| **Sample Size** | 625 | ~3000 | 4.8× more data |
| **OLS Slope** | 0.10 → ~0.4 | ~0.3-0.5 | Slightly lower (irregular geometry) |
| **Pearson r** | 0.3-0.7 | 0.3-0.7 | Similar (shape preserved) |
| **Amplitude Ratio** | 15% → ~40% | ~30-50% | Better with scaling |
| **Runtime per rep** | 2-3 min | 5-10 min | More data = longer |

### Best Performers (Expected)

From grid experiments with feature scaling:
1. **space2vec_rbf**: OLS slope ~0.4 (grid) → expect ~0.35 (counties)
2. **space2vec_grid**: OLS slope ~0.4 (grid) → expect ~0.35 (counties)
3. **tile_ffn**: OLS slope ~0.3 (grid) → expect ~0.3 (counties)
4. **wrap_ffn**: OLS slope ~0.3 (grid) → expect ~0.28 (counties)

## Workflow Integration

### Current Setup
```
Grid Experiments: embeddingsRun.py + start.bash
    ↓ (completed)
Results: AUTO_FINAL_DRAFT_NOV17/
    ↓ (analyzed)
Findings: Amplitude compression, feature scaling fix
```

### New Addition
```
County Experiments: countiesRun.py + run_counties.bash
    ↓ (to be run)
Results: counties_gradient_nov17/
    ↓ (to analyze)
Comparison: Grid vs Counties amplitude recovery
    ↓ (optional)
Moran Integration: Add spatial eigenvectors
```

## Next Steps

### Priority 1: Test Current Implementations
1. Test `us_counties_dgp.py` locally (2 min)
2. Test `countiesRun.py` with 1 encoder (5 min)
3. Verify metrics CSV format matches grid experiments
4. Check if GeoShapley works with county geometries

### Priority 2: Run Full Experiments
1. Submit `sbatch run_counties.bash` (2-3 hours)
2. Monitor SLURM logs for errors
3. Aggregate metrics with `aggregate_metrics.py`
4. Create visualizations

### Priority 3: Compare Results
1. Grid vs Counties OLS slope comparison
2. Which encoders benefit most from more data?
3. Does irregular geometry hurt performance?
4. Write up county-specific findings

### Priority 4 (Optional): Moran Integration
1. Install PySAL: `conda install -c conda-forge pysal`
2. Implement Moran eigenvector generation
3. Run ablation study (5 configurations)
4. Compare combined approach to individual methods

## Key Differences from Grid Experiments

| Aspect | Grid | Counties |
|--------|------|----------|
| **Geometry** | Regular 25×25 grid | Irregular county shapes |
| **Coordinates** | Uniform spacing | Centroid-based |
| **Sample Size** | 625 points | ~3000 points |
| **Spatial Weights** | Distance-based | Can use adjacency (queen) |
| **Visualization** | Heatmap grid | Choropleth map |
| **GeoShapley** | Native grid support | Uses centroids as "grid" |
| **Runtime** | 2-3 min/rep | 5-10 min/rep |

## Integration with Paper

### New Sections to Add

1. **Section X.X: County-Level Validation**
   - "To test generalization to irregular geometries, we apply our framework to US county data..."
   - Compare grid vs county amplitude recovery
   - Show choropleth visualizations

2. **Section X.X: Comparison with Moran Eigenvector Approach**
   - "We compare location encoders to the Moran eigenvector spatial filtering method..."
   - Ablation study results
   - Discussion of complementary vs redundant information

### New Figures

1. **Figure X**: County choropleth maps (3-panel)
   - True b1 | Mean Estimate b1 | Std Dev
   - Top 3 encoders shown

2. **Figure X+1**: Grid vs County comparison
   - Bar plot: OLS slope for each encoder
   - Error bars from 25 reps
   - Shows robustness to geometry type

3. **Figure X+2**: Ablation study (if Moran integration done)
   - 5 configurations: baseline, encoders only, Moran only, combined
   - OLS slope and amplitude ratio metrics

## Troubleshooting

### Issue: County download fails
**Fix**: Script automatically falls back to synthetic grid

### Issue: Memory error with 3000 counties
**Fix**: Use `--mem=64G` in run_counties.bash

### Issue: GeoShapley very slow
**Fix**: Reduce num_samples in GeoShapley (default: 1000)

### Issue: Space2vec_grid breaks with counties
**Fix**: Script uses extent bounds for minmax config - should work

## Questions?

- Check `COUNTIES_README.md` for detailed documentation
- Check `MORAN_INTEGRATION_GUIDE.md` for Moran eigenvector approach
- Check original `README.md` for overall project context
- See `FINAL_EXPERIMENT_PLAN.md` for comprehensive analysis plan

---

**Status**: Ready to test! Start with `python us_counties_dgp.py` to verify setup.
