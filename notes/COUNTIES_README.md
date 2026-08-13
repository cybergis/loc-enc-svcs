# US Counties Location Encoder Experiments

This directory contains code for running location encoder experiments on US county-level synthetic data with spatially-varying coefficients.

## Overview

**Goal**: Evaluate location encoders' ability to recover spatially-varying relationships when applied to irregular geographic units (counties) instead of regular grids.

**Key Difference from Grid Experiments**: Counties have irregular shapes and sizes, making this a more realistic test of location encoders for real-world applications.

## Files

### Core Scripts

1. **`us_counties_dgp.py`** - Synthetic data generator
   - Loads ~3000 US counties (continental US)
   - Generates spatially-varying coefficients: b0(lon, lat), b1(lon, lat), b2(lon, lat)
   - Supports multiple DGP types: gradient, quadratic, latitude, distance
   - Compatible with location encoders (uses county centroids)

2. **`countiesRun.py`** - Main experiment runner
   - Adapted from `embeddingsRun.py` for county geometries
   - Includes feature scaling (StandardScaler) and reduced regularization
   - Supports all 11 location encoders
   - Extracts spatial effects with GeoShapley
   - Saves metrics and spatial effects per repetition

3. **`visualize_counties.py`** - Choropleth map visualizations
   - Creates 3-panel maps: True | Mean Estimate | Std Dev
   - Multi-encoder comparison figures
   - Publication-quality choropleth maps

4. **`run_counties.bash`** - SLURM batch script
   - Runs all 11 encoders in parallel (array job)
   - 25 repetitions per encoder (default)
   - 4 hours, 32GB RAM, 8 CPUs per job

## Quick Start

### 1. Generate Sample Dataset

```python
from us_counties_dgp import create_us_counties_dataset

# Quick test with synthetic counties
data, extent, counties = create_us_counties_dataset(
    shapefile_path=None,  # Will create synthetic grid
    dgp_type='gradient',
    n_features=2,
    noise_std=0.1
)

print(data.head())
```

### 2. Run Single Experiment (Local)

```bash
# Test one encoder with 1 repetition
python countiesRun.py \
    --encoder_index 0 \
    --model_type MLP \
    --num_repetitions 1 \
    --output_dir ./results/test_counties
```

### 3. Run All Encoders (SLURM)

```bash
# Submit array job for all 11 encoders
sbatch run_counties.bash ./results/counties_gradient_nov17 MLP 25
```

This will run:
- 11 encoders × 25 repetitions = 275 experiments
- Expected time: ~2-3 hours per encoder

### 4. Visualize Results

```bash
# After experiments complete
python visualize_counties.py \
    --results_dir ./results/counties_gradient_nov17 \
    --encoders space2vec_rbf tile_ffn wrap_ffn \
    --coefficient b1 \
    --model MLP \
    --output_dir ./county_figs
```

## Data Generation Parameters

### DGP Types

1. **`gradient`** (default)
   - Linear gradients across space
   - b0: combination of lon + lat
   - b1: varies with longitude
   - b2: varies with latitude

2. **`quadratic`**
   - Parabolic patterns (like MGWR)
   - b0: peak at center, decreases toward edges
   - b1, b2: linear gradients

3. **`latitude`**
   - Temperature-like pattern
   - Coefficients vary primarily with latitude
   - Includes sinusoidal longitude variation

4. **`distance`**
   - Radiating from geographic center
   - Decreases with distance from center point

### County Geometries

**Real Counties** (if shapefile available):
```python
data, extent, counties = create_us_counties_dataset(
    shapefile_path='/path/to/counties.shp',
    dgp_type='gradient'
)
```

**Synthetic Counties** (fallback):
- If shapefile not available, creates ~3000 random points
- Covers continental US bounding box
- Still tests irregular spatial patterns

## Expected Results

### Amplitude Recovery Metrics

After feature scaling improvements (from grid experiments), expect:

| Encoder | OLS Slope (Grid) | OLS Slope (Counties) | Notes |
|---------|------------------|----------------------|-------|
| space2vec_rbf | 0.10 → ~0.4 | ~0.3-0.5 | May be slightly lower for irregular geometries |
| space2vec_grid | 0.10 → ~0.4 | ~0.3-0.5 | Grid config uses extent bounds |
| tile_ffn | 0.08 → ~0.3 | ~0.25-0.45 | Tile boundaries less regular |
| wrap_ffn | 0.07 → ~0.3 | ~0.25-0.45 | Similar to tile_ffn |
| nerf | 0.05 → ~0.2 | ~0.15-0.35 | Lower expected |
| sphere2vec | 0.04 → ~0.2 | ~0.15-0.35 | Lower expected |
| none (baseline) | 0.03 → ~0.1 | ~0.1-0.2 | No spatial info |

### Why Counties May Differ

1. **Irregular Sampling**: Counties vary in size/shape
2. **Spatial Autocorrelation**: Adjacent counties share borders
3. **Centroid Approximation**: Using centroids instead of full geometry
4. **GeoShapley Grid**: Designed for regular grids, adapted for counties

## Comparison to Grid Experiments

### Similarities
- Same location encoders
- Same feature scaling approach (StandardScaler)
- Same MLP regularization settings
- Same GeoShapley extraction
- Same metrics (Pearson r, OLS slope, amplitude ratio, Moran's I)

### Differences
- **Geometry**: Irregular counties vs. 25×25 regular grid
- **Sample Size**: ~3000 counties vs. 625 grid points
- **Spatial Pattern**: Real-world county layout vs. uniform grid
- **Visualization**: Choropleth maps vs. grid heatmaps
- **Computation**: Longer runtime (more data points)

## Integration with Moran Eigenvector Approach

To combine location encoders with Moran eigenvectors (from `US_all.ipynb`):

```python
# 1. Generate base dataset
from us_counties_dgp import create_us_counties_dataset
data, extent, counties = create_us_counties_dataset()

# 2. Compute Moran eigenvectors
import pysal as ps
from sklearn.linear_model import LassoCV

# Create spatial weights
w = ps.lib.weights.Queen.from_dataframe(counties)
w_exp = ps.lib.weights.DistanceBand.from_dataframe(counties, threshold=100)

# Generate Moran eigenvectors (implementation needed)
# ... (see US_all.ipynb for details)

# 3. Add to features
X_with_moran = pd.concat([data[['X1', 'X2']], moran_eigenvectors], axis=1)

# 4. Add location embeddings
# ... (as in countiesRun.py)

# 5. Train model with all features
```

## Troubleshooting

### Issue: Counties shapefile download fails
**Solution**: Script will fallback to synthetic county grid (3000 points)

### Issue: GeoShapley error with irregular geometries
**Solution**: Using county centroids as "grid" approximation - should work but may be less accurate than regular grids

### Issue: Memory error with 3000 counties
**Solution**: Reduce number of counties in `_create_synthetic_county_grid()` or use SLURM job with more memory (32GB default)

### Issue: Space2vec_grid requires grid config
**Solution**: Script automatically uses extent bounds for minmax_lat/minmax_lon

## Next Steps

1. **Run experiments**: Submit SLURM job for all encoders
2. **Analyze results**: Compare amplitude recovery to grid experiments
3. **Add Moran eigenvectors**: Integrate spatial weight matrix approach
4. **Visualize**: Create publication figures with choropleth maps
5. **Write up**: Document county-specific findings in paper

## Questions?

See main README.md for overall project documentation, or check:
- Grid experiments: `embeddingsRun.py`, `start.bash`
- Visualizations: `visualize_pub_mlp.py`, `visualize_pub_xgb.py`
- Metrics: `help_utils.py`, `aggregate_metrics.py`
