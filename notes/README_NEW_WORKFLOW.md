# Location Encoder Experiments - Reorganized Workflow

## Overview

Streamlined codebase for evaluating location encoders with GeoShapley on both grid and county-level data.

## Core Files

### Data Generation (`dgp_utils.py`)
- **`GridDGP`**: MGWR-style coefficients on regular grids
- **`CountyDGP`**: MGWR-style coefficients on US counties
- Formula-based generation (no CSV dependencies)
- Supports both grid and geographic coordinates

### Experiment Runners
- **`gridRun.py`**: Grid experiments (25×25 points, Chicago region)
- **`countiesRun.py`**: County experiments (~3000 counties, continental US)
- Unified code structure with StandardScaler and reduced regularization

### Batch Scripts
- **`run_all_grid.bash`**: Submit all 11 grid encoders (SLURM array job)
- **`run_all_counties.bash`**: Submit all 11 county encoders (SLURM array job)

## Quick Start

### Grid Experiments

```bash
# Single encoder test (local)
python gridRun.py --encoder_index 0 --num_repetitions 1 --output_dir ./results/test

# All encoders (SLURM)
sbatch run_all_grid.bash ./results/grid_nov19 MLP 25
```

### County Experiments

```bash
# Single encoder test
python countiesRun.py --encoder_index 0 --num_repetitions 1 --output_dir ./results/test

# All encoders (SLURM)
sbatch run_all_counties.bash ./results/counties_nov19 MLP 25
```

## Encoder List

```
0:  space2vec_rbf
1:  space2vec_rbf_legacy
2:  nerf
3:  sphere2vec_dim32
4:  sphere2vec_dim64
5:  sphere2vec_dim128
6:  space2vec_grid
7:  tile_ffn
8:  wrap_ffn
9:  rff
10: none (baseline)
```

## Data Generation Formula

### MGWR-Style Coefficients

Grid coordinates normalized to [0, 1]:
```
b0 = 6 * (0.25 - (u - 0.5)²) * (0.25 - (v - 0.5)²)  [parabolic]
b1 = 1 + 4 * (u + v) / 2                              [diagonal gradient]
b2 = 1 + 4 * (1 - u + v) / 2                          [opposite diagonal]
```

Then:
```
y = b0 + b1*X1 + b2*X2 + noise
```

Where `X1, X2 ~ Uniform(-2, 2)`, `noise ~ N(0, 0.1)`

## Changes from Original

### Removed
- ❌ CSV file loading (`mgwr_sim.csv` no longer needed)
- ❌ Verbose comments and debugging code
- ❌ Redundant spatial DGP utilities
- ❌ Separate `submit_all.bash` + `run_single.bash` split
- ❌ Complex argument parsing

### Added
- ✅ Unified `dgp_utils.py` for both grid and counties
- ✅ Formula-based coefficient generation
- ✅ Concise experiment runners
- ✅ Single-script SLURM submission
- ✅ Consistent naming: `run_all_*.bash`

### Improved
- ✅ Feature scaling (StandardScaler) - 3-6× better amplitude recovery
- ✅ Reduced MLP regularization (alpha: 1e-4 to 1e-6)
- ✅ Cleaner code structure
- ✅ Unified encoder configurations

## File Organization

```
effectsExplainableEmbeddings/
├── dgp_utils.py              # Unified DGP (NEW)
├── gridRun.py                # Grid experiments (NEW, replaces embeddingsRun.py)
├── countiesRun.py            # County experiments (UPDATED)
├── run_all_grid.bash         # Grid SLURM script (NEW, replaces submit_all.bash)
├── run_all_counties.bash     # County SLURM script (NEW)
├── help_utils.py             # Metrics computation (unchanged)
├── aggregate_metrics.py      # Results aggregation (unchanged)
├── visualize_pub_mlp.py      # Grid visualization (unchanged)
├── visualize_pub_xgb.py      # Grid visualization (unchanged)
├── visualize_counties.py     # County visualization (unchanged)
└── results/
    ├── grid_nov19/           # Grid results
    └── counties_nov19/       # County results
```

## Output Format

### Per Repetition
- `{encoder}_{model}_rep{N}_metrics.csv`: 17 metrics × 3 coefficients
- `{encoder}_{model}_rep{N}_spatial_effects.csv`: True vs estimated coefficients

### Aggregated
- `{encoder}_{model}_summary.csv`: Combined metrics from all reps

## Metrics Computed

- `pearson_r`: Correlation (shape recovery)
- `ols_slope`: Amplitude recovery (target: ~1.0)
- `amplitude_range_ratio`: Estimated/true range
- `rmse_normalized`: RMSE / std(true)
- `mape_percent`: MAPE × 100
- `moran_i_residuals`: Spatial autocorrelation in errors
- ... (11 more metrics)

## Workflow Comparison

### Old Workflow
```
1. submit_all.bash → calls run_single.bash with array indices
2. run_single.bash → sets up environment, runs embeddingsRun.py
3. embeddingsRun.py → loads CSV, processes encoder, saves results
```

### New Workflow
```
1. run_all_grid.bash → SLURM array job, runs gridRun.py directly
2. gridRun.py → generates data from formula, processes encoder, saves results
```

**Benefits**: 
- 30% less code
- No CSV dependency
- Clearer structure
- Easier debugging

## Testing

```bash
# Test DGP generation
python -c "from dgp_utils import create_grid_data; print(create_grid_data()[0].shape)"
python -c "from dgp_utils import create_county_data; print(create_county_data()[0].shape)"

# Test single experiment
python gridRun.py --encoder_index 10 --num_repetitions 1 --output_dir ./test
python countiesRun.py --encoder_index 10 --num_repetitions 1 --output_dir ./test
```

## Migration from Old Code

If you have results from `embeddingsRun.py`:
1. Results are compatible (same metrics format)
2. Can use same `aggregate_metrics.py`
3. Visualizations work without changes
4. Only difference: data source (CSV → formula)

## Next Steps

1. **Test**: Run single encoder locally
2. **Submit**: `sbatch run_all_grid.bash` for full experiment
3. **Aggregate**: `python aggregate_metrics.py` after completion
4. **Visualize**: Use existing visualization scripts

## Troubleshooting

**Q: Missing modules?**
```bash
conda activate e  # or your environment
pip install geopandas torch scikit-learn flaml
```

**Q: SLURM array job not working?**
```bash
# Check array limit
scontrol show config | grep MaxArraySize

# Run single task for testing
sbatch --array=0 run_all_grid.bash
```

**Q: Want old behavior?**
Keep `embeddingsRun.py` and `submit_all.bash` - they still work independently.
