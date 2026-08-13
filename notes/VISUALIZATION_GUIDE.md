# Visualization Guide

All visualization scripts now output **PDF files** as required.

## Grid Experiments Visualization

### New Aggregating Visualizer: `visualize_grid_aggregated.py`

This script **automatically aggregates** per-repetition CSV results and creates the exact same visualization format as the original `visualize_pub_mlp.py` and `visualize_pub_xgb.py`, but without requiring precomputed `.npy` files.

**Uses formulaic DGP** (same as `gridRun.py`) to generate ground truth - no CSV input needed!

**Input:**
- Per-repetition spatial_effects CSVs: `{encoder}_{model}_rep{N}_spatial_effects.csv`
- DGP parameters (grid_size, noise_std, random_seed - must match your experiment runs)

**Output:**
- PDF with side-by-side comparison of ground truth, mean estimates, and std dev for all coefficients

### Basic Usage

```bash
# MLP visualization (default)
python visualize_grid_aggregated.py \
  --results_dir ./results/grid_nov19 \
  --model_type MLP \
  --grid_size 25 \
  --noise_std 0.1 \
  --random_seed 222 \
  --encoders tile_ffn wrap_ffn sphere2vec_dim32 \
  --output comparison_mlp_all_coefficients.pdf

# XGBoost visualization
python visualize_grid_aggregated.py \
  --results_dir ./results/grid_nov19 \
  --model_type XGBoost \
  --grid_size 25 \
  --noise_std 0.1 \
  --random_seed 222 \
  --encoders tile_ffn wrap_ffn sphere2vec_dim32 \
  --output comparison_xgboost_all_coefficients.pdf
```

### Custom Display Names

```bash
python visualize_grid_aggregated.py \
  --results_dir ./results/grid_nov19 \
  --model_type MLP \
  --encoders tile_ffn wrap_ffn sphere2vec_dim32 \
  --encoder_labels "Tile FFN" "Wrap FFN" "Sphere2Vec (32D)" \
  --output my_comparison.pdf
```

### All Available Options

```bash
python visualize_grid_aggregated.py --help
```

Options:
- `--results_dir`: Directory with per-rep spatial_effects CSVs (default: `./results/grid`)
- `--model_type`: Model to visualize: `MLP` or `XGBoost` (default: `MLP`)
- `--grid_size`: Grid dimension, e.g., 25 for 25×25 (default: `25`)
- `--noise_std`: Noise std for DGP - **must match your experiment runs** (default: `0.1`)
- `--random_seed`: Random seed for DGP - **must match your experiment runs** (default: `222`)
- `--encoders`: List of encoder names to compare (space-separated)
- `--encoder_labels`: Display labels for encoders (optional, same order as `--encoders`)
- `--output`: Output PDF filename (auto-generated if not specified)

---

## County Experiments Visualization

### Updated: `visualize_counties.py`

Now saves as **PDF** (previously PNG).

**Input:**
- Per-repetition spatial_effects CSVs: `{encoder}_{model}_rep{N}_spatial_effects.csv`
- County shapefile (auto-downloads if not provided)

**Output:**
- PDF choropleth maps comparing true, mean estimate, and std dev

### Basic Usage

```bash
# Single coefficient, multiple encoders
python visualize_counties.py \
  --results_dir ./results/counties \
  --encoders space2vec_rbf tile_ffn wrap_ffn \
  --coefficient b1 \
  --model MLP \
  --output_dir ./county_figs

# Outputs:
#   ./county_figs/space2vec_rbf_MLP_b1.pdf
#   ./county_figs/tile_ffn_MLP_b1.pdf
#   ./county_figs/wrap_ffn_MLP_b1.pdf
#   ./county_figs/comparison_MLP_b1.pdf  (multi-encoder comparison)
```

### With Custom Shapefile

```bash
python visualize_counties.py \
  --results_dir ./results/counties \
  --shapefile /path/to/counties.shp \
  --encoders space2vec_rbf tile_ffn \
  --coefficient b2 \
  --model XGBoost \
  --output_dir ./figs
```

### All Available Options

```bash
python visualize_counties.py --help
```

Options:
- `--results_dir`: Directory with per-rep spatial_effects CSVs (required)
- `--shapefile`: Path to county shapefile (optional, auto-downloads US counties if omitted)
- `--encoders`: List of encoder names to visualize (space-separated)
- `--coefficient`: Which coefficient to visualize: `b0`, `b1`, or `b2` (default: `b1`)
- `--model`: Model type: `MLP` or `XGBoost` (default: `MLP`)
- `--output_dir`: Output directory for PDF figures (default: `./county_figs`)

---

## Legacy Visualizers (require precomputed .npy files)

If you have precomputed `.npy` mean/std surfaces, you can still use:

- `visualize_pub_mlp.py` — Grid MLP results (expects `.npy` files)
- `visualize_pub_xgb.py` — Grid XGBoost results (expects `.npy` files)

These expect files like:
```
results/finalDraft/{encoder}/mlp_{encoder}_{coeff_key}_mean_surface.npy
results/finalDraft/{encoder}/mlp_{encoder}_{coeff_key}_std_surface.npy
```

Where `coeff_key` ∈ {`intercept`, `svc_x1_smooth`, `svc_x2_smooth`}

---

## Summary

| Script | Input | Output | Use Case |
|--------|-------|--------|----------|
| `visualize_grid_aggregated.py` | Per-rep CSVs | PDF (grid) | **Recommended** for grid experiments |
| `visualize_counties.py` | Per-rep CSVs | PDF (choropleth) | County-level experiments |
| `visualize_pub_mlp.py` | Precomputed .npy | PDF (grid) | Legacy / if you have .npy files |
| `visualize_pub_xgb.py` | Precomputed .npy | PDF (grid) | Legacy / if you have .npy files |

**All scripts now produce PDF output.**
