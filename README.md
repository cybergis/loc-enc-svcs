# Explainable Spatial Effects via Location Embeddings

Benchmark framework for evaluating how well spatial location encoders recover spatially-varying coefficients (SVCs) from synthetic data, using [GeoShapley](https://github.com/Ziqi-Li/geoshapley) for post-hoc spatial effect extraction.

## Overview

Many geospatial ML models use **location encoders** (e.g., Space2Vec, Sphere2Vec, NeRF) to represent coordinates as high-dimensional embeddings. But how well do these encoders actually capture spatial heterogeneity?

This project answers that by:
1. Generating synthetic datasets with **known** spatially-varying coefficients (MGWR-style DGP)
2. Training ML models (MLP, XGBoost) with different location encoders
3. Using GeoShapley to extract estimated spatial effects
4. Comparing recovered effects against ground truth across 11 encoders

## Supported Encoders

All encoders are from [TorchSpatial](https://github.com/seai-lab/TorchSpatial) (Wu et al., NeurIPS 2024):

| Encoder | Type |
|---------|------|
| Space2Vec-theory | Fourier features (theory-grounded) |
| Space2Vec-grid | Multi-scale grid cells |
| tile_ffn | Tile + feedforward |
| wrap_ffn | Wrap + feedforward |
| rff | Random Fourier features |
| NeRF | Neural radiance field positional encoding |
| Sphere2Vec-sphereC/C+ | Spherical (Cartesian) |
| Sphere2Vec-sphereM/M+ | Spherical (Mercator) |
| Sphere2Vec-dfs | Spherical (DFS) |
| none | Baseline (no spatial encoding) |

## Setup

```bash
git clone https://github.com/danielkiv/ExplainEmbeddingSpatialEffects.git
cd ExplainEmbeddingSpatialEffects
pip install -r requirements.txt
```

TorchSpatial is included in the repo under `TorchSpatial/` (MIT licensed).

## Usage

### Quick test (single encoder, 1 repetition)

```bash
# Grid experiment (25x25 synthetic grid)
python gridRun.py --encoder_index 0 --model_type MLP --num_repetitions 1

# County experiment (US counties with synthetic DGP)
python countiesRun.py --encoder_index 0 --model_type MLP --num_repetitions 1
```

### Full experiment (all encoders, 25 reps)

```bash
# Run encoder index 0-11 (see ENCODER_CONFIGS in gridRun.py)
for i in $(seq 0 11); do
    python gridRun.py --encoder_index $i --model_type MLP --num_repetitions 25 --output_dir ./results/grid
done
```

Or with SLURM:
```bash
sbatch run_all_grid.bash ./results/grid MLP 25
```

### Aggregate and visualize

```bash
# Combine per-encoder summaries
python aggregate_metrics.py

# Create figures
python visualize_grid_aggregated.py --results_dir ./results/grid --model_type MLP
python visualize_counties.py --results_dir ./results/counties --model_type MLP
```

## Project Structure

```
gridRun.py              # Grid experiment runner
countiesRun.py          # US county experiment runner
dgp_utils.py            # Synthetic data generation (MGWR-style DGP)
help_utils.py           # Location encoder wrapper + spatial metrics
aggregate_metrics.py    # Combine results across encoders
visualize_grid_aggregated.py   # Grid result visualizations
visualize_counties.py          # County choropleth visualizations
TorchSpatial/           # Location encoder implementations (MIT, Wu et al.)
tests/                  # Unit tests
```

## Key Metrics

For each spatial effect, we compute:
- **Pearson r**: Shape recovery (does the spatial pattern match?)
- **OLS slope**: Amplitude recovery (is the magnitude correct?)
- **RMSE / MAE**: Overall error
- **Moran's I**: Spatial autocorrelation in residuals

## References

- **GeoShapley**: Li, Z. (2024). GeoShapley: A Game Theory Approach to Measuring Spatial Effects in Machine Learning Models. *Annals of the AAG*.
- **TorchSpatial**: Wu, N. et al. (2024). TorchSpatial: A Location Encoding Framework and Benchmark for Spatial Representation Learning. *NeurIPS Datasets and Benchmarks*.
- **Sphere2Vec**: Mai, G. et al. (2023). Sphere2Vec: A General-Purpose Location Representation Learning over a Spherical Surface. *ISPRS*.

## License

MIT