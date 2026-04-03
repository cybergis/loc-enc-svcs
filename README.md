# Explainable Spatial Effects via Location Embeddings

Benchmark framework for evaluating how well spatial location encoders recover spatially-varying coefficients (SVCs) from synthetic data, using [GeoShapley](https://github.com/Ziqi-Li/geoshapley) for post-hoc spatial effect extraction.

## Overview

Many geospatial ML models use **location encoders** (e.g., Space2Vec, Sphere2Vec, NeRF) to represent coordinates as high-dimensional embeddings. But how well do these encoders actually capture spatial heterogeneity?

This project answers that by:
1. Generating synthetic datasets with **known** spatially-varying coefficients (MGWR-style DGP)
2. Training ML models (MLP, XGBoost) with different location encoders
3. Using GeoShapley to extract estimated spatial effects
4. Comparing recovered effects against ground truth across 11 encoders

Three encoder conditions are compared:
- **Untrained** — random-initialized encoder weights
- **Contrastively trained** — spatial contrastive loss (cosine similarity matched to Gaussian kernel of great-circle distance)
- **Pretrained** — TorchSpatial task-supervised weights (iNat species classification)

## Supported Encoders

All encoders are from [TorchSpatial](https://github.com/seai-lab/TorchSpatial) (Wu et al., NeurIPS 2024):

| Encoder | Type | Tier |
|---------|------|------|
| Sphere2Vec-dfs | Spherical (DFS) | 1 — works untrained |
| Sphere2Vec-sphereM+ | Spherical (Mercator+) | 1 — works untrained |
| wrap_ffn | Wrap + feedforward | 1 — works untrained |
| Sphere2Vec-sphereM | Spherical (Mercator) | 2 — benefits from training |
| Sphere2Vec-sphereC/C+ | Spherical (Cartesian) | 2 — benefits from training |
| Space2Vec-theory | Fourier features (theory-grounded) | 2 — benefits from training |
| Space2Vec-grid | Multi-scale grid cells | 2 — benefits from training |
| NeRF | Neural radiance field positional encoding | 2 — limited training gain |
| rff | Random Fourier features | 3 — architecturally limited |
| tile_ffn | Tile + feedforward | 3 — architecturally limited |
| none | Baseline (no spatial encoding) | — |

## Setup

```bash
git clone https://github.com/danielkiv/loc-enc-svcs.git
cd loc-enc-svcs
pip install -e .
```

This installs all dependencies including [TorchSpatial](https://github.com/danielkiv/TorchSpatial) (pulled automatically from GitHub) and `mgwr` for GWR-smoothed SVC estimation.

For development (includes pytest):

```bash
pip install -e ".[dev]"
```

## Usage

### Quick test (single encoder, 1 repetition)

```bash
# Grid experiment (25x25 synthetic grid)
python gridRun.py --encoder_index 0 --model_type MLP --num_repetitions 1 --output_dir ./results/test

# County experiment (US counties with synthetic DGP)
python countiesRun.py --encoder_index 0 --model_type MLP --num_repetitions 1 --output_dir ./results/test

# Global experiment (spherical DGP — main benchmark scale)
python globalRun.py --encoder_index 0 --model_type MLP --num_repetitions 1 --output_dir ./results/test
```

### Full experiment via SLURM

```bash
# All 12 encoders, 25 reps — untrained condition
sbatch --job-name=global run_experiments.bash global ./results/global_simple_dim8 MLP 25

# Contrastively trained condition
sbatch --job-name=global_trained run_experiments.bash global ./results/global_simple_trained_dim8 MLP 25 "--train_encoder --encoder_epochs 500"

# Emb-only variant (no raw coordinates passed to model)
sbatch --job-name=global_embonly run_experiments.bash global ./results/global_simple_embonly_dim8 MLP 25 "--no_coords"

# TorchSpatial pretrained weights (encoders 0,9,10 have clean inat_2018 checkpoints)
sbatch --job-name=global_pretrained --array=0,9,10 run_pretrained.bash global ./results/global_simple_pretrained_dim8 MLP 5
```

### Aggregate and visualize

```bash
# Combine per-encoder summaries across result directories
python aggregate_metrics.py

# Figures
python visualize_paper.py
python visualize_grid_aggregated.py --results_dir ./results/global_simple_dim8 --model_type MLP
python visualize_counties.py --results_dir ./results/county_simple_dim8 --model_type MLP
```

## Project Structure

```
gridRun.py                   # Grid experiment runner (25×25 regional grid)
countiesRun.py               # US county experiment runner
globalRun.py                 # Global experiment runner (spherical DGP)
run_utils.py                 # Shared CLI args, encoder loading, experiment loop
dgp_utils.py                 # Synthetic data generation (MGWR-style DGP)
help_utils.py                # Location encoder wrapper, contrastive training,
                             #   pretrained weight loader, spatial metrics
aggregate_metrics.py         # Combine results across encoders and directories
run_experiments.bash         # SLURM array script for untrained/trained/emb-only
run_pretrained.bash          # SLURM array script for TorchSpatial pretrained weights
visualize_paper.py           # Publication figures
visualize_grid_aggregated.py # Grid result heatmaps
visualize_counties.py        # County choropleth visualizations
tests/                       # Unit tests (pytest)
```

## Key CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--encoder_index` | required | 0–11, selects encoder from ENCODER_CONFIGS |
| `--model_type` | `MLP` | `MLP` or `XGBoost` |
| `--num_repetitions` | 25 | Number of random seeds |
| `--train_encoder` | off | Enable spatial contrastive training |
| `--pretrained_weights` | None | Path to TorchSpatial `.pth.tar` checkpoint |
| `--no_coords` | off | Exclude raw lon/lat (emb-only condition) |
| `--embed_dim` | 4 | Embedding dimension fed to ML model |
| `--encoder_epochs` | 500 | Contrastive training epochs |
| `--noise_std` | 0.1 | DGP noise level |

`--train_encoder` and `--pretrained_weights` are mutually exclusive.

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
