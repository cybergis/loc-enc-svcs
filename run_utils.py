"""
Shared pipeline utilities for grid, county, and global experiment runners.

Each run script (gridRun.py, countiesRun.py, globalRun.py) handles only:
  - Script-specific argparse flags
  - Data generation
  - Calling run_experiment_loop()

Everything else lives here.
"""

import argparse
import time
import traceback
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message="The total space of parameters")

import numpy as np
import pandas as pd
from pathlib import Path
import torch
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from flaml import AutoML
from geoshapley import GeoShapleyExplainer

from help_utils import calculate_spatial_metrics, get_loc_embeddings, train_loc_encoder


ENCODER_CONFIGS = [
    {'name': 'Space2Vec-theory',    'encoder_type': 'Space2Vec-theory'},
    {'name': 'tile_ffn',            'encoder_type': 'tile_ffn'},
    {'name': 'wrap_ffn',            'encoder_type': 'wrap_ffn'},
    {'name': 'Sphere2Vec-sphereM',  'encoder_type': 'Sphere2Vec-sphereM'},
    {'name': 'Sphere2Vec-sphereM+', 'encoder_type': 'Sphere2Vec-sphereM+'},
    {'name': 'rff',                 'encoder_type': 'rff'},
    {'name': 'Sphere2Vec-sphereC',  'encoder_type': 'Sphere2Vec-sphereC'},
    {'name': 'Sphere2Vec-sphereC+', 'encoder_type': 'Sphere2Vec-sphereC+'},
    {'name': 'NeRF',                'encoder_type': 'NeRF'},
    {'name': 'Sphere2Vec-dfs',      'encoder_type': 'Sphere2Vec-dfs'},
    {'name': 'Space2Vec-grid',      'encoder_type': 'Space2Vec-grid'},
    {'name': 'none',                'encoder_type': None},
]


def build_base_parser(description):
    """Return an argparse.ArgumentParser with all flags shared across run scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--encoder_index', type=int, required=True)
    parser.add_argument('--model_type', type=str, default='MLP', choices=['MLP', 'XGBoost'])
    parser.add_argument('--num_repetitions', type=int, default=25)
    parser.add_argument('--noise_std', type=float, default=0.1)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--random_seed', type=int, default=222)
    parser.add_argument('--train_encoder', action='store_true', default=False,
                        help='Train encoder before extracting embeddings')
    parser.add_argument('--encoder_epochs', type=int, default=500)
    parser.add_argument('--encoder_lr', type=float, default=1e-3)
    parser.add_argument('--embed_dim', type=int, default=4,
                        help='Embedding dimension for spatial encoders (default: 4)')
    parser.add_argument('--simple_dgp', action='store_true', default=False,
                        help='Use simple DGP (y=b0+b1*X1+b2*X2) instead of complex Li&Peng DGP')
    parser.add_argument('--no_coords', action='store_true', default=False,
                        help='Exclude lon/lat from features (embeddings only)')
    return parser


def get_embeddings(encoder_name, encoder_type, coords, X1, X2, y,
                   train_idx, args):
    """Return a 2D numpy embeddings array [N, D]. Falls back to zeros on error."""
    if encoder_type is None:
        print("  No encoder (baseline)")
        return np.zeros((len(coords), 0))

    try:
        edim = getattr(args, 'embed_dim', 4)
        if args.train_encoder:
            print(f"  Training encoder on {len(train_idx)} points "
                  f"({args.encoder_epochs} epochs, lr={args.encoder_lr}, dim={edim})...")
            trained_enc = train_loc_encoder(
                coords=coords[train_idx], X1=X1[train_idx], X2=X2[train_idx],
                y=y[train_idx], encoder_type=encoder_type, extent=args._extent,
                device="cpu", n_epochs=args.encoder_epochs, lr=args.encoder_lr,
                random_seed=args._rep_seed, embed_dim=edim,
            )
            with torch.no_grad():
                result = trained_enc(np.expand_dims(coords, axis=1))
        else:
            result = get_loc_embeddings(
                coords, encoder_type=encoder_type,
                extent=args._extent, device="cpu", embed_dim=edim,
            )

        if isinstance(result, torch.Tensor):
            embeddings = result.detach().cpu().numpy()
        else:
            print("  Detected non-tensor embeddings. Stacking into dense array...")
            embeddings = np.array(result)

        if embeddings.ndim > 2:
            embeddings = embeddings.squeeze(axis=1)

        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings not 2D after squeeze. Shape: {embeddings.shape}")

        print(f"  Generated embeddings: shape {embeddings.shape}")
        return embeddings

    except Exception as e:
        print(f"  Error generating embeddings: {e}")
        traceback.print_exc()
        print("  Falling back to baseline (no embeddings).")
        return np.zeros((len(coords), 0))


def prepare_features(X1, X2, coords, embeddings, extent, no_coords=False):
    """Build ML feature DataFrame from covariates, coordinates, and embeddings.

    Column order: [X1, X2, emb_0..emb_D, lon, lat]
    Non-geo features (X1, X2) come first; all location-derived features
    (embeddings + lon/lat) are grouped at the end for GeoShapley's g parameter.
    If no_coords=True, lon/lat are excluded (embeddings only).

    Coordinates are normalized to [0, 1] using extent, matching the DGP's
    coordinate system for generating true SVCs.
    """
    parts = [pd.DataFrame({'X1': X1, 'X2': X2})]
    if embeddings.shape[1] > 0:
        parts.append(pd.DataFrame(embeddings, columns=[f'emb_{i}' for i in range(embeddings.shape[1])]))
    if not no_coords:
        lon_min, lon_max, lat_min, lat_max = extent
        lon_norm = (coords[:, 0] - lon_min) / (lon_max - lon_min)
        lat_norm = (coords[:, 1] - lat_min) / (lat_max - lat_min)
        parts.append(pd.DataFrame({'lon': lon_norm, 'lat': lat_norm}))
    return pd.concat(parts, axis=1)


def train_ml_model(model_type, X_train, y_train, rep_seed):
    """Train and return a fitted sklearn-compatible model."""
    if model_type == 'MLP':
        param_dist = {
            'hidden_layer_sizes': [(100, 50), (150, 100), (200, 100)],
            'activation': ['relu'],
            'solver': ['adam'],
            'alpha': [10**-x for x in range(4, 7)],
            'learning_rate_init': [0.001, 0.0005],
            'max_iter': [2000],
        }
        search = RandomizedSearchCV(
            MLPRegressor(random_state=rep_seed),
            param_dist, n_iter=20, cv=5,
            random_state=rep_seed, n_jobs=-1,
        )
        search.fit(X_train, y_train)
        return search.best_estimator_

    elif model_type == 'XGBoost':
        automl = AutoML()
        automl.fit(X_train, y_train, time_budget=90, metric='r2',
                   estimator_list=['xgboost'], task='regression',
                   seed=rep_seed, verbose=0)
        return automl.model.estimator

    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def compute_moran_predictions(y_test, y_pred, coords_test, grid_size=None):
    """Compute Moran's I on prediction residuals. Returns (I, p-value)."""
    try:
        from libpysal import weights
        from esda.moran import Moran

        residuals = y_test - y_pred
        n = len(coords_test)

        if grid_size is not None and n == grid_size * grid_size:
            w = weights.lat2W(nrows=grid_size, ncols=grid_size, rook=False)
        else:
            w = weights.KNN.from_array(coords_test, k=min(8, n - 1))
        w.transform = 'r'

        moran = Moran(residuals, w, permutations=99)
        print(f"  Moran's I on prediction residuals: {moran.I:.4f} (p={moran.p_sim:.4f})")
        return moran.I, moran.p_sim

    except Exception as e:
        print(f"  Warning: Could not calculate Moran's I on predictions: {e}")
        return np.nan, np.nan


def run_geoshapley(model, X_train_gs, X_for_geoshapley, feature_names, coords):
    """Run GeoShapley and return (shap_b0, shap_b1_raw, shap_b2_raw, shap_b1_smooth, shap_b2_smooth)."""
    def predict_func(X):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if isinstance(X, np.ndarray):
                return model.predict(X)
            return model.predict(X[feature_names].values)

    # Count geo features: all emb_* columns + lon/lat (grouped at end of feature list)
    n_emb = sum(1 for f in feature_names if f.startswith('emb_'))
    n_coord = sum(1 for f in feature_names if f in ('lon', 'lat'))
    n_geo = n_emb + n_coord

    # Subsample background to cap memory (standard SHAP practice)
    max_bg = 200
    if len(X_train_gs) > max_bg:
        bg = X_train_gs.sample(n=max_bg, random_state=42).values
    else:
        bg = X_train_gs.values
    print(f"  Background: {len(bg)} points (from {len(X_train_gs)} train)")
    print(f"  Geo features grouped: {n_geo} (emb_* + lon + lat)")
    print(f"  Explaining: {len(X_for_geoshapley)} total points")

    explainer = GeoShapleyExplainer(predict_func, bg, g=n_geo)
    rslt = explainer.explain(X_for_geoshapley, n_jobs=-1)

    x1_idx = feature_names.index('X1')
    x2_idx = feature_names.index('X2')

    svc_raw = rslt.get_svc(col=[x1_idx, x2_idx], coef_type="raw", include_primary=True, coords=coords)
    svc_smooth = rslt.get_svc(col=[x1_idx, x2_idx], coef_type="gwr", include_primary=True, coords=coords)

    return (
        rslt.base_value + rslt.geo,
        svc_raw[:, 0], svc_raw[:, 1],
        svc_smooth[:, 0], svc_smooth[:, 1],
    )


def run_experiment_loop(args, data_fn, experiment_label, grid_size=None,
                        extra_spatial_cols_fn=None):
    """
    Main repetition loop shared by all run scripts.

    Args:
        args:                 parsed argparse namespace
        data_fn:              callable(rep_seed) -> (DataFrame, extent)
                              DataFrame must have columns: X1, X2, lon, lat, y, b0, b1, b2
        experiment_label:     string printed in headers, e.g. 'GRID', 'COUNTY', 'GLOBAL'
        grid_size:            int or None; used for lat2W Moran's I on regular grids
        extra_spatial_cols_fn: optional callable(data_df) -> dict of extra columns
                               to include in the spatial_effects CSV (e.g. GEOID, NAME)
    """
    if not (0 <= args.encoder_index < len(ENCODER_CONFIGS)):
        raise ValueError(f"encoder_index must be 0-{len(ENCODER_CONFIGS)-1}")

    encoder_config = ENCODER_CONFIGS[args.encoder_index]
    encoder_name = encoder_config['name']
    encoder_type = encoder_config['encoder_type']

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    all_metrics = []
    total_start = time.time()

    for repetition in range(args.num_repetitions):
        rep_start = time.time()
        print(f"\n{'='*60}")
        print(f"{experiment_label} EXPERIMENT: {encoder_name} | {args.model_type} | Rep {repetition}")
        print(f"{'='*60}")

        rep_seed = args.random_seed + repetition

        # [1] Generate data
        print(f"\n[1/6] Generating synthetic data (seed={rep_seed})...")
        data, extent = data_fn(rep_seed)
        X1 = data['X1'].values
        X2 = data['X2'].values
        coords = data[['lon', 'lat']].values
        y = data['y'].values
        true_b0, true_b1, true_b2 = data['b0'].values, data['b1'].values, data['b2'].values
        print(f"  Generated {len(data)} points")

        train_idx, _ = train_test_split(np.arange(len(data)), test_size=0.20, random_state=rep_seed)

        # Stash per-rep state onto args so get_embeddings can access it without extra params
        args._extent = extent
        args._rep_seed = rep_seed

        # [2] Embeddings
        print(f"\n[2/6] Generating location embeddings: {encoder_name}...")
        embeddings = get_embeddings(encoder_name, encoder_type, coords, X1, X2, y, train_idx, args)

        # [3] Feature prep
        print(f"\n[3/6] Preparing ML features...")
        X_features = prepare_features(X1, X2, coords, embeddings, extent,
                                       no_coords=getattr(args, 'no_coords', False))
        feature_names = list(X_features.columns)

        # [3.5] Split
        print(f"\n[3.5/6] Splitting data (80/20, seed={rep_seed})...")
        X_train_gs, X_test_gs, y_train, y_test = train_test_split(
            X_features, y, test_size=0.20, random_state=rep_seed
        )
        X_train_ml = X_train_gs[feature_names]
        X_test_ml  = X_test_gs[feature_names]
        print(f"  Train: {len(X_train_ml)}, Test: {len(X_test_ml)}")

        # [4] Train model
        print(f"\n[4/6] Training {args.model_type} model...")
        model = train_ml_model(args.model_type, X_train_ml, y_train, rep_seed)
        train_score = model.score(X_train_ml, y_train)
        test_score  = model.score(X_test_ml,  y_test)
        print(f"  Model Train R² = {train_score:.4f}, Test R² = {test_score:.4f}")

        coords_test = coords[X_test_gs.index]
        moran_i_pred, moran_i_pred_pval = compute_moran_predictions(
            y_test, model.predict(X_test_ml), coords_test, grid_size=grid_size
        )

        # [5] GeoShapley
        n_geo_feats = sum(1 for f in feature_names if f.startswith('emb_') or f in ('lon', 'lat'))
        if n_geo_feats == 0:
            print(f"\n[5/6] Skipping GeoShapley (no geo features — none encoder + no_coords)")
            n = len(X_features)
            shap_b0 = shap_b1_raw = shap_b2_raw = shap_b1_smooth = shap_b2_smooth = np.full(n, np.nan)
        else:
            print(f"\n[5/6] Extracting spatial effects with GeoShapley...")
            try:
                shap_b0, shap_b1_raw, shap_b2_raw, shap_b1_smooth, shap_b2_smooth = run_geoshapley(
                    model, X_train_gs, X_features, feature_names, coords
                )
                print(f"  Extracted spatial effects (raw & smoothed SVCs)")
            except Exception as e:
                print(f"ERROR during GeoShapley explanation: {type(e).__name__}: {e}")
                traceback.print_exc()
                raise

        # [6] Metrics
        print(f"\n[6/6] Computing metrics...")
        metrics_dict = {
            'b0':        calculate_spatial_metrics(true_b0, shap_b0,        "Intercept",    encoder_name, args.model_type, coords, grid_size),
            'b1_raw':    calculate_spatial_metrics(true_b1, shap_b1_raw,    "SVC_X1_Raw",   encoder_name, args.model_type, coords, grid_size),
            'b2_raw':    calculate_spatial_metrics(true_b2, shap_b2_raw,    "SVC_X2_Raw",   encoder_name, args.model_type, coords, grid_size),
            'b1_smooth': calculate_spatial_metrics(true_b1, shap_b1_smooth, "SVC_X1_Smooth",encoder_name, args.model_type, coords, grid_size),
            'b2_smooth': calculate_spatial_metrics(true_b2, shap_b2_smooth, "SVC_X2_Smooth",encoder_name, args.model_type, coords, grid_size),
        }
        metrics = pd.DataFrame(metrics_dict).T
        rep_elapsed = time.time() - rep_start
        print(f"  Rep {repetition} completed in {rep_elapsed:.1f}s ({rep_elapsed/60:.1f}m)")

        metrics['encoder']          = encoder_name
        metrics['model']            = args.model_type
        metrics['repetition']       = repetition
        metrics['embed_dim']        = getattr(args, 'embed_dim', 4)
        metrics['train_r2']         = train_score
        metrics['test_r2']          = test_score
        metrics['encoder_trained']  = args.train_encoder
        metrics['encoder_epochs']   = args.encoder_epochs if args.train_encoder else 0
        metrics['rep_time_sec']     = rep_elapsed
        metrics['moran_i_predictions']      = moran_i_pred
        metrics['moran_i_predictions_pval'] = moran_i_pred_pval

        for label, key in [('b1_raw', 'b1_raw'), ('b2_raw', 'b2_raw'),
                           ('b1_smooth', 'b1_smooth'), ('b2_smooth', 'b2_smooth')]:
            r = metrics_dict[key]['pearson_r'] if metrics_dict[key] else float('nan')
            s = metrics_dict[key]['ols_slope'] if metrics_dict[key] else float('nan')
            print(f"  {label} - Pearson r: {r:.3f}, OLS slope: {s:.3f}")

        # Save
        metrics_file = output_dir / f'{encoder_name}_{args.model_type}_rep{repetition}_metrics.csv'
        metrics.to_csv(metrics_file, index=True)

        spatial_cols = {'lon': coords[:, 0], 'lat': coords[:, 1],
                        'b0_true': true_b0, 'b1_true': true_b1, 'b2_true': true_b2,
                        'b0_estimated': shap_b0,
                        'b1_raw_estimated': shap_b1_raw,   'b2_raw_estimated': shap_b2_raw,
                        'b1_smooth_estimated': shap_b1_smooth, 'b2_smooth_estimated': shap_b2_smooth}
        if extra_spatial_cols_fn is not None:
            spatial_cols.update(extra_spatial_cols_fn(data))
        spatial_file = output_dir / f'{encoder_name}_{args.model_type}_rep{repetition}_spatial_effects.csv'
        pd.DataFrame(spatial_cols).to_csv(spatial_file, index=False)

        print(f"\n  Saved: {metrics_file.name}, {spatial_file.name}")
        all_metrics.append(metrics)

    # Summary
    total_elapsed = time.time() - total_start
    summary_file = output_dir / f'{encoder_name}_{args.model_type}_summary.csv'
    pd.concat(all_metrics, ignore_index=True).to_csv(summary_file, index=False)
    print(f"\n{'='*60}\nCOMPLETED ALL REPETITIONS in {total_elapsed:.1f}s ({total_elapsed/60:.1f}m)\n  Summary: {summary_file}\n{'='*60}")
