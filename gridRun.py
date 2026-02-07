"""
Grid-based location encoder experiments with GeoShapley.
Concise version with formula-based DGP generation.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV
from flaml import AutoML
from geoshapley import GeoShapleyExplainer

from dgp_utils import create_grid_data
from help_utils import calculate_spatial_metrics, get_loc_embeddings


# Encoder configurations (all standardized to output_dim=12 for efficient GeoShapley computation)
# Using TorchSpatial encoder type names compatible with get_loc_embeddings()
ENCODER_CONFIGS = [
    {'name': 'Space2Vec-theory', 'encoder_type': 'Space2Vec-theory'},
    {'name': 'tile_ffn', 'encoder_type': 'tile_ffn'},
    {'name': 'wrap_ffn', 'encoder_type': 'wrap_ffn'},
    {'name': 'Sphere2Vec-sphereM', 'encoder_type': 'Sphere2Vec-sphereM'},
    {'name': 'Sphere2Vec-sphereM+', 'encoder_type': 'Sphere2Vec-sphereM+'},
    {'name': 'rff', 'encoder_type': 'rff'},
    {'name': 'Sphere2Vec-sphereC', 'encoder_type': 'Sphere2Vec-sphereC'},
    {'name': 'Sphere2Vec-sphereC+', 'encoder_type': 'Sphere2Vec-sphereC+'},
    {'name': 'NeRF', 'encoder_type': 'NeRF'},
    {'name': 'Sphere2Vec-dfs', 'encoder_type': 'Sphere2Vec-dfs'},
    {'name': 'Space2Vec-grid', 'encoder_type': 'Space2Vec-grid'},
    {'name': 'none', 'encoder_type': None}
]


# Parse arguments at module level (like embeddingsRun.py)
parser = argparse.ArgumentParser(description='Grid location encoder experiments')
parser.add_argument('--encoder_index', type=int, required=True)
parser.add_argument('--model_type', type=str, default='MLP', choices=['MLP', 'XGBoost'])
parser.add_argument('--num_repetitions', type=int, default=25)
parser.add_argument('--grid_size', type=int, default=25)
parser.add_argument('--noise_std', type=float, default=0.1)
parser.add_argument('--output_dir', type=str, default='./results/grid')
parser.add_argument('--random_seed', type=int, default=222)

args = parser.parse_args()

if not (0 <= args.encoder_index < len(ENCODER_CONFIGS)):
    raise ValueError(f"encoder_index must be 0-{len(ENCODER_CONFIGS)-1}")

encoder_config = ENCODER_CONFIGS[args.encoder_index]
encoder_name = encoder_config['name']
encoder_type = encoder_config['encoder_type']

output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)

all_metrics = []

# Main repetition loop at module level (NO if __name__ guard - needed for joblib pickling!)
for repetition in range(args.num_repetitions):
        print(f"\n{'='*60}")
        print(f"GRID EXPERIMENT: {encoder_name} | {args.model_type} | Rep {repetition}")
        print(f"{'='*60}")
        
        # Generate data
        rep_seed = args.random_seed + repetition
        print(f"\n[1/6] Generating synthetic grid data (seed={rep_seed})...")
        data, extent = create_grid_data(size=args.grid_size, coord_system='regional', 
                                       noise_std=args.noise_std, random_seed=rep_seed)
        
        X1 = data['X1'].values
        X2 = data['X2'].values
        coords = data[['lon', 'lat']].values
        y = data['y'].values
        
        true_b0 = data['b0'].values
        true_b1 = data['b1'].values
        true_b2 = data['b2'].values
        
        print(f"  ✓ Generated {len(data)} points")
        
        # Generate embeddings
        print(f"\n[2/6] Generating location embeddings: {encoder_name}...")
        
        if encoder_type is None:
            embeddings = np.zeros((len(data), 0))
            print("  ✓ No encoder (baseline)")
        else:
            try:
                # ALWAYS use CPU for embeddings (matching embeddingsRun.py - avoids CUDA/pickling issues)
                embeddings_result = get_loc_embeddings(
                    coords,  # Pass numpy array directly (not as keyword arg)
                    encoder_type=encoder_type,
                    extent=extent,
                    device="cpu"  # Force CPU to avoid CUDA tensor pickling issues in parallel workers
                )
                
                # Handle both tensor and non-tensor returns (matching embeddingsRun.py logic)
                if isinstance(embeddings_result, torch.Tensor):
                    embeddings = embeddings_result.detach().cpu().numpy()
                else:
                    print("  Detected non-tensor embeddings. Stacking into dense array...")
                    embeddings = np.array(embeddings_result)
                    if embeddings.ndim > 2:
                        embeddings = np.squeeze(embeddings)
                
                if embeddings.ndim != 2:
                    raise ValueError(f"Embeddings are not 2D. Shape: {embeddings.shape}")
                
                print(f"  ✓ Generated embeddings: shape {embeddings.shape}")
            except Exception as e:
                print(f"  ✗ Error generating embeddings: {e}")
                import traceback
                traceback.print_exc()
                print("  Using baseline (no embeddings).")
                embeddings = np.zeros((len(data), 0))
    
        # Prepare features (mirror notebook setup: train on X1/X2 plus coords, then add embeddings)
        print(f"\n[3/6] Preparing ML features...")
        X_base = pd.DataFrame({'X1': X1, 'X2': X2, 'lon': coords[:, 0], 'lat': coords[:, 1]})
        
        if embeddings.shape[1] > 0:
            X_embeddings = pd.DataFrame(embeddings, columns=[f'emb_{i}' for i in range(embeddings.shape[1])])
            X_ml_features = pd.concat([X_base, X_embeddings], axis=1)
        else:
            X_ml_features = X_base.copy()
        
        # Feature scaling: disabled to keep coefficients in original units
        X_ml_features_scaled = X_ml_features.copy()
        print(f"  ✓ Skipping scaling; using original feature scales")
        
        ml_feature_names = list(X_ml_features_scaled.columns)
        
        # GeoShapley expects the same feature matrix; we already included coords
        X_for_geoshapley = X_ml_features_scaled.copy()
        geoshapley_feature_names = ml_feature_names
        
        # Train/Test split (80/20) - matching embeddingsRun.py pattern
        from sklearn.model_selection import train_test_split
        print(f"\n[3.5/6] Splitting data (80% train, 20% test) with seed {rep_seed}...")
        X_train_gs, X_test_gs, y_train, y_test = train_test_split(
            X_for_geoshapley, y, test_size=0.20, random_state=rep_seed
        )
        X_train_ml = X_train_gs[ml_feature_names]
        X_test_ml = X_test_gs[ml_feature_names]
        print(f"  ✓ Train: {len(X_train_ml)} points, Test: {len(X_test_ml)} points")
        
        # Train model AT MODULE LEVEL (not in a function - critical for XGBoost pickling)
        print(f"\n[4/6] Training {args.model_type} model...")
        
        if args.model_type == 'MLP':
            param_dist = {
                'hidden_layer_sizes': [(100, 50), (150, 100), (200, 100)],
                'activation': ['relu'],
                'solver': ['adam'],
                'alpha': [10**-x for x in range(4, 7)],
                'learning_rate_init': [0.001, 0.0005],
                'max_iter': [2000]
            }
            
            search = RandomizedSearchCV(
                MLPRegressor(random_state=rep_seed),
                param_dist, n_iter=20, cv=5,
                random_state=rep_seed, n_jobs=-1
            )
            search.fit(X_train_ml, y_train)  # Train on 80% train set
            model = search.best_estimator_
            
        elif args.model_type == 'XGBoost':
            automl = AutoML()
            automl.fit(X_train_ml, y_train,  # Train on 80% train set
                      time_budget=90, metric='r2',
                      estimator_list=['xgboost'], task='regression',
                      seed=rep_seed,
                      verbose=0)
            model = automl.model.estimator  # Extract actual XGBoost model (FLAML wrapper doesn't pickle well)
        
        else:
            raise ValueError(f"Unknown model_type: {args.model_type}")
    
        # Evaluate on both train and test sets
        train_score = model.score(X_train_ml, y_train)
        test_score = model.score(X_test_ml, y_test)
        print(f"  ✓ Model Train R² = {train_score:.4f}, Test R² = {test_score:.4f}")
        
        # Calculate Moran's I on prediction residuals (model-level spatial autocorrelation check)
        y_pred = model.predict(X_test_ml)
        pred_residuals = y_test - y_pred
        
        try:
            from libpysal import weights
            from esda.moran import Moran
            
            # Get test set coordinates
            test_indices = X_test_gs.index
            coords_test = coords[test_indices]
            
            # Create spatial weights for test set
            n_test = len(coords_test)
            grid_size_test = int(np.sqrt(n_test))
            
            if grid_size_test * grid_size_test == n_test:
                # Regular grid - use lat2W
                w_test = weights.lat2W(nrows=grid_size_test, ncols=grid_size_test, rook=False)
                w_test.transform = 'r'
                moran_pred = Moran(pred_residuals, w_test, permutations=99)
                moran_i_predictions = moran_pred.I
                moran_i_predictions_pval = moran_pred.p_sim
                print(f"  ✓ Moran's I on prediction residuals: {moran_i_predictions:.4f} (p={moran_i_predictions_pval:.4f})")
            else:
                # Irregular - use KNN
                w_test = weights.KNN.from_array(coords_test, k=8)
                w_test.transform = 'r'
                moran_pred = Moran(pred_residuals, w_test, permutations=99)
                moran_i_predictions = moran_pred.I
                moran_i_predictions_pval = moran_pred.p_sim
                print(f"  ✓ Moran's I on prediction residuals: {moran_i_predictions:.4f} (p={moran_i_predictions_pval:.4f})")
        except Exception as e:
            print(f"  ⚠ Could not calculate Moran's I on predictions: {e}")
            moran_i_predictions = np.nan
            moran_i_predictions_pval = np.nan
        
        # Extract spatial effects with GeoShapley
        print(f"\n[5/6] Extracting spatial effects with GeoShapley...")
        
        try:
            # Define predict function that extracts only ML features (no coordinates)
            def predict_func(X_gs):
                """Extract ML features from full GeoShapley input (features + coordinates)"""
                if isinstance(X_gs, np.ndarray):
                    X_gs = pd.DataFrame(X_gs, columns=geoshapley_feature_names)
                X_ml = X_gs[ml_feature_names]
                return model.predict(X_ml)
            
            # GeoShapley setup following embeddingsRun.py pattern:
            # - Background: Train set (80%) WITH coordinates (prevents data leakage)
            # - Explanation: Full dataset (100%) WITH coordinates (get SVCs for all points)
            background_data_gs = X_train_gs
            explanation_data_gs = X_for_geoshapley
            
            print(f"  Background: {len(background_data_gs)} train points")
            print(f"  Explaining: {len(explanation_data_gs)} total points")
            
            explainer = GeoShapleyExplainer(
                predict_func,
                background_data_gs.values  # Train set with coordinates
            )
            
            # Explain full dataset with n_jobs=-1 (matching embeddingsRun.py structure!)
            rslt = explainer.explain(explanation_data_gs, n_jobs=-1)
        except Exception as e:
            print(f"ERROR during GeoShapley explanation: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise
    
        # Extract feature indices for X1 and X2 in the GeoShapley feature array
        x1_idx = geoshapley_feature_names.index('X1')
        x2_idx = geoshapley_feature_names.index('X2')
        
        # Extract spatially-varying coefficients using get_svc() method
        # RAW: Direct SHAP-based coefficients (already in original units since scaling is disabled)
        svc_raw = rslt.get_svc(col=[x1_idx, x2_idx], coef_type="raw", include_primary=True, coords=coords)
        shap_b1_raw = svc_raw[:, 0]
        shap_b2_raw = svc_raw[:, 1]
        
        # SMOOTH: GWR-smoothed coefficients (helps with amplitude recovery)
        svc_smooth = rslt.get_svc(col=[x1_idx, x2_idx], coef_type="gwr", include_primary=True, coords=coords)
        shap_b1_smooth = svc_smooth[:, 0]
        shap_b2_smooth = svc_smooth[:, 1]
        
        # Extract intercept with geographic component
        shap_b0 = rslt.base_value + rslt.geo
        
        print(f"  ✓ Extracted spatial effects (raw & smoothed SVCs using get_svc())")
        
        # Compute metrics
        print(f"\n[6/6] Computing metrics...")
        
        metrics_b0 = calculate_spatial_metrics(
            true_b0, shap_b0, "Intercept", encoder_name, args.model_type, coords, args.grid_size
        )
        metrics_b1_raw = calculate_spatial_metrics(
            true_b1, shap_b1_raw, "SVC_X1_Raw", encoder_name, args.model_type, coords, args.grid_size
        )
        metrics_b2_raw = calculate_spatial_metrics(
            true_b2, shap_b2_raw, "SVC_X2_Raw", encoder_name, args.model_type, coords, args.grid_size
        )
        metrics_b1_smooth = calculate_spatial_metrics(
            true_b1, shap_b1_smooth, "SVC_X1_Smooth", encoder_name, args.model_type, coords, args.grid_size
        )
        metrics_b2_smooth = calculate_spatial_metrics(
            true_b2, shap_b2_smooth, "SVC_X2_Smooth", encoder_name, args.model_type, coords, args.grid_size
        )
        
        metrics = pd.DataFrame({
            'b0': metrics_b0,
            'b1_raw': metrics_b1_raw,
            'b2_raw': metrics_b2_raw,
            'b1_smooth': metrics_b1_smooth,
            'b2_smooth': metrics_b2_smooth
        }).T
        
        metrics['encoder'] = encoder_name
        metrics['model'] = args.model_type
        metrics['repetition'] = repetition
        metrics['train_r2'] = train_score
        metrics['test_r2'] = test_score
        metrics['moran_i_predictions'] = moran_i_predictions
        metrics['moran_i_predictions_pval'] = moran_i_predictions_pval
        
        print(f"  b1_raw - Pearson r: {metrics_b1_raw['pearson_r']:.3f}, OLS slope: {metrics_b1_raw['ols_slope']:.3f}")
        print(f"  b2_raw - Pearson r: {metrics_b2_raw['pearson_r']:.3f}, OLS slope: {metrics_b2_raw['ols_slope']:.3f}")
        print(f"  b1_smooth - Pearson r: {metrics_b1_smooth['pearson_r']:.3f}, OLS slope: {metrics_b1_smooth['ols_slope']:.3f}")
        print(f"  b2_smooth - Pearson r: {metrics_b2_smooth['pearson_r']:.3f}, OLS slope: {metrics_b2_smooth['ols_slope']:.3f}")
        
        # Save results
        metrics_file = output_dir / f'{encoder_name}_{args.model_type}_rep{repetition}_metrics.csv'
        metrics.to_csv(metrics_file, index=True)
        
        spatial_effects = pd.DataFrame({
            'lon': coords[:, 0],
            'lat': coords[:, 1],
            'b0_true': true_b0,
            'b1_true': true_b1,
            'b2_true': true_b2,
            'b0_estimated': shap_b0,
            'b1_raw_estimated': shap_b1_raw,
            'b2_raw_estimated': shap_b2_raw,
            'b1_smooth_estimated': shap_b1_smooth,
            'b2_smooth_estimated': shap_b2_smooth
        })
        
        spatial_file = output_dir / f'{encoder_name}_{args.model_type}_rep{repetition}_spatial_effects.csv'
        spatial_effects.to_csv(spatial_file, index=False)
        
        print(f"\n✓ Saved results:")
        print(f"  - {metrics_file}")
        print(f"  - {spatial_file}")
        
        all_metrics.append(metrics)

# Aggregate summary
summary_file = output_dir / f'{encoder_name}_{args.model_type}_summary.csv'

all_metrics_df = pd.concat(all_metrics, ignore_index=True)
all_metrics_df.to_csv(summary_file, index=False)

print(f"\n{'='*60}")
print(f"✓ COMPLETED ALL REPETITIONS")
print(f"  Summary: {summary_file}")
print(f"{'='*60}")
