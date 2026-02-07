"""
Aggregating Grid Visualization Script
Reads per-repetition spatial_effects CSVs, computes mean/std, and creates
the exact same PDF visualization format as visualize_pub_mlp.py and visualize_pub_xgb.py

Uses formulaic DGP (same as gridRun.py) to generate ground truth - no CSV needed.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple
import argparse

from dgp_utils import create_grid_data


def load_and_aggregate_results(
    results_dir: Path,
    encoder_name: str,
    model_type: str,
    grid_size: int = 25
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load per-repetition spatial_effects CSVs and aggregate to mean/std surfaces.
    
    Args:
        results_dir: Directory containing results
        encoder_name: Name of encoder (e.g., 'tile_ffn')
        model_type: 'MLP' or 'XGBoost'
        grid_size: Grid dimension (for reshaping)
        
    Returns:
        Dictionary with coefficient keys, each containing 'mean' and 'std' arrays
    """
    # Find all rep files for this encoder+model
    pattern = f"{encoder_name}_{model_type}_rep*_spatial_effects.csv"
    rep_files = sorted(results_dir.glob(pattern))
    
    if not rep_files:
        raise FileNotFoundError(
            f"No spatial_effects files found for {encoder_name} {model_type}\n"
            f"  Searched pattern: {pattern} in {results_dir}"
        )
    
    print(f"  Found {len(rep_files)} repetitions for {encoder_name} {model_type}")
    
    # Load all repetitions
    all_b0 = []
    all_b1 = []
    all_b2 = []
    
    for rep_file in rep_files:
        df = pd.read_csv(rep_file)
        
        # Sort by lat, lon to ensure consistent ordering
        df = df.sort_values(['lat', 'lon']).reset_index(drop=True)
        
        all_b0.append(df['b0_estimated'].values)
        all_b1.append(df['b1_estimated'].values)
        all_b2.append(df['b2_estimated'].values)
    
    # Stack and compute statistics
    all_b0 = np.array(all_b0)  # shape: (n_reps, n_points)
    all_b1 = np.array(all_b1)
    all_b2 = np.array(all_b2)
    
    # Compute mean and std across repetitions
    b0_mean = np.mean(all_b0, axis=0)
    b0_std = np.std(all_b0, axis=0)
    
    b1_mean = np.mean(all_b1, axis=0)
    b1_std = np.std(all_b1, axis=0)
    
    b2_mean = np.mean(all_b2, axis=0)
    b2_std = np.std(all_b2, axis=0)
    
    # Reshape to grid
    try:
        result = {
            'intercept': {
                'mean': b0_mean.reshape(grid_size, grid_size),
                'std': b0_std.reshape(grid_size, grid_size)
            },
            'svc_x1_smooth': {
                'mean': b1_mean.reshape(grid_size, grid_size),
                'std': b1_std.reshape(grid_size, grid_size)
            },
            'svc_x2_smooth': {
                'mean': b2_mean.reshape(grid_size, grid_size),
                'std': b2_std.reshape(grid_size, grid_size)
            }
        }
    except ValueError as e:
        raise ValueError(
            f"Could not reshape to {grid_size}x{grid_size}. "
            f"Points: {len(b0_mean)}. Error: {e}"
        )
    
    return result


def create_visualization(
    results_dir: Path,
    experiments: Dict[str, str],
    model_type: str,
    grid_size: int = 25,
    noise_std: float = 0.1,
    random_seed: int = 222,
    output_file: str = None
):
    """
    Create the exact same visualization as visualize_pub_mlp.py / visualize_pub_xgb.py
    
    Args:
        results_dir: Directory with experiment results
        experiments: Dict mapping encoder_name -> display_name
        model_type: 'MLP' or 'XGBoost'
        grid_size: Grid dimension
        noise_std: Noise std for DGP (should match experiment runs)
        random_seed: Random seed for DGP (should match experiment runs)
        output_file: Output PDF path (auto-generated if None)
    """
    
    # Define coefficients (same as original scripts)
    COEFFICIENTS = {
        'intercept': {'name': 'Intercept (b0)', 'true_col': 'b0'},
        'svc_x1_smooth': {'name': 'Coefficient X1 (b1)', 'true_col': 'b1'},
        'svc_x2_smooth': {'name': 'Coefficient X2 (b2)', 'true_col': 'b2'}
    }
    
    # --- Generate Ground Truth (formulaic DGP, same as gridRun.py) ---
    print("🔎 Generating ground truth from formulaic DGP...")
    try:
        data, extent = create_grid_data(
            size=grid_size,
            coord_system='regional',
            noise_std=noise_std,
            random_seed=random_seed
        )
        
        # Extract ground truth surfaces (sorted by lat, lon for consistency)
        data_sorted = data.sort_values(['lat', 'lon']).reset_index(drop=True)
        
        ground_truth = {
            'intercept': data_sorted['b0'].values.reshape(grid_size, grid_size),
            'svc_x1_smooth': data_sorted['b1'].values.reshape(grid_size, grid_size),
            'svc_x2_smooth': data_sorted['b2'].values.reshape(grid_size, grid_size)
        }
        print(f"  ✅ Generated ground truth (grid_size={grid_size}, noise_std={noise_std}, seed={random_seed})")
    except Exception as e:
        print(f"  ❌ ERROR: Could not generate ground truth. Error: {e}")
        raise
    
    # --- Load and Aggregate Experiment Data ---
    print("\n🔎 Loading and aggregating experiment data for all coefficients...")
    loaded_data = {}
    
    for exp_key, disp_name in experiments.items():
        try:
            exp_data = load_and_aggregate_results(
                results_dir=results_dir,
                encoder_name=exp_key,
                model_type=model_type,
                grid_size=grid_size
            )
            
            # Add display name to each coefficient's data
            for coeff_key in COEFFICIENTS.keys():
                exp_data[coeff_key]['name'] = disp_name
            
            loaded_data[exp_key] = exp_data
            print(f"  ✅ Aggregated data for '{disp_name}'")
            
        except FileNotFoundError as e:
            print(f"  ⚠️  Skipping '{disp_name}': {e}")
            continue
    
    if not loaded_data:
        print("\n❌ No experiment data loaded. Exiting.")
        return
    
    # --- Create Visualization (exact same as original scripts) ---
    print("\n🎨 Generating comprehensive plot...")
    
    fig, axes = plt.subplots(
        nrows=len(COEFFICIENTS) * 2,
        ncols=len(experiments) + 1,
        figsize=(16, 18),
        dpi=300,
        gridspec_kw={'width_ratios': [1.2] + [1] * len(experiments)}
    )
    
    # Populate Grid
    for i, (coeff_key, coeff_info) in enumerate(COEFFICIENTS.items()):
        mean_row_idx = i * 2
        std_row_idx = i * 2 + 1
        
        # Compute std range across all encoders for this coefficient
        all_stds_for_row = [loaded_data[exp_key][coeff_key]['std'] 
                           for exp_key in experiments.keys() 
                           if exp_key in loaded_data]
        std_vmin, std_vmax = np.min(all_stds_for_row), np.max(all_stds_for_row)
        
        # Column 0: Ground Truth
        ax_mean_true = axes[mean_row_idx, 0]
        im_mean = ax_mean_true.imshow(ground_truth[coeff_key], vmin=0, vmax=5, cmap='viridis')
        fig.colorbar(im_mean, ax=ax_mean_true)
        
        ax_std_true = axes[std_row_idx, 0]
        ax_std_true.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=16, color='grey')
        for spine in ax_std_true.spines.values():
            spine.set_visible(False)
        ax_std_true.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        
        # Columns 1+: Experiment results
        for j, (exp_key, exp_name) in enumerate(experiments.items(), 1):
            if exp_key not in loaded_data:
                # Skip if data not available
                for row_idx in [mean_row_idx, std_row_idx]:
                    axes[row_idx, j].text(0.5, 0.5, 'N/A', ha='center', va='center', 
                                         fontsize=14, color='grey')
                    axes[row_idx, j].axis('off')
                continue
            
            data = loaded_data[exp_key][coeff_key]
            
            # Set color range based on model and coefficient
            if coeff_key == 'intercept':
                model_mean_vmin, model_mean_vmax = 0, 5
            else:
                if model_type == 'MLP':
                    model_mean_vmin, model_mean_vmax = 0, 0.05
                else:  # XGBoost
                    model_mean_vmin, model_mean_vmax = 0, 0.45
            
            # Mean panel
            ax_mean = axes[mean_row_idx, j]
            im_mean_exp = ax_mean.imshow(data['mean'], vmin=model_mean_vmin, 
                                        vmax=model_mean_vmax, cmap='viridis')
            fig.colorbar(im_mean_exp, ax=ax_mean)
            
            # Std panel
            ax_std = axes[std_row_idx, j]
            im_std = ax_std.imshow(data['std'], vmin=std_vmin, vmax=std_vmax, cmap='plasma')
            fig.colorbar(im_std, ax=ax_std)
    
    # Set Titles & Labels
    axes[0, 0].set_title("Ground Truth", fontsize=14, fontweight='bold')
    for j, exp_name in enumerate(experiments.values(), 1):
        axes[0, j].set_title(exp_name, fontsize=14, fontweight='bold')
    
    # Clean up all axis ticks
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    
    fig.suptitle(
        f'{model_type} Model: Ground Truth vs. Estimated Surfaces for All Coefficients',
        fontsize=20,
        y=0.98
    )
    
    # --- Precise Label Placement (same as original) ---
    plt.tight_layout(rect=[0.1, 0, 1, 0.96])
    
    for i, (coeff_key, coeff_info) in enumerate(COEFFICIENTS.items()):
        mean_row_idx = i * 2
        std_row_idx = i * 2 + 1
        
        pos_mean = axes[mean_row_idx, 0].get_position()
        pos_std = axes[std_row_idx, 0].get_position()
        
        # Main coefficient label
        fig.text(0.04, (pos_mean.y1 + pos_std.y0) / 2,
                coeff_info['name'], va='center', rotation='vertical', fontsize=16)
        
        # "Mean" and "Std. Dev." labels
        fig.text(0.12, pos_mean.y0 + pos_mean.height / 2, "Mean", 
                ha='right', va='center', rotation='vertical', fontsize=14)
        fig.text(0.12, pos_std.y0 + pos_std.height / 2, "Std. Dev.", 
                ha='right', va='center', rotation='vertical', fontsize=14)
    
    # --- Save as PDF ---
    if output_file is None:
        model_lower = model_type.lower().replace('boost', '')
        output_file = f'comparison_{model_lower}_all_coefficients.pdf'
    
    plt.savefig(output_file, bbox_inches='tight')
    print(f"\n✨ Plot saved as '{output_file}'")


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate per-rep results and create grid visualization PDF'
    )
    parser.add_argument('--results_dir', type=str, default='./results/grid',
                       help='Directory with per-repetition spatial_effects CSVs')
    parser.add_argument('--model_type', type=str, default='MLP',
                       choices=['MLP', 'XGBoost'],
                       help='Model type to visualize')
    parser.add_argument('--grid_size', type=int, default=25,
                       help='Grid dimension (e.g., 25 for 25x25)')
    parser.add_argument('--noise_std', type=float, default=0.1,
                       help='Noise std for DGP (must match experiment runs)')
    parser.add_argument('--random_seed', type=int, default=222,
                       help='Random seed for DGP (must match experiment runs)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output PDF filename (auto-generated if not specified)')
    
    # Experiment configuration
    parser.add_argument('--encoders', nargs='+', 
                       default=['tile_ffn', 'wrap_ffn', 'sphere2vec_dim32'],
                       help='Encoder names to compare')
    parser.add_argument('--encoder_labels', nargs='+', default=None,
                       help='Display labels for encoders (same order as --encoders)')
    
    args = parser.parse_args()
    
    # Build experiments dict
    if args.encoder_labels and len(args.encoder_labels) == len(args.encoders):
        experiments = dict(zip(args.encoders, args.encoder_labels))
    else:
        # Use encoder names as display names
        experiments = {enc: enc.replace('_', ' ').title() for enc in args.encoders}
    
    create_visualization(
        results_dir=Path(args.results_dir),
        experiments=experiments,
        model_type=args.model_type,
        grid_size=args.grid_size,
        noise_std=args.noise_std,
        random_seed=args.random_seed,
        output_file=args.output
    )


if __name__ == '__main__':
    main()
