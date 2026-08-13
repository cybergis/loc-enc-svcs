import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- Configuration ---

# 1. Set the base directory for experiment results.
BASE_RESULTS_DIR = Path('./results/finalDraft')

# 2. Path to the CSV file containing the ground truth data.
DATA_FILE = Path('./data/mgwr_sim.csv')

# 3. Model display name (for titles).
MODEL_DISPLAY_NAME = 'MLP'

# 4. Encoders to compare.
EXPERIMENTS = {
    'tile_ffn': 'Tile FFN',
    'wrap_ffn': 'Wrap FFN',
    'Sphere2Vec-sphereC': 'Sphere2Vec (2D)'
}

# 5. Define coefficients to plot.
COEFFICIENTS = {
    'intercept': {'name': 'Intercept (b0)', 'true_col': 'b0'},
    'svc_x1_smooth': {'name': 'Coefficient X1 (b1)', 'true_col': 'b1'},
    'svc_x2_smooth': {'name': 'Coefficient X2 (b2)', 'true_col': 'b2'}
}

# --- Data Loading ---

print("🔎 Loading ground truth data...")
try:
    mgwr_sim_df = pd.read_csv(DATA_FILE)
    size = int(np.sqrt(len(mgwr_sim_df)))
    ground_truth = {}
    for key, info in COEFFICIENTS.items():
        true_surface = mgwr_sim_df[info['true_col']].values.reshape(size, size)
        ground_truth[key] = true_surface
    print("  ✅ Successfully loaded all ground truth surfaces.")
except Exception as e:
    print(f"  ❌ ERROR: Could not load or process ground truth data from {DATA_FILE}. Error: {e}")
    exit()

print("\n🔎 Loading experiment data for all coefficients...")
loaded_data = {exp_key: {} for exp_key in EXPERIMENTS}
for exp_key, disp_name in EXPERIMENTS.items():
    for coeff_key, coeff_info in COEFFICIENTS.items():
        file_prefix = f"mlp_{exp_key}_{coeff_key}"
        encoder_dir = BASE_RESULTS_DIR / exp_key
        mean_file = encoder_dir / f"{file_prefix}_mean_surface.npy"
        std_file = encoder_dir / f"{file_prefix}_std_surface.npy"

        try:
            mean_surface = np.load(mean_file).reshape(size, size)
            std_surface = np.load(std_file).reshape(size, size)

            loaded_data[exp_key][coeff_key] = {
                'mean': mean_surface,
                'std': std_surface,
                'name': disp_name
            }
            print(f"  ✅ Loaded {coeff_info['name']} for '{disp_name}'")
        except FileNotFoundError:
            print(f"\n  ❌ ERROR: Could not find data for '{disp_name}' - {coeff_info['name']}.")
            print(f"       - Searched for mean: {mean_file}")
            print(f"       - Searched for std:  {std_file}")
            exit()

# --- Visualization ---

fig, axes = plt.subplots(
    nrows=len(COEFFICIENTS) * 2,
    ncols=len(EXPERIMENTS) + 1,
    figsize=(16, 18),
    dpi=300,
    gridspec_kw={'width_ratios': [1.2, 1, 1, 1]}
)
print("\n🎨 Generating comprehensive plot...")

# Populate Grid
for i, (coeff_key, coeff_info) in enumerate(COEFFICIENTS.items()):
    mean_row_idx = i * 2
    std_row_idx = i * 2 + 1

    all_stds_for_row = [loaded_data[exp_key][coeff_key]['std'] for exp_key in EXPERIMENTS]
    std_vmin, std_vmax = np.min(all_stds_for_row), np.max(all_stds_for_row)

    ax_mean_true = axes[mean_row_idx, 0]
    im_mean = ax_mean_true.imshow(ground_truth[coeff_key], vmin=0, vmax=5, cmap='viridis')
    fig.colorbar(im_mean, ax=ax_mean_true)

    ax_std_true = axes[std_row_idx, 0]
    ax_std_true.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=16, color='grey')
    for spine in ax_std_true.spines.values():
        spine.set_visible(False)
    ax_std_true.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


    for j, (exp_key, exp_name) in enumerate(EXPERIMENTS.items(), 1):
        data = loaded_data[exp_key][coeff_key]
        
        if coeff_key == 'intercept':
            model_mean_vmin, model_mean_vmax = 0, 5
        else:
            model_mean_vmin, model_mean_vmax = 0, 0.05

        ax_mean = axes[mean_row_idx, j]
        im_mean_exp = ax_mean.imshow(data['mean'], vmin=model_mean_vmin, vmax=model_mean_vmax, cmap='viridis')
        fig.colorbar(im_mean_exp, ax=ax_mean)

        ax_std = axes[std_row_idx, j]
        im_std = ax_std.imshow(data['std'], vmin=std_vmin, vmax=std_vmax, cmap='plasma')
        fig.colorbar(im_std, ax=ax_std)

# Set Titles & Labels
axes[0, 0].set_title("Ground Truth", fontsize=14, fontweight='bold')
for j, exp_name in enumerate(EXPERIMENTS.values(), 1):
    axes[0, j].set_title(exp_name, fontsize=14, fontweight='bold')

# Clean up all axis ticks first
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle(
    f'{MODEL_DISPLAY_NAME} Model: Ground Truth vs. Estimated Surfaces for All Coefficients',
    fontsize=20,
    y=0.98
)

# --- UPDATED: Precise Label Placement ---
# Finalize the layout first, then get coordinates and add text.
plt.tight_layout(rect=[0.1, 0, 1, 0.96])

for i, (coeff_key, coeff_info) in enumerate(COEFFICIENTS.items()):
    mean_row_idx = i * 2
    std_row_idx = i * 2 + 1
    
    # Get the exact positions of the axes in the first column AFTER the layout is finalized
    pos_mean = axes[mean_row_idx, 0].get_position()
    pos_std = axes[std_row_idx, 0].get_position()
    
    # Place the main coefficient label in the vertical center of the two-row block
    fig.text(0.04, (pos_mean.y1 + pos_std.y0) / 2,
             coeff_info['name'], va='center', rotation='vertical', fontsize=16)

    # Place the "Mean" and "Std. Dev." labels at the precise center of their respective axes
    # using the same x-coordinate to ensure horizontal alignment.
    fig.text(0.12, pos_mean.y0 + pos_mean.height / 2, "Mean", ha='right', va='center', rotation='vertical', fontsize=14)
    fig.text(0.12, pos_std.y0 + pos_std.height / 2, "Std. Dev.", ha='right', va='center', rotation='vertical', fontsize=14)

output_filename = 'comparison_mlp_all_coefficients.pdf'
plt.savefig(output_filename, bbox_inches='tight')
print(f"\n✨ Plot saved as '{output_filename}'")