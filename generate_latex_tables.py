#!/usr/bin/env python3
import pandas as pd
import numpy as np

# Read the data
df = pd.read_csv('results/combined_summary_stats.csv')

# Filter for trained/untrained smooth SVCs only
mask = (df['feature_config'].isin(['baseline', 'trained'])) & (df['spatial_effect'].str.contains('Smooth'))
filtered = df[mask]

print("Data shape:", filtered.shape)
print("\nScales:", sorted(filtered['scale'].unique()))
print("Encoders:", sorted(filtered['encoder'].unique()))
print("Feature configs:", sorted(filtered['feature_config'].unique()))

# For each scale, create a table comparing untrained vs trained
for scale in ['grid', 'county', 'global']:
    print(f"\n{'='*80}")
    print(f"SCALE: {scale.upper()}")
    print('='*80)

    scale_data = filtered[filtered['scale'] == scale]

    # Get SVCs (X1 and X2)
    for svc_type in ['SVC_X1_Smooth', 'SVC_X2_Smooth']:
        svc_data = scale_data[scale_data['spatial_effect'] == svc_type]

        if len(svc_data) == 0:
            continue

        print(f"\n{svc_type}:")

        # Group by encoder and feature_config
        encoders = sorted(svc_data['encoder'].unique())

        for encoder in encoders:
            enc_data = svc_data[svc_data['encoder'] == encoder]

            untrained = enc_data[enc_data['feature_config'] == 'baseline']
            trained = enc_data[enc_data['feature_config'] == 'trained']

            if len(untrained) > 0:
                r_untrained = untrained['pearson_r_mean'].values[0]
                print(f"  {encoder:25s} untrained: r={r_untrained:.4f}")

            if len(trained) > 0:
                r_trained = trained['pearson_r_mean'].values[0]
                print(f"  {encoder:25s} trained:   r={r_trained:.4f}")

print("\n\nNow generating LaTeX tables...")

# Create LaTeX tables for paper
# Organize: for each scale, separate β₁ and β₂ tables
for scale in ['grid', 'county', 'global']:
    scale_data = filtered[filtered['scale'] == scale]

    for svc_idx, svc_type in enumerate(['SVC_X1_Smooth', 'SVC_X2_Smooth'], 1):
        svc_data = scale_data[scale_data['spatial_effect'] == svc_type]

        if len(svc_data) == 0:
            continue

        # Build table data
        encoders = sorted(svc_data['encoder'].unique())

        latex = f"\\begin{{table}}[h!]\n"
        latex += f"\\centering\n"
        latex += f"\\caption{{Correlation ($r$) for β{svc_idx} — {scale.capitalize()} Scale}}\n"
        latex += f"\\label{{tab:results_beta{svc_idx}_{scale}}}\n"
        latex += f"\\small\n"
        latex += f"\\begin{{tabular}}{{lcc|c}}\n"
        latex += f"\\toprule\n"
        latex += f"Encoder & Untrained & Trained & Δ \\\\\n"
        latex += f"\\midrule\n"

        for encoder in encoders:
            enc_data = svc_data[svc_data['encoder'] == encoder]

            untrained_row = enc_data[enc_data['feature_config'] == 'baseline']
            trained_row = enc_data[enc_data['feature_config'] == 'trained']

            r_untrained = untrained_row['pearson_r_mean'].values[0] if len(untrained_row) > 0 else np.nan
            r_trained = trained_row['pearson_r_mean'].values[0] if len(trained_row) > 0 else np.nan

            delta = r_trained - r_untrained if not np.isnan(r_untrained) and not np.isnan(r_trained) else np.nan

            # Format with appropriate precision
            r_u_str = f"{r_untrained:.4f}" if not np.isnan(r_untrained) else "—"
            r_t_str = f"{r_trained:.4f}" if not np.isnan(r_trained) else "—"
            delta_str = f"{delta:+.4f}" if not np.isnan(delta) else "—"

            latex += f"{encoder} & {r_u_str} & {r_t_str} & {delta_str} \\\\\n"

        latex += f"\\bottomrule\n"
        latex += f"\\end{{tabular}}\n"
        latex += f"\\end{{table}}\n"

        print(f"\n{scale.upper()} - β{svc_idx}:")
        print(latex)

