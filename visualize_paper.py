"""
Generate publication figures for the spatial embeddings paper.

Produces:
  plots/fig_ground_truth.pdf      — True SVC surfaces across 3 scales
  plots/fig_main_heatmap.pdf      — Encoder performance heatmap (OLS slope + Pearson r)
  plots/fig_spatial_global.pdf    — Global β₂ recovery: true vs estimated maps
  plots/fig_training_effect.pdf   — Contrastive training impact by encoder tier
  plots/tab_main_results.tex      — LaTeX table of key metrics

Usage:
    python visualize_paper.py --results_root ./results --output_dir ./plots
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, TwoSlopeNorm
import seaborn as sns

# ── Style ────────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', context='paper', font_scale=1.0,
              rc={'figure.dpi': 300, 'savefig.dpi': 300,
                  'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05})

# Encoder display order: Tier 1 → Tier 2 → Tier 3
ENCODER_ORDER = [
    'Sphere2Vec-dfs', 'Sphere2Vec-sphereM+', 'wrap_ffn',
    'Sphere2Vec-sphereM', 'Sphere2Vec-sphereC', 'Sphere2Vec-sphereC+',
    'Space2Vec-grid', 'Space2Vec-theory', 'NeRF',
    'rff', 'tile_ffn',
]

TIER_LABELS = {
    'Sphere2Vec-dfs': 1, 'Sphere2Vec-sphereM+': 1, 'wrap_ffn': 1,
    'Sphere2Vec-sphereM': 2, 'Sphere2Vec-sphereC': 2, 'Sphere2Vec-sphereC+': 2,
    'Space2Vec-grid': 2, 'Space2Vec-theory': 2, 'NeRF': 2,
    'rff': 3, 'tile_ffn': 3,
}

TIER_COLORS = {1: '#2ecc71', 2: '#f39c12', 3: '#e74c3c'}

SCALE_LABELS = {'global': 'Global', 'county': 'County', 'grid': 'Grid'}


def load_data(results_root):
    """Load summary stats and statistical tests."""
    stats = pd.read_csv(os.path.join(results_root, 'combined_summary_stats.csv'))
    tests = pd.read_csv(os.path.join(results_root, 'statistical_tests.csv'))
    return stats, tests


# ═════════════════════════════════════════════════════════════════════════════
# Figure 1: Ground Truth Surfaces
# ═════════════════════════════════════════════════════════════════════════════

def fig_ground_truth(results_root, output_dir):
    """Show true β₀, β₁, β₂ surfaces across all 3 scales."""
    fig = plt.figure(figsize=(7.2, 7.0))
    gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.30,
                           left=0.08, right=0.92, top=0.90, bottom=0.05)

    scales = ['global', 'county', 'grid']
    coeffs = [('b0_true', r'$\beta_0$ (Intercept)'),
              ('b1_true', r'$\beta_1$ (Smooth gradient)'),
              ('b2_true', r'$\beta_2$ (Multi-scale oscillation)')]

    for col, scale in enumerate(scales):
        # Read one rep from baseline
        csv_path = os.path.join(results_root,
                                f'{scale}_simple_baseline',
                                'none_MLP_rep0_spatial_effects.csv')
        if not os.path.exists(csv_path):
            import glob as gl
            candidates = gl.glob(os.path.join(results_root,
                                              f'{scale}_simple_baseline',
                                              '*_rep0_spatial_effects.csv'))
            csv_path = candidates[0] if candidates else None

        if csv_path is None:
            continue

        df = pd.read_csv(csv_path)

        for row, (coeff, label) in enumerate(coeffs):
            ax = fig.add_subplot(gs[row, col])
            sc = ax.scatter(df['lon'], df['lat'], c=df[coeff],
                            s=0.3 if scale == 'global' else (0.8 if scale == 'county' else 8),
                            cmap=sns.color_palette('RdYlBu_r', as_cmap=True),
                            rasterized=True)
            plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)

            if row == 0:
                ax.set_title(SCALE_LABELS[scale], fontweight='bold', fontsize=11)
            if col == 0:
                ax.set_ylabel(label, fontsize=9)
            ax.set_xlabel('')
            ax.tick_params(labelsize=6)

            if scale == 'global':
                ax.set_xlim(-180, 180)
                ax.set_ylim(-90, 90)
                ax.set_aspect('equal')

    fig.suptitle('True Spatially-Varying Coefficients', fontsize=12, fontweight='bold', y=0.96)
    out = os.path.join(output_dir, 'fig_ground_truth.pdf')
    fig.savefig(out)
    plt.close(fig)
    print(f'Saved: {out}')


# ═════════════════════════════════════════════════════════════════════════════
# Figure 2: Main Results Heatmap
# ═════════════════════════════════════════════════════════════════════════════

def fig_main_heatmap(stats, tests, output_dir):
    """Heatmap of β₂ OLS slope and Pearson r across encoders, scales, conditions."""
    metrics = [('ols_slope_mean', r'Amplitude Recovery (OLS Slope)'),
               ('pearson_r_mean', r'Pattern Recovery (Pearson $r$)')]

    conditions = [
        ('baseline', False, 'BL'),
        ('emb_only', False, 'EO'),
        ('emb_only', True, 'EO-T'),
    ]
    scales = ['global', 'county', 'grid']
    n_cond = len(conditions)
    n_enc = len(ENCODER_ORDER)
    n_cols = len(scales) * n_cond

    # Build column labels for sns.heatmap
    col_labels = []
    for scale in scales:
        for _, _, short in conditions:
            col_labels.append(f'{SCALE_LABELS[scale]}\n{short}')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5),
                             gridspec_kw={'wspace': 0.30,
                                          'left': 0.14, 'right': 0.95,
                                          'top': 0.88, 'bottom': 0.12})

    for ax_idx, (metric, title) in enumerate(metrics):
        ax = axes[ax_idx]

        # Build matrix
        matrix = np.full((n_enc, n_cols), np.nan)
        for i, enc in enumerate(ENCODER_ORDER):
            for j, scale in enumerate(scales):
                for k, (fc, trained, _) in enumerate(conditions):
                    col = j * n_cond + k
                    dim = 0 if fc == 'baseline' else 8
                    mask = (
                        (stats['encoder'] == enc) &
                        (stats['scale'] == scale) &
                        (stats['feature_config'] == fc) &
                        (stats['embed_dim'] == dim) &
                        (stats['encoder_trained'] == trained) &
                        (stats['spatial_effect'] == 'SVC_X2_Smooth')
                    )
                    vals = stats.loc[mask, metric]
                    if not vals.empty:
                        matrix[i, col] = vals.values[0]

        # Build annotation matrix with significance stars
        annot = np.full((n_enc, n_cols), '', dtype=object)
        for i in range(n_enc):
            for j in range(n_cols):
                val = matrix[i, j]
                if np.isnan(val):
                    annot[i, j] = '—'
                else:
                    annot[i, j] = f'{val:.2f}'

        # Add significance markers to annotations
        metric_name = 'ols_slope' if 'slope' in metric else 'pearson_r'
        for i, enc in enumerate(ENCODER_ORDER):
            for j, scale in enumerate(scales):
                for k, (fc, trained, _) in enumerate(conditions):
                    if fc == 'baseline':
                        continue
                    col = j * n_cond + k
                    train_label = 'trained' if trained else 'untrained'
                    comp = f'emb_only_vs_baseline_{train_label}'
                    tmask = (
                        (tests['encoder'] == enc) &
                        (tests['scale'] == scale) &
                        (tests['comparison'] == comp) &
                        (tests['spatial_effect'] == 'SVC_X2_Smooth') &
                        (tests['metric'] == metric_name)
                    )
                    rows = tests.loc[tmask]
                    if not rows.empty and rows.iloc[0]['significant']:
                        annot[i, col] = annot[i, col] + '*'

        # Colormap range
        vmin = 0.0 if 'slope' in metric else 0.3
        vmax = 1.0

        df_matrix = pd.DataFrame(matrix, index=ENCODER_ORDER, columns=col_labels)
        sns.heatmap(df_matrix, ax=ax, cmap='RdYlGn', vmin=vmin, vmax=vmax,
                    annot=annot, fmt='', annot_kws={'size': 7},
                    linewidths=0.5, linecolor='white',
                    cbar_kws={'shrink': 0.7, 'pad': 0.03})

        # Scale group separators (thicker white lines)
        for sep in [n_cond, 2 * n_cond]:
            ax.axvline(sep, color='white', linewidth=3)

        # Tier separators
        ax.axhline(3, color='white', linewidth=3)
        ax.axhline(9, color='white', linewidth=3)

        # Tier color dots on y-axis
        if ax_idx == 0:
            for i, enc in enumerate(ENCODER_ORDER):
                tier = TIER_LABELS[enc]
                ax.plot(-0.03, (i + 0.5) / n_enc, 'o', color=TIER_COLORS[tier],
                        markersize=6, clip_on=False, transform=ax.transAxes)
        else:
            ax.set_yticklabels([])

        ax.set_title(title, fontsize=10, fontweight='bold', pad=10)
        ax.set_xlabel('')
        ax.set_ylabel('')

    fig.suptitle(r'$\beta_2$ SVC Recovery (Multi-Scale Oscillation)',
                 fontsize=12, fontweight='bold', y=0.96)

    # Legend at bottom
    fig.text(0.5, 0.02,
             'BL = Baseline (coords only)      '
             'EO = Embedding Only (untrained)      '
             'EO-T = Embedding Only (trained)      '
             '* significantly > baseline (p<0.05)',
             ha='center', fontsize=7.5, style='italic', color='#555555')

    out = os.path.join(output_dir, 'fig_main_heatmap.pdf')
    fig.savefig(out)
    plt.close(fig)
    print(f'Saved: {out}')


# ═════════════════════════════════════════════════════════════════════════════
# Figure 3: Global β₂ Spatial Maps
# ═════════════════════════════════════════════════════════════════════════════

def fig_spatial_global(results_root, output_dir):
    """Global β₂: true vs estimated for representative encoders."""
    # Representative encoders: Tier1, Tier2 (trained), Tier3, Baseline
    panels = [
        ('Truth', None, None, None),
        ('Baseline\n(coords only)', 'global_simple_baseline', 'none', None),
        ('Sphere2Vec-dfs\n(Tier 1, emb only)', 'global_simple_embonly_dim8', 'Sphere2Vec-dfs', None),
        ('Space2Vec-grid\n(Tier 2, trained emb only)', 'global_simple_trained_embonly_dim8', 'Space2Vec-grid', None),
        ('tile_ffn\n(Tier 3, emb only)', 'global_simple_embonly_dim8', 'tile_ffn', None),
    ]

    fig, axes = plt.subplots(2, 5, figsize=(11, 4.2),
                             gridspec_kw={'hspace': 0.05, 'wspace': 0.08,
                                          'left': 0.04, 'right': 0.96,
                                          'top': 0.88, 'bottom': 0.05})

    truth_df = None
    vmin_coeff, vmax_coeff = None, None

    # Load data and find color range
    data_list = []
    for label, subdir, enc, _ in panels:
        if subdir is None:
            data_list.append(None)
            continue
        csv = os.path.join(results_root, subdir, f'{enc}_MLP_rep0_spatial_effects.csv')
        df = pd.read_csv(csv)
        if truth_df is None:
            truth_df = df
            vmin_coeff = df['b2_true'].quantile(0.02)
            vmax_coeff = df['b2_true'].quantile(0.98)
        data_list.append(df)

    norm_coeff = Normalize(vmin=vmin_coeff, vmax=vmax_coeff)

    # Row 1: Coefficient surfaces
    for col, (label, subdir, enc, _) in enumerate(panels):
        ax = axes[0, col]
        if col == 0:
            vals = truth_df['b2_true']
        else:
            vals = data_list[col]['b2_smooth_estimated']

        sc = ax.scatter(truth_df['lon'], truth_df['lat'], c=vals,
                        s=0.15, cmap='RdYlBu_r', norm=norm_coeff, rasterized=True)
        ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
        ax.set_aspect('equal')
        ax.set_title(label, fontsize=7.5, pad=3)
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(r'$\hat{\beta}_2$ surface', fontsize=8)

    # Row 2: Error maps
    max_err = 0
    errors = []
    for col, (label, subdir, enc, _) in enumerate(panels):
        if col == 0:
            errors.append(None)
            continue
        err = data_list[col]['b2_smooth_estimated'] - truth_df['b2_true']
        errors.append(err)
        max_err = max(max_err, np.abs(err).quantile(0.98))

    norm_err = TwoSlopeNorm(vmin=-max_err, vcenter=0, vmax=max_err)

    for col in range(5):
        ax = axes[1, col]
        if col == 0:
            ax.axis('off')
            continue
        sc = ax.scatter(truth_df['lon'], truth_df['lat'], c=errors[col],
                        s=0.15, cmap='RdBu_r', norm=norm_err, rasterized=True)
        ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        if col == 1:
            ax.set_ylabel('Estimation error', fontsize=8)

    # Colorbars
    cbar_ax1 = fig.add_axes([0.97, 0.52, 0.01, 0.35])
    plt.colorbar(plt.cm.ScalarMappable(norm=norm_coeff, cmap='RdYlBu_r'),
                 cax=cbar_ax1, label=r'$\beta_2$')
    cbar_ax2 = fig.add_axes([0.97, 0.07, 0.01, 0.35])
    plt.colorbar(plt.cm.ScalarMappable(norm=norm_err, cmap='RdBu_r'),
                 cax=cbar_ax2, label='Error')

    fig.suptitle(r'Global $\beta_2$ Recovery: Representative Encoders (rep 0)',
                 fontsize=11, fontweight='bold', y=0.97)

    out = os.path.join(output_dir, 'fig_spatial_global.pdf')
    fig.savefig(out)
    plt.close(fig)
    print(f'Saved: {out}')


# ═════════════════════════════════════════════════════════════════════════════
# Figure 4: Contrastive Training Effect
# ═════════════════════════════════════════════════════════════════════════════

def fig_training_effect(stats, output_dir):
    """Paired dot plot: untrained vs trained emb_only for each encoder on global β₂."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5),
                             gridspec_kw={'wspace': 0.35})

    metrics = [('ols_slope_mean', 'OLS Slope (Amplitude)'),
               ('pearson_r_mean', r'Pearson $r$ (Pattern)')]

    tier_palette = {1: '#2ecc71', 2: '#f39c12', 3: '#e74c3c'}

    for ax_idx, (metric, ylabel) in enumerate(metrics):
        ax = axes[ax_idx]
        sns.despine(ax=ax, left=True)

        sub = stats[
            (stats['scale'] == 'global') &
            (stats['spatial_effect'] == 'SVC_X2_Smooth') &
            (stats['feature_config'] == 'emb_only') &
            (stats['embed_dim'] == 8) &
            (stats['encoder'].isin(ENCODER_ORDER))
        ].copy()

        # Get baseline reference
        bl = stats[
            (stats['scale'] == 'global') &
            (stats['spatial_effect'] == 'SVC_X2_Smooth') &
            (stats['feature_config'] == 'baseline') &
            (stats['embed_dim'] == 0) &
            (stats['encoder'] == 'none')
        ]
        bl_val = bl[metric].values[0] if not bl.empty else None

        y_positions = np.arange(len(ENCODER_ORDER))

        for i, enc in enumerate(ENCODER_ORDER):
            untrained = sub[(sub['encoder'] == enc) & (sub['encoder_trained'] == False)]
            trained = sub[(sub['encoder'] == enc) & (sub['encoder_trained'] == True)]

            ut_val = untrained[metric].values[0] if not untrained.empty else np.nan
            tr_val = trained[metric].values[0] if not trained.empty else np.nan

            tier = TIER_LABELS[enc]
            color = tier_palette[tier]

            # Draw connecting line
            if not (np.isnan(ut_val) or np.isnan(tr_val)):
                line_color = '#2ecc71' if tr_val > ut_val else '#e74c3c'
                ax.plot([ut_val, tr_val], [i, i], color=line_color, alpha=0.4,
                        linewidth=2, zorder=1)

            # Untrained dot
            if not np.isnan(ut_val):
                ax.scatter(ut_val, i, color=color, s=40,
                           edgecolors='black', linewidths=0.5, zorder=3,
                           marker='o')
            # Trained dot
            if not np.isnan(tr_val):
                ax.scatter(tr_val, i, color=color, s=40,
                           edgecolors='black', linewidths=0.5, zorder=3,
                           marker='D')

        # Baseline reference line
        if bl_val is not None:
            ax.axvline(bl_val, color='gray', linestyle='--', linewidth=0.8,
                       alpha=0.7, zorder=0)
            ax.text(bl_val, len(ENCODER_ORDER) - 0.3, 'Baseline',
                    fontsize=6.5, color='gray', ha='center', va='bottom')

        ax.set_yticks(y_positions)
        ax.set_yticklabels(ENCODER_ORDER if ax_idx == 0 else [], fontsize=7.5)
        ax.set_xlabel(ylabel, fontsize=9)
        ax.invert_yaxis()
        ax.set_title(f'Global $\\beta_2$: {ylabel}', fontsize=9, fontweight='bold')

        # Tier separators
        ax.axhline(2.5, color='lightgray', linewidth=0.8, linestyle='-')
        ax.axhline(8.5, color='lightgray', linewidth=0.8, linestyle='-')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markeredgecolor='black', markersize=6, label='Untrained'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='gray',
               markeredgecolor='black', markersize=6, label='Trained'),
        Line2D([0], [0], color='#2ecc71', linewidth=2, alpha=0.5,
               label='Training helps'),
        Line2D([0], [0], color='#e74c3c', linewidth=2, alpha=0.5,
               label='Training hurts'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.05))

    out = os.path.join(output_dir, 'fig_training_effect.pdf')
    fig.savefig(out)
    plt.close(fig)
    print(f'Saved: {out}')


# ═════════════════════════════════════════════════════════════════════════════
# LaTeX Table: Main Results
# ═════════════════════════════════════════════════════════════════════════════

def tab_main_results(stats, tests, output_dir):
    """Generate LaTeX table of key metrics for β₂ smooth."""
    lines = []
    lines.append(r'\begin{table}[ht]')
    lines.append(r'\centering')
    lines.append(r'\caption{SVC recovery for $\beta_2$ (multi-scale oscillation). '
                 r'Pearson $r$ and OLS slope (mean $\pm$ std across 25 repetitions). '
                 r'* = emb\_only significantly better than baseline ($p<0.05$, Wilcoxon).}')
    lines.append(r'\label{tab:main_results}')
    lines.append(r'\resizebox{\textwidth}{!}{%')
    lines.append(r'\begin{tabular}{ll' + 'cc' * 3 + '}')
    lines.append(r'\toprule')
    lines.append(r' & & \multicolumn{2}{c}{\textbf{Global}} & '
                 r'\multicolumn{2}{c}{\textbf{County}} & '
                 r'\multicolumn{2}{c}{\textbf{Grid}} \\')
    lines.append(r'\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}')
    lines.append(r'\textbf{Tier} & \textbf{Encoder} & '
                 r'Pearson $r$ & OLS slope & '
                 r'Pearson $r$ & OLS slope & '
                 r'Pearson $r$ & OLS slope \\')
    lines.append(r'\midrule')

    # Baseline row first
    lines.append(r'\multicolumn{2}{l}{\textit{Baseline (coords only)}}')
    for scale in ['global', 'county', 'grid']:
        mask = (
            (stats['scale'] == scale) &
            (stats['feature_config'] == 'baseline') &
            (stats['encoder'] == 'none') &
            (stats['spatial_effect'] == 'SVC_X2_Smooth')
        )
        row = stats.loc[mask]
        if not row.empty:
            r_val = row['pearson_r_mean'].values[0]
            r_std = row['pearson_r_std'].values[0]
            s_val = row['ols_slope_mean'].values[0]
            s_std = row['ols_slope_std'].values[0]
            lines[-1] += f' & {r_val:.3f}$\\pm${r_std:.3f} & {s_val:.3f}$\\pm${s_std:.3f}'
        else:
            lines[-1] += r' & --- & ---'
    lines[-1] += r' \\'
    lines.append(r'\midrule')
    lines.append(r'\multicolumn{8}{l}{\textit{Embedding only (untrained)}} \\')

    prev_tier = None
    for enc in ENCODER_ORDER:
        tier = TIER_LABELS[enc]
        if prev_tier is not None and tier != prev_tier:
            lines.append(r'\addlinespace[2pt]')
        prev_tier = tier

        line = f'{tier} & {enc}'
        for scale in ['global', 'county', 'grid']:
            mask = (
                (stats['scale'] == scale) &
                (stats['feature_config'] == 'emb_only') &
                (stats['embed_dim'] == 8) &
                (stats['encoder_trained'] == False) &
                (stats['encoder'] == enc) &
                (stats['spatial_effect'] == 'SVC_X2_Smooth')
            )
            row = stats.loc[mask]
            if not row.empty:
                r_val = row['pearson_r_mean'].values[0]
                r_std = row['pearson_r_std'].values[0]
                s_val = row['ols_slope_mean'].values[0]
                s_std = row['ols_slope_std'].values[0]

                # Check significance
                sig_r = _is_significant(tests, enc, scale, 'pearson_r', 'untrained')
                sig_s = _is_significant(tests, enc, scale, 'ols_slope', 'untrained')

                star_r = '*' if sig_r else ''
                star_s = '*' if sig_s else ''

                line += f' & {r_val:.3f}{star_r} & {s_val:.3f}{star_s}'
            else:
                line += r' & --- & ---'
        line += r' \\'
        lines.append(line)

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'}')
    lines.append(r'\end{table}')

    out = os.path.join(output_dir, 'tab_main_results.tex')
    with open(out, 'w') as f:
        f.write('\n'.join(lines))
    print(f'Saved: {out}')


def _is_significant(tests, encoder, scale, metric, train_label):
    """Check if emb_only vs baseline is significant."""
    comp = f'emb_only_vs_baseline_{train_label}'
    mask = (
        (tests['encoder'] == encoder) &
        (tests['scale'] == scale) &
        (tests['comparison'] == comp) &
        (tests['spatial_effect'] == 'SVC_X2_Smooth') &
        (tests['metric'] == metric)
    )
    rows = tests.loc[mask]
    if not rows.empty:
        return rows.iloc[0]['significant']
    return False


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate paper figures')
    parser.add_argument('--results_root', type=str, default='./results')
    parser.add_argument('--output_dir', type=str, default='./plots')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    stats, tests = load_data(args.results_root)

    print("Generating figures...")
    fig_ground_truth(args.results_root, args.output_dir)
    fig_main_heatmap(stats, tests, args.output_dir)
    fig_spatial_global(args.results_root, args.output_dir)
    fig_training_effect(stats, args.output_dir)
    tab_main_results(stats, tests, args.output_dir)
    print("Done!")
