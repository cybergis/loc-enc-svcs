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
import glob as gl
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
import seaborn as sns

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', context='paper', font_scale=1.0,
              rc={'figure.dpi': 300, 'savefig.dpi': 300,
                  'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05})

# Seaborn pastel palette colors (pastel green / orange / red)
TIER_COLORS = {1: '#98df8a', 2: '#ffbb78', 3: '#ff9896'}
TIER_DARK   = {1: '#2ca02c', 2: '#ff7f0e', 3: '#d62728'}

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

SCALE_LABELS = {'global': 'Global', 'county': 'County', 'grid': 'Grid'}


def load_data(results_root):
    stats = pd.read_csv(os.path.join(results_root, 'combined_summary_stats.csv'))
    tests = pd.read_csv(os.path.join(results_root, 'statistical_tests.csv'))
    return stats, tests


def _aspect_for_lat(lat_series):
    """Return Mercator-corrected data aspect ratio for a given latitude band."""
    mean_lat = float(np.mean(lat_series))
    return 1.0 / np.cos(np.radians(mean_lat))


def _scatter_map(ax, lon, lat, vals, cmap, norm, s, **kw):
    """Scatter plot with latitude-corrected aspect ratio."""
    sc = ax.scatter(lon, lat, c=vals, cmap=cmap, norm=norm,
                    s=s, linewidths=0, rasterized=True, **kw)
    ax.set_aspect(_aspect_for_lat(lat))
    ax.set_xticks([]); ax.set_yticks([])
    return sc


# ═════════════════════════════════════════════════════════════════════════════
# Figure 1: Ground Truth Surfaces
# ═════════════════════════════════════════════════════════════════════════════

def fig_ground_truth(results_root, output_dir):
    """Show true β₀, β₁, β₂ surfaces across all 3 scales."""
    scales = ['global', 'county', 'grid']
    coeffs = [
        ('b0_true', r'$\beta_0$ (Intercept)'),
        ('b1_true', r'$\beta_1$ (Smooth gradient)'),
        ('b2_true', r'$\beta_2$ (Multi-scale oscillation)'),
    ]
    # Dot sizes per scale (global: many tiny pts; grid: few large pts)
    dot_s = {'global': 0.6, 'county': 2.5, 'grid': 12}

    # Load data
    dfs = {}
    for scale in scales:
        csv_path = os.path.join(results_root, f'{scale}_simple_baseline',
                                'none_MLP_rep0_spatial_effects.csv')
        if not os.path.exists(csv_path):
            cands = gl.glob(os.path.join(results_root, f'{scale}_simple_baseline',
                                         '*_rep0_spatial_effects.csv'))
            csv_path = cands[0] if cands else None
        if csv_path:
            dfs[scale] = pd.read_csv(csv_path)

    # Shared vmin/vmax per coefficient row (same color scale across scales)
    vlims = {}
    for coeff, _ in coeffs:
        all_vals = np.concatenate([dfs[s][coeff].values for s in scales if s in dfs])
        vlims[coeff] = (np.percentile(all_vals, 2), np.percentile(all_vals, 98))

    # Layout: 3 rows × (3 panels + narrow colorbar col)
    fig = plt.figure(figsize=(9.0, 7.5))
    gs = gridspec.GridSpec(3, 4, width_ratios=[1, 1, 1, 0.05],
                           hspace=0.45, wspace=0.25,
                           left=0.07, right=0.95, top=0.90, bottom=0.06)

    for row, (coeff, label) in enumerate(coeffs):
        vmin, vmax = vlims[coeff]
        norm = Normalize(vmin=vmin, vmax=vmax)
        row_sc = None

        for col, scale in enumerate(scales):
            ax = fig.add_subplot(gs[row, col])
            if scale not in dfs:
                ax.axis('off'); continue

            df = dfs[scale]
            row_sc = _scatter_map(ax, df['lon'], df['lat'], df[coeff],
                                  cmap='coolwarm', norm=norm, s=dot_s[scale])

            if row == 0:
                ax.set_title(SCALE_LABELS[scale], fontweight='bold', fontsize=11)
            if col == 0:
                ax.set_ylabel(label, fontsize=9)
            sns.despine(ax=ax, left=True, bottom=True)

        # One shared colorbar per row (right side)
        if row_sc is not None:
            cbar_ax = fig.add_subplot(gs[row, 3])
            cb = plt.colorbar(row_sc, cax=cbar_ax)
            cb.set_ticks(np.linspace(vmin, vmax, 5))
            cb.ax.tick_params(labelsize=7)

    fig.suptitle('True Spatially-Varying Coefficients', fontsize=12,
                 fontweight='bold', y=0.95)
    out = os.path.join(output_dir, 'fig_ground_truth.pdf')
    fig.savefig(out)
    plt.close(fig)
    print(f'Saved: {out}')


# ═════════════════════════════════════════════════════════════════════════════
# Figure 2: Main Results Heatmap
# ═════════════════════════════════════════════════════════════════════════════

def fig_main_heatmap(stats, tests, output_dir):
    """Heatmap of β₂ OLS slope and Pearson r across encoders, scales, conditions."""
    metrics = [
        ('ols_slope_mean', r'Amplitude Recovery (OLS Slope)'),
        ('pearson_r_mean', r'Pattern Recovery (Pearson $r$)'),
    ]
    conditions = [
        ('baseline', False, 'BL'),
        ('emb_only', False, 'EO'),
        ('emb_only', True,  'EO-T'),
    ]
    scales = ['global', 'county', 'grid']
    n_cond = len(conditions)
    n_enc  = len(ENCODER_ORDER)
    n_cols = len(scales) * n_cond

    col_labels = [f'{SCALE_LABELS[s]}\n{short}'
                  for s in scales for _, _, short in conditions]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6),
                             gridspec_kw={'wspace': 0.35})
    plt.subplots_adjust(left=0.16, right=0.96, top=0.86, bottom=0.22)

    for ax_idx, (metric, title) in enumerate(metrics):
        ax = axes[ax_idx]

        # Build data matrix
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
                    v = stats.loc[mask, metric]
                    if not v.empty:
                        matrix[i, col] = v.values[0]

        # Annotation strings: value + significance star
        metric_name = 'ols_slope' if 'slope' in metric else 'pearson_r'
        annot = np.full((n_enc, n_cols), '', dtype=object)
        for i, enc in enumerate(ENCODER_ORDER):
            for j in range(n_cols):
                val = matrix[i, j]
                annot[i, j] = '—' if np.isnan(val) else f'{val:.2f}'

        for i, enc in enumerate(ENCODER_ORDER):
            for j, scale in enumerate(scales):
                for k, (fc, trained, _) in enumerate(conditions):
                    if fc == 'baseline':
                        continue
                    col = j * n_cond + k
                    train_lbl = 'trained' if trained else 'untrained'
                    tmask = (
                        (tests['encoder'] == enc) &
                        (tests['scale'] == scale) &
                        (tests['comparison'] == f'emb_only_vs_baseline_{train_lbl}') &
                        (tests['spatial_effect'] == 'SVC_X2_Smooth') &
                        (tests['metric'] == metric_name)
                    )
                    rows = tests.loc[tmask]
                    if not rows.empty and rows.iloc[0]['significant']:
                        annot[i, col] += '*'

        vmin = 0.0 if 'slope' in metric else 0.3
        df_m = pd.DataFrame(matrix, index=ENCODER_ORDER, columns=col_labels)
        sns.heatmap(df_m, ax=ax, cmap='RdYlGn', vmin=vmin, vmax=1.0,
                    annot=annot, fmt='', annot_kws={'size': 7},
                    linewidths=0.4, linecolor='white',
                    cbar_kws={'shrink': 0.65, 'pad': 0.02})

        # Scale group separators
        for sep in [n_cond, 2 * n_cond]:
            ax.axvline(sep, color='white', linewidth=3)

        # Tier separators
        ax.axhline(3, color='white', linewidth=3)
        ax.axhline(9, color='white', linewidth=3)

        # Tier indicator: thin colored strip on the left edge of each encoder group
        for i, enc in enumerate(ENCODER_ORDER):
            tier = TIER_LABELS[enc]
            ax.add_patch(mpatches.Rectangle((-0.35, i + 0.05), 0.30, 0.90,
                                            facecolor=TIER_COLORS[tier],
                                            edgecolor=TIER_DARK[tier],
                                            linewidth=0.5,
                                            clip_on=False,
                                            transform=ax.transData))

        # Keep all tick labels black
        ax.set_yticklabels(ax.get_yticklabels(), color='black', fontsize=8)
        ax.set_title(title, fontsize=10, fontweight='bold', pad=10)
        ax.set_xlabel(''); ax.set_ylabel('')
        ax.tick_params(axis='x', labelsize=7.5)

    # Tier legend in upper-left of left panel
    tier_patches = [
        mpatches.Patch(facecolor=TIER_COLORS[t], edgecolor=TIER_DARK[t],
                       linewidth=0.8, label=f'Tier {t}')
        for t in [1, 2, 3]
    ]
    axes[0].legend(handles=tier_patches, loc='upper left', fontsize=7.5,
                   framealpha=0.85, title='Encoder tier', title_fontsize=7.5)

    fig.suptitle(r'$\beta_2$ SVC Recovery (Multi-Scale Oscillation)',
                 fontsize=12, fontweight='bold')
    fig.text(0.5, 0.07,
             'BL = Baseline (coords only)   '
             'EO = Embedding Only (untrained)   '
             'EO-T = Embedding Only (trained)   '
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
    panels = [
        ('Truth',                               None,                                 None),
        ('Baseline\n(coords only)',             'global_simple_baseline',             'none'),
        ('Sphere2Vec-dfs\n(Tier 1, untrained)', 'global_simple_embonly_dim8',         'Sphere2Vec-dfs'),
        ('Space2Vec-grid\n(Tier 2, trained)',   'global_simple_trained_embonly_dim8', 'Space2Vec-grid'),
        ('tile_ffn\n(Tier 3, untrained)',       'global_simple_embonly_dim8',         'tile_ffn'),
    ]
    n_panels = len(panels)

    # Load data
    data_list, truth_df = [], None
    for _, subdir, enc in panels:
        if subdir is None:
            data_list.append(None); continue
        csv = os.path.join(results_root, subdir, f'{enc}_MLP_rep0_spatial_effects.csv')
        df = pd.read_csv(csv)
        if truth_df is None:
            truth_df = df
        data_list.append(df)

    # Color ranges
    vmin_c = np.percentile(truth_df['b2_true'], 2)
    vmax_c = np.percentile(truth_df['b2_true'], 98)
    norm_c = Normalize(vmin=vmin_c, vmax=vmax_c)

    errs_all = np.concatenate([
        (data_list[i]['b2_smooth_estimated'] - truth_df['b2_true']).values
        for i, (_, sd, _) in enumerate(panels) if sd is not None
    ])
    max_err = np.percentile(np.abs(errs_all), 98)
    norm_e  = TwoSlopeNorm(vmin=-max_err, vcenter=0, vmax=max_err)

    # Layout: 2 rows × (n_panels + 1 cbar col each)
    fig = plt.figure(figsize=(12, 5.5))
    gs = gridspec.GridSpec(2, n_panels + 1,
                           width_ratios=[1]*n_panels + [0.04],
                           hspace=0.10, wspace=0.06,
                           left=0.03, right=0.95, top=0.88, bottom=0.03)

    im0_last = im1_last = None
    for col_idx, (label, subdir, _) in enumerate(panels):
        ax0 = fig.add_subplot(gs[0, col_idx])
        ax1 = fig.add_subplot(gs[1, col_idx])

        lon = truth_df['lon'].values
        lat = truth_df['lat'].values

        # Row 0: β₂ surface
        vals0 = truth_df['b2_true'].values if col_idx == 0 \
                else data_list[col_idx]['b2_smooth_estimated'].values
        im0 = _scatter_map(ax0, lon, lat, vals0,
                           cmap='coolwarm', norm=norm_c, s=1.0)
        im0_last = im0
        ax0.set_title(label, fontsize=7.5, pad=3)
        if col_idx == 0:
            ax0.set_ylabel(r'$\hat{\beta}_2$', fontsize=8)

        # Row 1: error map
        if col_idx == 0:
            ax1.axis('off')
        else:
            err = data_list[col_idx]['b2_smooth_estimated'].values - truth_df['b2_true'].values
            im1 = _scatter_map(ax1, lon, lat, err,
                               cmap='RdBu_r', norm=norm_e, s=1.0)
            im1_last = im1
            if col_idx == 1:
                ax1.set_ylabel('Error', fontsize=8)

    # Colorbars
    cbar_ax0 = fig.add_subplot(gs[0, n_panels])
    cb0 = plt.colorbar(im0_last, cax=cbar_ax0, label=r'$\beta_2$')
    cb0.set_ticks(np.linspace(vmin_c, vmax_c, 5))
    cb0.ax.tick_params(labelsize=7); cb0.ax.yaxis.label.set_size(8)

    cbar_ax1 = fig.add_subplot(gs[1, n_panels])
    cb1 = plt.colorbar(im1_last, cax=cbar_ax1, label='Error')
    cb1.set_ticks([-max_err, -max_err/2, 0, max_err/2, max_err])
    cb1.ax.tick_params(labelsize=7); cb1.ax.yaxis.label.set_size(8)

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
    metrics = [
        ('ols_slope_mean', 'OLS Slope (Amplitude)'),
        ('pearson_r_mean', r'Pearson $r$ (Pattern)'),
    ]
    n_enc = len(ENCODER_ORDER)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0),
                             gridspec_kw={'wspace': 0.25})
    plt.subplots_adjust(left=0.18, right=0.78, top=0.88, bottom=0.12)

    for ax_idx, (metric, xlabel) in enumerate(metrics):
        ax = axes[ax_idx]
        sns.despine(ax=ax, left=True, bottom=False)

        sub = stats[
            (stats['scale'] == 'global') &
            (stats['spatial_effect'] == 'SVC_X2_Smooth') &
            (stats['feature_config'] == 'emb_only') &
            (stats['embed_dim'] == 8) &
            (stats['encoder'].isin(ENCODER_ORDER))
        ]
        bl = stats[
            (stats['scale'] == 'global') &
            (stats['spatial_effect'] == 'SVC_X2_Smooth') &
            (stats['feature_config'] == 'baseline') &
            (stats['embed_dim'] == 0) &
            (stats['encoder'] == 'none')
        ]
        bl_val = bl[metric].values[0] if not bl.empty else None

        for i, enc in enumerate(ENCODER_ORDER):
            ut = sub[(sub['encoder'] == enc) & (sub['encoder_trained'] == False)]
            tr = sub[(sub['encoder'] == enc) & (sub['encoder_trained'] == True)]
            ut_val = ut[metric].values[0] if not ut.empty else np.nan
            tr_val = tr[metric].values[0] if not tr.empty else np.nan

            tier = TIER_LABELS[enc]
            dot_c = TIER_COLORS[tier]
            edge_c = TIER_DARK[tier]

            if not (np.isnan(ut_val) or np.isnan(tr_val)):
                lc = '#2ca02c' if tr_val > ut_val else '#d62728'
                ax.plot([ut_val, tr_val], [i, i], color=lc, alpha=0.45,
                        linewidth=1.8, zorder=1)

            if not np.isnan(ut_val):
                ax.scatter(ut_val, i, color=dot_c, edgecolors=edge_c,
                           linewidths=0.8, s=50, zorder=3, marker='o')
            if not np.isnan(tr_val):
                ax.scatter(tr_val, i, color=dot_c, edgecolors=edge_c,
                           linewidths=0.8, s=50, zorder=3, marker='D')

        if bl_val is not None:
            ax.axvline(bl_val, color='#888', linestyle='--', linewidth=0.9,
                       alpha=0.8, zorder=0)
            ax.text(bl_val + 0.003, n_enc - 0.6, 'Baseline',
                    fontsize=6.5, color='#888', ha='left', va='bottom')

        ax.set_yticks(range(n_enc))
        ax.set_yticklabels(ENCODER_ORDER if ax_idx == 0 else [], fontsize=7.5)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.invert_yaxis()
        ax.set_title(f'Global $\\beta_2$: {xlabel}', fontsize=9, fontweight='bold')
        ax.axhline(2.5, color='lightgray', linewidth=0.8)
        ax.axhline(8.5, color='lightgray', linewidth=0.8)

    # Legend to the right of both panels
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d0d0d0',
               markeredgecolor='#555', markersize=7, label='Untrained'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#d0d0d0',
               markeredgecolor='#555', markersize=7, label='Trained'),
        Line2D([0], [0], color='#2ca02c', linewidth=2, alpha=0.6, label='Training helps'),
        Line2D([0], [0], color='#d62728', linewidth=2, alpha=0.6, label='Training hurts'),
        mpatches.Patch(facecolor=TIER_COLORS[1], edgecolor=TIER_DARK[1], label='Tier 1'),
        mpatches.Patch(facecolor=TIER_COLORS[2], edgecolor=TIER_DARK[2], label='Tier 2'),
        mpatches.Patch(facecolor=TIER_COLORS[3], edgecolor=TIER_DARK[3], label='Tier 3'),
    ]
    fig.legend(handles=legend_elements, loc='center left', fontsize=8,
               frameon=True, framealpha=0.9,
               bbox_to_anchor=(0.80, 0.5), borderaxespad=0)

    out = os.path.join(output_dir, 'fig_training_effect.pdf')
    fig.savefig(out)
    plt.close(fig)
    print(f'Saved: {out}')


# ═════════════════════════════════════════════════════════════════════════════
# LaTeX Table: Main Results
# ═════════════════════════════════════════════════════════════════════════════

def _is_significant(tests, encoder, scale, metric, train_label):
    mask = (
        (tests['encoder'] == encoder) &
        (tests['scale'] == scale) &
        (tests['comparison'] == f'emb_only_vs_baseline_{train_label}') &
        (tests['spatial_effect'] == 'SVC_X2_Smooth') &
        (tests['metric'] == metric)
    )
    rows = tests.loc[mask]
    return bool(rows.iloc[0]['significant']) if not rows.empty else False


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

    line = r'\multicolumn{2}{l}{\textit{Baseline (coords only)}}'
    for scale in ['global', 'county', 'grid']:
        mask = (
            (stats['scale'] == scale) &
            (stats['feature_config'] == 'baseline') &
            (stats['encoder'] == 'none') &
            (stats['spatial_effect'] == 'SVC_X2_Smooth')
        )
        row = stats.loc[mask]
        if not row.empty:
            r_v = row['pearson_r_mean'].values[0]; r_s = row['pearson_r_std'].values[0]
            s_v = row['ols_slope_mean'].values[0]; s_s = row['ols_slope_std'].values[0]
            line += f' & {r_v:.3f}$\\pm${r_s:.3f} & {s_v:.3f}$\\pm${s_s:.3f}'
        else:
            line += r' & --- & ---'
    lines.append(line + r' \\')
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
                r_v = row['pearson_r_mean'].values[0]
                s_v = row['ols_slope_mean'].values[0]
                sr = '*' if _is_significant(tests, enc, scale, 'pearson_r', 'untrained') else ''
                ss = '*' if _is_significant(tests, enc, scale, 'ols_slope', 'untrained') else ''
                line += f' & {r_v:.3f}{sr} & {s_v:.3f}{ss}'
            else:
                line += r' & --- & ---'
        lines.append(line + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'}')
    lines.append(r'\end{table}')

    out = os.path.join(output_dir, 'tab_main_results.tex')
    with open(out, 'w') as f:
        f.write('\n'.join(lines))
    print(f'Saved: {out}')


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate paper figures')
    parser.add_argument('--results_root', type=str, default='./results')
    parser.add_argument('--output_dir',   type=str, default='./plots')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    stats, tests = load_data(args.results_root)

    print('Generating figures...')
    fig_ground_truth(args.results_root, args.output_dir)
    fig_main_heatmap(stats, tests, args.output_dir)
    fig_spatial_global(args.results_root, args.output_dir)
    fig_training_effect(stats, args.output_dir)
    tab_main_results(stats, tests, args.output_dir)
    print('Done!')
