"""
Aggregate experiment results across all conditions.

Scans results directories matching naming convention and produces:
  - combined_all_reps.csv         (every row from every experiment)
  - combined_summary_stats.csv    (mean/std per condition)
  - comparison_table.csv          (key metrics pivoted for easy reading)

Usage:
    python aggregate_metrics.py --results_root ./results
    python aggregate_metrics.py --results_dir ./results/grid   # single dir mode
"""

import argparse
import glob
import os
import re
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def parse_dir_name(dirname):
    """Extract experimental condition from directory name.

    Expected patterns:
        global_simple_dim8, global_full_embonly_dim32, global_simple_baseline, etc.
    """
    info = {'scale': None, 'dgp': None, 'feature_config': None, 'embed_dim': None, 'encoder_trained': False}

    for scale in ('global', 'grid', 'counties', 'county'):
        if dirname.startswith(scale):
            info['scale'] = 'county' if scale in ('counties', 'county') else scale
            break

    if '_simple_' in dirname:
        info['dgp'] = 'simple'
    elif '_full_' in dirname:
        info['dgp'] = 'full'

    if 'embonly' in dirname:
        info['feature_config'] = 'emb_only'
    elif 'baseline' in dirname:
        info['feature_config'] = 'baseline'
    else:
        info['feature_config'] = 'emb+coords'

    info['encoder_trained'] = 'trained' in dirname

    dim_match = re.search(r'dim(\d+)', dirname)
    if dim_match:
        info['embed_dim'] = int(dim_match.group(1))
    elif 'baseline' in dirname:
        info['embed_dim'] = 0

    return info


def aggregate_single(results_dir):
    """Combine summary CSVs from a single results directory."""
    summary_files = glob.glob(os.path.join(results_dir, "*_summary.csv"))
    frames = []
    for f in summary_files:
        try:
            df = pd.read_csv(f)
            frames.append(df)
        except Exception as e:
            print(f"  Error reading {f}: {e}")
    if frames:
        return pd.concat(frames, ignore_index=True)
    return None


def run_statistical_tests(combined, output_dir):
    """
    Paired Wilcoxon tests comparing feature configs and embed dims.

    Three comparisons (per encoder, scale, dgp, spatial_effect):
      1. emb+coords vs baseline        — do embeddings help over raw coords?
      2. emb_only vs baseline          — can embeddings replace raw coords?
      3. emb+coords vs emb_only        — does adding coords on top help?

    Uses dim=8 for all feature config comparisons (sweet spot).
    Uses pearson_r and ols_slope as test metrics.
    """
    print(f"\n{'='*80}")
    print("STATISTICAL TESTS (paired Wilcoxon, α=0.05)")
    print(f"{'='*80}")

    records = []
    effects = ['SVC_X1_Smooth', 'SVC_X2_Smooth']
    test_metrics = ['pearson_r', 'ols_slope']

    comparisons = [
        ('emb+coords', 'baseline',   'emb+coords_vs_baseline'),
        ('emb_only',   'baseline',   'emb_only_vs_baseline'),
        ('emb+coords', 'emb_only',   'emb+coords_vs_emb_only'),
    ]

    group_keys = [k for k in ['scale', 'dgp', 'encoder', 'spatial_effect'] if k in combined.columns]

    for scale in combined['scale'].dropna().unique():
        for dgp in combined['dgp'].dropna().unique():
            for effect in effects:
                for encoder in combined['encoder'].dropna().unique():
                    for metric in test_metrics:
                        for (cfg_a, cfg_b, label) in comparisons:
                            # Use dim=8 for feature config comparisons; dim=0 for baseline
                            dim_a = 0 if cfg_a == 'baseline' else 8
                            dim_b = 0 if cfg_b == 'baseline' else 8

                            mask_a = (
                                (combined['scale'] == scale) &
                                (combined['dgp'] == dgp) &
                                (combined['encoder'] == encoder) &
                                (combined['spatial_effect'] == effect) &
                                (combined['feature_config'] == cfg_a) &
                                (combined['embed_dim'] == dim_a)
                            )
                            mask_b = (
                                (combined['scale'] == scale) &
                                (combined['dgp'] == dgp) &
                                (combined['encoder'] == encoder) &
                                (combined['spatial_effect'] == effect) &
                                (combined['feature_config'] == cfg_b) &
                                (combined['embed_dim'] == dim_b)
                            )

                            a = combined.loc[mask_a, metric].dropna().values
                            b = combined.loc[mask_b, metric].dropna().values

                            if len(a) < 5 or len(b) < 5 or len(a) != len(b):
                                continue

                            try:
                                stat, p = wilcoxon(a, b, alternative='greater')
                                records.append({
                                    'scale': scale, 'dgp': dgp,
                                    'encoder': encoder, 'spatial_effect': effect,
                                    'metric': metric, 'comparison': label,
                                    'n': len(a),
                                    'mean_a': round(a.mean(), 4),
                                    'mean_b': round(b.mean(), 4),
                                    'mean_diff': round(a.mean() - b.mean(), 4),
                                    'statistic': round(stat, 4),
                                    'p_value': round(p, 4),
                                    'significant': p < 0.05,
                                })
                            except Exception:
                                pass

    if not records:
        print("  Not enough data for statistical tests yet.")
        return None

    results_df = pd.DataFrame(records)
    stat_file = os.path.join(output_dir, "statistical_tests.csv")
    results_df.to_csv(stat_file, index=False)
    print(f"Saved: {stat_file}")

    # Console summary: % significant per comparison
    print("\n  % encoders with significant improvement (p<0.05, b1_smooth, pearson_r):")
    b1_r = results_df[
        (results_df['spatial_effect'] == 'SVC_X1_Smooth') &
        (results_df['metric'] == 'pearson_r')
    ]
    for label in [c[2] for c in comparisons]:
        sub = b1_r[b1_r['comparison'] == label]
        if sub.empty:
            continue
        pct = sub['significant'].mean() * 100
        med_diff = sub['mean_diff'].median()
        print(f"    {label}: {pct:.0f}% significant, median Δ={med_diff:+.4f}")

    return results_df


def aggregate_all(results_root, output_dir=None, prefix=None):
    """Aggregate across all result directories under results_root."""
    if output_dir is None:
        output_dir = results_root

    prefixes = tuple(prefix.split(',')) if prefix else ('global_', 'grid_', 'counties_')
    result_dirs = sorted([
        d for d in os.listdir(results_root)
        if os.path.isdir(os.path.join(results_root, d))
        and any(d.startswith(p) for p in prefixes)
    ])
    print(f"Found {len(result_dirs)} result directories")

    all_frames = []
    for dirname in result_dirs:
        dirpath = os.path.join(results_root, dirname)
        info = parse_dir_name(dirname)
        print(f"  {dirname} -> {info}")

        df = aggregate_single(dirpath)
        if df is not None:
            for k, v in info.items():
                df[k] = v
            all_frames.append(df)

    if not all_frames:
        print("No data found!")
        return

    combined = pd.concat(all_frames, ignore_index=True)
    print(f"\nTotal rows: {len(combined)}")

    # Save raw
    raw_file = os.path.join(output_dir, "combined_all_reps.csv")
    combined.to_csv(raw_file, index=False)
    print(f"Saved: {raw_file}")

    # Summary stats
    group_cols = ['scale', 'dgp', 'feature_config', 'embed_dim', 'encoder', 'model', 'spatial_effect']
    group_cols = [c for c in group_cols if c in combined.columns]
    exclude = set(group_cols + ['repetition'])
    metric_cols = [c for c in combined.columns
                   if c not in exclude and combined[c].dtype in ('float64', 'float32', 'int64')]

    stats = combined.groupby(group_cols)[metric_cols].agg(['mean', 'std']).round(4)
    stats.columns = ['_'.join(col) for col in stats.columns]
    stats.reset_index(inplace=True)

    stats_file = os.path.join(output_dir, "combined_summary_stats.csv")
    stats.to_csv(stats_file, index=False)
    print(f"Saved: {stats_file}")

    # Comparison table: key metrics for smoothed SVCs
    key_metrics = ['pearson_r_mean', 'ols_slope_mean', 'rmse_mean', 'r2_score_mean']
    key_metrics = [m for m in key_metrics if m in stats.columns]

    for effect in ('SVC_X1_Smooth', 'SVC_X2_Smooth'):
        mask = stats['spatial_effect'] == effect
        if mask.any():
            pivot_cols = [c for c in group_cols if c != 'spatial_effect'] + key_metrics
            pivot = stats.loc[mask, pivot_cols].sort_values(
                ['dgp', 'feature_config', 'embed_dim', 'encoder'])
            table_file = os.path.join(output_dir, f"comparison_{effect}.csv")
            pivot.to_csv(table_file, index=False)
            print(f"Saved: {table_file}")

    # Statistical tests
    stat_results = run_statistical_tests(combined, output_dir)

    # Quick console summary
    print(f"\n{'='*80}")
    print("QUICK SUMMARY (smoothed b1, mean Pearson r across reps)")
    print(f"{'='*80}")
    b1 = stats[stats['spatial_effect'] == 'SVC_X1_Smooth'].copy()
    if not b1.empty and 'pearson_r_mean' in b1.columns:
        pivot = b1.pivot_table(
            index='encoder', columns=['dgp', 'feature_config', 'embed_dim'],
            values='pearson_r_mean', aggfunc='first')
        print(pivot.to_string())

    print(f"\n{'='*80}")
    print("QUICK SUMMARY (smoothed b2, mean Pearson r across reps)")
    print(f"{'='*80}")
    b2 = stats[stats['spatial_effect'] == 'SVC_X2_Smooth'].copy()
    if not b2.empty and 'pearson_r_mean' in b2.columns:
        pivot = b2.pivot_table(
            index='encoder', columns=['dgp', 'feature_config', 'embed_dim'],
            values='pearson_r_mean', aggfunc='first')
        print(pivot.to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aggregate experiment results')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Single directory containing *_summary.csv files')
    parser.add_argument('--results_root', type=str, default=None,
                        help='Root directory containing multiple result dirs')
    parser.add_argument('--prefix', type=str, default=None,
                        help='Comma-separated prefixes to filter dirs, e.g. "global_simple_,global_full_"')
    args = parser.parse_args()

    if args.results_root:
        aggregate_all(args.results_root, prefix=getattr(args, 'prefix', None))
    elif args.results_dir:
        df = aggregate_single(args.results_dir)
        if df is not None:
            out = os.path.join(args.results_dir, "all_encoders_all_repetitions.csv")
            df.to_csv(out, index=False)
            print(f"Saved: {out}")
    else:
        parser.error("Provide either --results_root or --results_dir")
