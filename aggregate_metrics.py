"""
Aggregate per-encoder summary CSVs into combined results.

Reads all *_summary.csv files from a results directory and produces:
  - all_encoders_all_repetitions.csv  (raw data)
  - all_encoders_summary_stats.csv    (mean/std per encoder+model+effect)

Usage:
    python aggregate_metrics.py --results_dir ./results/grid
"""

import argparse
import glob
import os
import traceback

import numpy as np
import pandas as pd


def aggregate(results_dir):
    """Combine summary CSVs and compute mean/std statistics."""
    print(f"Aggregating results from: {results_dir}")

    summary_files = glob.glob(os.path.join(results_dir, "*_summary.csv"))
    print(f"Found {len(summary_files)} summary files")

    all_encoder_summaries = []
    for summary_file in summary_files:
        try:
            filename = os.path.basename(summary_file)
            print(f"  Processing: {filename}")
            df = pd.read_csv(summary_file)
            all_encoder_summaries.append(df)
        except Exception as e:
            print(f"  Error processing {summary_file}: {e}")
            traceback.print_exc()

    if not all_encoder_summaries:
        print("No summary files found to aggregate!")
        return

    combined_df = pd.concat(all_encoder_summaries, ignore_index=True)
    print(f"\nCombined {len(all_encoder_summaries)} files -> {combined_df.shape}")

    # Save raw combined data
    all_reps_file = os.path.join(results_dir, "all_encoders_all_repetitions.csv")
    combined_df.to_csv(all_reps_file, index=False)
    print(f"Saved: {all_reps_file}")

    # Compute mean/std across repetitions
    grouping_cols = ['encoder', 'model', 'spatial_effect']
    exclude_cols = grouping_cols + ['repetition']
    metric_cols = [c for c in combined_df.columns if c not in exclude_cols]

    summary_stats = combined_df.groupby(grouping_cols)[metric_cols].agg(['mean', 'std', 'count'])
    summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]
    summary_stats.reset_index(inplace=True)

    summary_file = os.path.join(results_dir, "all_encoders_summary_stats.csv")
    summary_stats.to_csv(summary_file, index=False)
    print(f"Saved: {summary_file}")
    print(f"\nUnique encoders: {summary_stats['encoder'].unique()}")
    print(f"Unique models: {summary_stats['model'].unique()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aggregate experiment results')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing *_summary.csv files')
    args = parser.parse_args()
    aggregate(args.results_dir)
