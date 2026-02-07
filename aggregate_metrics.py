import pandas as pd
import os
import glob
import numpy as np

# --- Configuration ---
BASE_RESULTS_DIR = "./results/grid_dec12"
OUTPUT_DIR = BASE_RESULTS_DIR  # Save combined files in the base results directory

# --- Aggregation from FLAT structure (gridRun.py and countiesRun.py save files flat) ---
# gridRun.py saves: {encoder_name}_{model_type}_summary.csv (and rep0_metrics.csv, etc.)
# So we need to aggregate all *_summary.csv files in the BASE_RESULTS_DIR

print(f"Aggregating results from: {BASE_RESULTS_DIR}")

all_encoder_summaries = []

# Find all summary files in the flat structure
summary_files = glob.glob(os.path.join(BASE_RESULTS_DIR, "*_summary.csv"))
print(f"Found {len(summary_files)} summary files")

for summary_file in summary_files:
    try:
        filename = os.path.basename(summary_file)
        print(f"... Processing: {filename}")
        
        # Read the summary file (already has encoder, model, and all metrics columns)
        df = pd.read_csv(summary_file)
        all_encoder_summaries.append(df)
        
    except Exception as e:
        print(f"Error processing {summary_file}: {e}")
        import traceback
        traceback.print_exc()


# --- Combine and Save ---
if all_encoder_summaries:
    # Combine all encoder summaries (each has repetitions for one encoder)
    combined_df = pd.concat(all_encoder_summaries, ignore_index=True)
    
    print(f"\nCombined data from {len(all_encoder_summaries)} summary files")
    print(f"Combined DataFrame shape: {combined_df.shape}")
    print(f"Columns: {list(combined_df.columns)}")
    
    # Save raw combined data (all repetitions, all encoders)
    all_reps_file = os.path.join(OUTPUT_DIR, "all_encoders_all_repetitions.csv")
    combined_df.to_csv(all_reps_file, index=False)
    print(f"\nSaved all repetitions to {all_reps_file}")
    
    # Compute mean and std across repetitions for each encoder+model+spatial_effect
    print("\nComputing mean/std statistics across repetitions...")
    
    # Get all metric columns (exclude encoder, model, spatial_effect, repetition)
    grouping_cols = ['encoder', 'model', 'spatial_effect']
    exclude_cols = grouping_cols + ['repetition']
    metric_cols = [c for c in combined_df.columns if c not in exclude_cols]
    
    # Group by encoder, model, and spatial_effect, compute mean/std for each metric
    summary_stats = combined_df.groupby(grouping_cols)[metric_cols].agg(['mean', 'std', 'count'])
    
    # Flatten multi-level columns
    summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]
    summary_stats.reset_index(inplace=True)
    
    # Save summary with mean/std
    summary_file = os.path.join(OUTPUT_DIR, "all_encoders_summary_stats.csv")
    summary_stats.to_csv(summary_file, index=False)
    
    print(f"\nSaved summary statistics (mean/std) to {summary_file}")
    print(f"\nPreview of summary stats:")
    print(summary_stats.head(10))
    print(f"\nShape: {summary_stats.shape}")
    print(f"Unique encoders: {summary_stats['encoder'].unique()}")
    print(f"Unique models: {summary_stats['model'].unique()}")
else:
    print("No summary files found to aggregate!")

print("\nAggregation script finished.")
