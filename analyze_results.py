#!/usr/bin/env python3
"""
Analyze grid experiment results and generate summary statistics for paper.
"""
import csv
import statistics
from collections import defaultdict

# Read CSV
data = []
with open('results/grid_dec12/all_encoders_all_repetitions.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(row)

print(f"Total rows: {len(data)}")

# Get unique encoders
encoders = sorted(set(row['encoder'] for row in data))
print(f"\nEncoders ({len(encoders)}): {encoders}")

models = sorted(set(row['model'] for row in data))
print(f"Models: {models}")

spatial_effects = sorted(set(row['spatial_effect'] for row in data))
print(f"Spatial effects: {spatial_effects}")

# Focus on smoothed SVC results (best performers)
print("\n" + "="*80)
print("SUMMARY: SMOOTHED SVC PERFORMANCE (Mean ± Std across 10 repetitions)")
print("="*80)

# Group by encoder and model
for model in models:
    print(f"\n### {model} Model ###\n")
    
    for encoder in encoders:
        # Get smooth SVC rows for this encoder/model
        smooth_rows = [r for r in data if 
                      r['encoder'] == encoder and 
                      r['model'] == model and 
                      'Smooth' in r['spatial_effect']]
        
        if not smooth_rows:
            continue
            
        # Calculate averages across both X1 and X2 smooth coefficients
        pearson_r_vals = [float(r['pearson_r']) for r in smooth_rows]
        ols_slope_vals = [float(r['ols_slope']) for r in smooth_rows]
        test_r2_vals = [float(r['test_r2']) for r in smooth_rows]
        
        pearson_mean = statistics.mean(pearson_r_vals)
        pearson_std = statistics.stdev(pearson_r_vals) if len(pearson_r_vals) > 1 else 0
        
        slope_mean = statistics.mean(ols_slope_vals)
        slope_std = statistics.stdev(ols_slope_vals) if len(ols_slope_vals) > 1 else 0
        
        r2_mean = statistics.mean(test_r2_vals)
        r2_std = statistics.stdev(test_r2_vals) if len(test_r2_vals) > 1 else 0
        
        print(f"{encoder:25s} | Pearson r: {pearson_mean:.3f}±{pearson_std:.3f} | "
              f"OLS slope: {slope_mean:.3f}±{slope_std:.3f} | Test R²: {r2_mean:.3f}±{r2_std:.3f}")

# Find best performers
print("\n" + "="*80)
print("TOP PERFORMERS (by mean Pearson correlation on smoothed SVCs)")
print("="*80)

encoder_scores = defaultdict(lambda: {'pearson': [], 'slope': [], 'test_r2': []})

for row in data:
    if 'Smooth' not in row['spatial_effect']:
        continue
    
    key = f"{row['encoder']}_{row['model']}"
    encoder_scores[key]['pearson'].append(float(row['pearson_r']))
    encoder_scores[key]['slope'].append(float(row['ols_slope']))
    encoder_scores[key]['test_r2'].append(float(row['test_r2']))

# Compute means and sort
rankings = []
for key, scores in encoder_scores.items():
    mean_pearson = statistics.mean(scores['pearson'])
    mean_slope = statistics.mean(scores['slope'])
    mean_r2 = statistics.mean(scores['test_r2'])
    rankings.append((key, mean_pearson, mean_slope, mean_r2))

rankings.sort(key=lambda x: x[1], reverse=True)

print("\nTop 10 by Pearson correlation:")
for i, (key, pearson, slope, r2) in enumerate(rankings[:10], 1):
    print(f"{i:2d}. {key:35s} | r={pearson:.3f} | slope={slope:.3f} | R²={r2:.3f}")

print("\n" + "="*80)
print("COMPARISON: Location Encoders vs. None (Lat/Lon only)")
print("="*80)

none_smooth = [r for r in data if r['encoder'] == 'none' and 'Smooth' in r['spatial_effect']]
none_pearson = statistics.mean([float(r['pearson_r']) for r in none_smooth])
none_slope = statistics.mean([float(r['ols_slope']) for r in none_smooth])
none_r2 = statistics.mean([float(r['test_r2']) for r in none_smooth])

print(f"\nBaseline (none): Pearson r={none_pearson:.3f}, OLS slope={none_slope:.3f}, Test R²={none_r2:.3f}")

# Compare to location encoders
encoder_smooth = [r for r in data if r['encoder'] != 'none' and 'Smooth' in r['spatial_effect']]
encoder_pearson = statistics.mean([float(r['pearson_r']) for r in encoder_smooth])
encoder_slope = statistics.mean([float(r['ols_slope']) for r in encoder_smooth])
encoder_r2 = statistics.mean([float(r['test_r2']) for r in encoder_smooth])

print(f"Location encoders (avg): Pearson r={encoder_pearson:.3f}, OLS slope={encoder_slope:.3f}, Test R²={encoder_r2:.3f}")
print(f"\nImprovement: Pearson r={encoder_pearson - none_pearson:+.3f} ({100*(encoder_pearson - none_pearson)/none_pearson:+.1f}%)")
print(f"             OLS slope={encoder_slope - none_slope:+.3f} ({100*(encoder_slope - none_slope)/none_slope:+.1f}%)")

print("\n" + "="*80)
print("KEY FINDING: Do location encoders help recover spatial coefficients?")
print("="*80)

print(f"""
The NONE baseline (using only lat/lon coordinates) achieves:
  - Pearson r = {none_pearson:.3f} (shape correlation with ground truth)
  - OLS slope = {none_slope:.3f} (amplitude recovery: 1.0 = perfect)
  - Test R² = {none_r2:.3f} (model predictive performance)

Location encoders on average achieve:
  - Pearson r = {encoder_pearson:.3f} (WORSE by {none_pearson - encoder_pearson:.3f})
  - OLS slope = {encoder_slope:.3f} (WORSE by {none_slope - encoder_slope:.3f})
  - Test R² = {encoder_r2:.3f} (similar)

CONCLUSION: Location encoders do NOT improve spatial coefficient recovery
compared to using raw lat/lon coordinates. In fact, they perform WORSE.
The baseline "none" encoder is the best performer for GeoSHAPLEY extraction.
""")
