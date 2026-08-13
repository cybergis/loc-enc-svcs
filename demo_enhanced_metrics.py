"""
Demo: Using Enhanced Metrics to Diagnose Spatial Effect Recovery

This script demonstrates how the new enhanced metrics help diagnose
amplitude compression and other issues in location encoder experiments.
"""

import numpy as np
import pandas as pd
from help_utils import calculate_spatial_metrics, interpret_metrics

print("="*80)
print("DEMO: Enhanced Metrics for Spatial Effect Recovery")
print("="*80)

# Create synthetic examples of different recovery scenarios
np.random.seed(42)
n_points = 625  # 25x25 grid
grid_size = 25

# True surface (ground truth)
true_surface = np.linspace(1, 5, n_points)  # Simple gradient from 1 to 5

# Create fake coords for Moran's I
fake_coords = np.random.randn(n_points, 2)

print("\nGround Truth: Linear gradient from 1.0 to 5.0")
print(f"  Range: [{true_surface.min():.2f}, {true_surface.max():.2f}]")
print(f"  Std: {true_surface.std():.3f}")

# ========== Scenario 1: Perfect Recovery ==========
print("\n" + "-"*80)
print("SCENARIO 1: Perfect Recovery (Ideal Case)")
print("-"*80)

perfect_recovery = true_surface + np.random.normal(0, 0.1, n_points)  # Small noise

metrics_perfect = calculate_spatial_metrics(
    true_surface=true_surface,
    estimated_surface=perfect_recovery,
    effect_name="Perfect_Recovery",
    encoder_name="Ideal",
    model_name="Test",
    coords_for_moran=fake_coords,
    grid_size=grid_size
)

print(f"\nKey Metrics:")
print(f"  Pearson r:         {metrics_perfect['pearson_r']:.3f}")
print(f"  OLS Slope:         {metrics_perfect['ols_slope']:.3f}  ← Should be ~1.0")
print(f"  Amplitude Ratio:   {metrics_perfect['amplitude_range_ratio']:.3f}  ← Should be ~1.0")
print(f"  RMSE (normalized): {metrics_perfect['rmse_normalized_by_std']:.3f}  ← Should be small")

interpret_metrics(metrics_perfect, verbose=True)

# ========== Scenario 2: Amplitude Compression (Your Current Issue) ==========
print("\n" + "-"*80)
print("SCENARIO 2: Amplitude Compression (Shape OK, Magnitude Weak)")
print("-"*80)
print("This simulates what you're seeing in your results!")

# Compress to 10% of true amplitude (what your models might be doing)
compressed = 2.5 + 0.1 * (true_surface - true_surface.mean())  # 10% of variance
compressed += np.random.normal(0, 0.1, n_points)

metrics_compressed = calculate_spatial_metrics(
    true_surface=true_surface,
    estimated_surface=compressed,
    effect_name="Compressed_Amplitude",
    encoder_name="NeRF",
    model_name="MLP",
    coords_for_moran=fake_coords,
    grid_size=grid_size
)

print(f"\nKey Metrics:")
print(f"  Pearson r:         {metrics_compressed['pearson_r']:.3f}  ← High! (shape captured)")
print(f"  OLS Slope:         {metrics_compressed['ols_slope']:.3f}  ← Low! (amplitude lost)")
print(f"  Amplitude Ratio:   {metrics_compressed['amplitude_range_ratio']:.3f}  ← Only 10% recovered!")
print(f"  RMSE (normalized): {metrics_compressed['rmse_normalized_by_std']:.3f}")

interpret_metrics(metrics_compressed, verbose=True)

# ========== Scenario 3: Shape Mismatch ==========
print("\n" + "-"*80)
print("SCENARIO 3: Shape Mismatch (Wrong Pattern)")
print("-"*80)

# Random pattern (model failed to learn)
wrong_shape = np.random.uniform(1, 5, n_points)

metrics_wrong = calculate_spatial_metrics(
    true_surface=true_surface,
    estimated_surface=wrong_shape,
    effect_name="Wrong_Shape",
    encoder_name="BadEncoder",
    model_name="MLP",
    coords_for_moran=fake_coords,
    grid_size=grid_size
)

print(f"\nKey Metrics:")
print(f"  Pearson r:         {metrics_wrong['pearson_r']:.3f}  ← Low! (no pattern)")
print(f"  OLS Slope:         {metrics_wrong['ols_slope']:.3f}")
print(f"  Amplitude Ratio:   {metrics_wrong['amplitude_range_ratio']:.3f}")
print(f"  RMSE (normalized): {metrics_wrong['rmse_normalized_by_std']:.3f}")

interpret_metrics(metrics_wrong, verbose=True)

# ========== Scenario 4: Good Recovery (Realistic Goal) ==========
print("\n" + "-"*80)
print("SCENARIO 4: Good Recovery (Realistic Target)")
print("-"*80)

# 80% amplitude + noise (realistic good case)
good_recovery = 2.5 + 0.8 * (true_surface - true_surface.mean())
good_recovery += np.random.normal(0, 0.3, n_points)

metrics_good = calculate_spatial_metrics(
    true_surface=true_surface,
    estimated_surface=good_recovery,
    effect_name="Good_Recovery",
    encoder_name="Space2Vec",
    model_name="XGBoost",
    coords_for_moran=fake_coords,
    grid_size=grid_size
)

print(f"\nKey Metrics:")
print(f"  Pearson r:         {metrics_good['pearson_r']:.3f}  ← Good!")
print(f"  OLS Slope:         {metrics_good['ols_slope']:.3f}  ← Good!")
print(f"  Amplitude Ratio:   {metrics_good['amplitude_range_ratio']:.3f}  ← 80% recovered")
print(f"  RMSE (normalized): {metrics_good['rmse_normalized_by_std']:.3f}")

interpret_metrics(metrics_good, verbose=True)

# ========== Summary Comparison ==========
print("\n" + "="*80)
print("SUMMARY COMPARISON")
print("="*80)

comparison_df = pd.DataFrame([
    {
        "Scenario": "Perfect",
        "Pearson r": metrics_perfect['pearson_r'],
        "OLS Slope": metrics_perfect['ols_slope'],
        "Amp Ratio": metrics_perfect['amplitude_range_ratio'],
        "RMSE (norm)": metrics_perfect['rmse_normalized_by_std'],
        "Interpretation": "Ideal case"
    },
    {
        "Scenario": "Compressed",
        "Pearson r": metrics_compressed['pearson_r'],
        "OLS Slope": metrics_compressed['ols_slope'],
        "Amp Ratio": metrics_compressed['amplitude_range_ratio'],
        "RMSE (norm)": metrics_compressed['rmse_normalized_by_std'],
        "Interpretation": "YOUR CURRENT ISSUE"
    },
    {
        "Scenario": "Wrong Shape",
        "Pearson r": metrics_wrong['pearson_r'],
        "OLS Slope": metrics_wrong['ols_slope'],
        "Amp Ratio": metrics_wrong['amplitude_range_ratio'],
        "RMSE (norm)": metrics_wrong['rmse_normalized_by_std'],
        "Interpretation": "Model failed"
    },
    {
        "Scenario": "Good",
        "Pearson r": metrics_good['pearson_r'],
        "OLS Slope": metrics_good['ols_slope'],
        "Amp Ratio": metrics_good['amplitude_range_ratio'],
        "RMSE (norm)": metrics_good['rmse_normalized_by_std'],
        "Interpretation": "Target performance"
    }
])

print("\n" + comparison_df.to_string(index=False))

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)
print("""
1. HIGH Pearson r + LOW OLS Slope = AMPLITUDE COMPRESSION
   → This is likely your issue! Model captures shape but not magnitude.
   
2. LOW Pearson r = SHAPE MISMATCH  
   → Model not learning spatial pattern at all.
   
3. OLS Slope ≈ 1.0 + High Pearson r = GOOD RECOVERY
   → This is your target!

4. Use Amplitude Ratio to see what % of signal is recovered.
   → Your current results likely show 10-30% recovery.

FIXES TO TRY:
- Feature scaling (StandardScaler on embeddings + features)
- Reduce MLP regularization (alpha parameter)
- Use proper geographic coordinates (now implemented!)
- Increase model capacity (more hidden layers/neurons)
- Try XGBoost (less sensitive to scaling)
""")

print("="*80)
print("\n✓ Enhanced metrics now integrated into help_utils.py")
print("✓ Run your experiments and check OLS slope + amplitude ratio!")
print("="*80)
