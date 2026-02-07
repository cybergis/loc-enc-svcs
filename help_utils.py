import torch
import numpy as np
import matplotlib.pyplot as plt

# --- Metric Calculation Imports ---
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

# +++ Additions for Moran's I +++
from libpysal import weights
from esda.moran import Moran

# +++++++++++++++++++++++++++++++
import os, sys

# 1) Where is this script?
HERE = os.path.dirname(os.path.abspath(__file__))

# 2) Point at the TorchSpatial/main folder so that its .py files become top-level modules
TS_DIR = os.path.join(HERE, "TorchSpatial", "main")

# 3) Prepend it to sys.path
if TS_DIR not in sys.path:
    sys.path.insert(0, TS_DIR)

# 4) Now import exactly as TorchSpatial expects internally:
from SpatialRelationEncoder import *
from module import *
from data_utils import *
from utils import *

SPA_EMBED_DIM = 4  # Default embedding dimension for spatial encoders


def get_loc_embeddings(coords, encoder_type, extent=None, device="cpu"):
    """
    Compute location embeddings for 2D coordinates using the specified spatial encoder type.

    Parameters:
        coords (np.ndarray): Array of shape [batch_size, 2] containing the coordinates.
        encoder_type (str): The string identifier for the spatial encoder (e.g., 'Space2Vec-grid', 'NeRF', etc).
        extent (tuple): The spatial extent as (x_min, x_max, y_min, y_max). 
                       For geographic data, use (lon_min, lon_max, lat_min, lat_max).
                       If None, will use a default that may not be appropriate for your data.
        device (str): Device to use for the computation ('cpu', 'cuda:0', etc).

    Returns:
        torch.Tensor: The location embeddings, a tensor of shape [batch_size, spa_embed_dim].
    """
    # Handle extent parameter
    if extent is None:
        print("⚠️  WARNING: No extent provided to get_loc_embeddings().")
        print("   Using default extent (0, 200, 0, 200) which may not match your data!")
        print("   For geographic coordinates, pass extent=(lon_min, lon_max, lat_min, lat_max)")
        print("   For grid coordinates, pass the actual coordinate ranges.")
        extent = (0, 200, 0, 200)
    
    # Define the parameter dictionary.
    params = {
        "spa_enc_type": encoder_type,  # use the provided encoder type
        "spa_embed_dim": SPA_EMBED_DIM,  # embedding dimension
        "extent": extent,  # extent of the coordinates (now configurable!)
        "freq": 16,  # number of scales (related to multi-scale Fourier features)
        "max_radius": 1,  # maximum scale (lambda_max)
        "min_radius": 0.0001,  # minimum scale (lambda_min)
        "spa_f_act": "leakyrelu",  # non-linear activation function
        "freq_init": "geometric",  # Fourier frequency initialization
        "num_hidden_layer": 1,  # number of hidden layers in the encoder
        "dropout": 0.5,  # dropout rate
        "hidden_dim": 64,  # hidden dimension of the MLP (if applicable)
        "use_layn": True,  # use layer normalization flag
        "skip_connection": True,  # apply skip connections
        "spa_enc_use_postmat": True,  # whether to use the post-processing matrix
        "device": device,  # device for computation
    }

    # Instantiate the spatial relation encoder using the parameters.
    loc_enc = get_spa_encoder(
        train_locs=[],  # no training coordinates provided here
        params=params,
        spa_enc_type=params["spa_enc_type"],
        spa_embed_dim=params["spa_embed_dim"],
        extent=params["extent"],
        coord_dim=2,  # working in 2D
        frequency_num=params["freq"],
        max_radius=params["max_radius"],
        min_radius=params["min_radius"],
        f_act=params["spa_f_act"],
        freq_init=params["freq_init"],
        use_postmat=params["spa_enc_use_postmat"],
        device=params["device"],
    ).to(params["device"])

    # Ensure coords is a NumPy array. If coords is 2D ([batch_size, 2]),
    # expand dims so that it has shape [batch_size, 1, 2] as required by the encoder.
    coords = np.array(coords)
    if coords.ndim == 2:
        coords = np.expand_dims(coords, axis=1)

    # Pass the coordinates through the encoder.
    # The mapping is mathematically represented as: loc_embeds = f(coords)
    loc_embeds = torch.squeeze(loc_enc(coords))
    return loc_embeds


# --- Plotting Function ---
def plot_s(
    bs, size, vmin=None, vmax=None, title="", filename=None, experiment_dir=None
):
    """
    Plots spatial coefficient surfaces and saves the figure to a specific directory.
    Now handles potential vmin/vmax being passed for individual plots via lists.
    """
    if not isinstance(bs, list):
        if isinstance(bs, np.ndarray) and bs.ndim == 2 and bs.shape[1] == size * size:
            bs = [bs[i, :] for i in range(bs.shape[0])]
        elif isinstance(bs, np.ndarray) and bs.ndim == 1 and bs.shape[0] == size * size:
            bs = [bs]
        else:
            print(
                f"Error: Invalid input shape/type for plot_s: {type(bs)}. Expected list or array of shape ({size*size},)."
            )
            if bs is not None:
                print(
                    f"Actual shape: {bs.shape if isinstance(bs, np.ndarray) else 'N/A'}"
                )
            return

    k = len(bs)
    fig, axs = plt.subplots(1, k, figsize=(6 * k, 4), dpi=300)
    if k == 1:
        axs = [axs]

    vmin_list = vmin if isinstance(vmin, list) else [vmin] * k
    vmax_list = vmax if isinstance(vmax, list) else [vmax] * k
    if len(vmin_list) != k or len(vmax_list) != k:
        print(
            "Warning: Length of vmin/vmax lists does not match number of plots. Using first value or None."
        )
        vmin_list = [vmin_list[0]] * k
        vmax_list = [vmax_list[0]] * k

    plots_successful = 0
    for i in range(k):
        current_vmin = vmin_list[i]
        current_vmax = vmax_list[i]
        if (
            bs[i] is not None
            and hasattr(bs[i], "shape")
            and bs[i].shape == (size * size,)
        ):
            is_constant = np.all(bs[i] == bs[i][0]) if bs[i].size > 0 else True
            if (
                current_vmin is not None
                and current_vmax is not None
                and np.isclose(current_vmin, current_vmax)
            ):
                if not is_constant:
                    print(
                        f"Note: vmin ({current_vmin:.2f}) and vmax ({current_vmax:.2f}) are very close for non-constant data in plot {i}. Adjusting slightly."
                    )
                    buffer = (
                        0.1 * abs(current_vmin) if abs(current_vmin) > 1e-6 else 0.1
                    )
                    current_vmin -= buffer
                    current_vmax += buffer
            try:
                im = axs[i].imshow(
                    bs[i].reshape(size, size),
                    cmap="viridis",
                    vmin=current_vmin,
                    vmax=current_vmax,
                )
                fig.colorbar(im, ax=axs[i])
                axs[i].set_xticks([])
                axs[i].set_yticks([])
                axs[i].set_xticklabels([])
                axs[i].set_yticklabels([])
                if isinstance(title, list) and i < len(title):
                    axs[i].set_title(title[i])
                plots_successful += 1
            except Exception as img_err:
                print(f"Error during imshow/colorbar for plot {i}: {img_err}")
                axs[i].text(
                    0.5,
                    0.5,
                    f"Plot Failed\n({type(img_err).__name__})",
                    ha="center",
                    va="center",
                    transform=axs[i].transAxes,
                    color="red",
                )
                plot_title = f"Component {i+1} (Failed)"
                if isinstance(title, list) and i < len(title):
                    plot_title = f"{title[i]} (Failed)"
                axs[i].set_title(plot_title)
        else:
            reason = (
                "Shape Mismatch"
                if bs[i] is not None and hasattr(bs[i], "shape")
                else "Data Missing/Invalid"
            )
            actual_shape_info = (
                f"Actual shape: {bs[i].shape}"
                if bs[i] is not None and hasattr(bs[i], "shape")
                else ""
            )
            print(
                f"Warning: Skipping plot for component {i} ({reason}). {actual_shape_info}"
            )
            axs[i].text(
                0.5,
                0.5,
                f"Plot Skipped\n({reason})",
                ha="center",
                va="center",
                transform=axs[i].transAxes,
                color="red",
            )
            plot_title = f"Component {i+1} (Skipped)"
            if isinstance(title, list) and i < len(title):
                plot_title = f"{title[i]} (Skipped)"
            axs[i].set_title(plot_title)

    if isinstance(title, str) and title:
        try:
            fig.suptitle(title, fontsize=16, y=1.02)
        except Exception as suptitle_err:
            print(f"Error setting suptitle '{title}': {suptitle_err}")
    try:
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    except Exception as layout_err:
        print(f"Error during tight_layout: {layout_err}. Plot might overlap.")

    if filename and experiment_dir and plots_successful > 0:
        save_path = os.path.join(experiment_dir, filename)
        try:
            plt.savefig(save_path, bbox_inches="tight")
            print(f"Saved figure: {save_path}")
        except Exception as e:
            print(f"Error saving figure {save_path}: {e}")
        plt.close(fig)
    elif plots_successful == 0 and filename:
        print(f"No plots successful for '{filename}'. Closing figure.")
        plt.close(fig)
    elif not filename:
        plt.close(fig)


# --- Spatial Effect Metric Calculation Functions ---
def calculate_spatial_metrics(
    true_surface,
    estimated_surface,
    effect_name,
    encoder_name,
    model_name,
    coords_for_moran,
    grid_size=None,
):  # ++ Added coords_for_moran
    """
    Calculates various metrics comparing true and estimated spatial surfaces.
    Now includes Moran's I for the residuals.

    Parameters:
        true_surface (np.ndarray): The true surface values.
        estimated_surface (np.ndarray): The estimated surface values.
        effect_name (str): Name of the effect being measured (e.g., "Intercept", "SVC_X1_Raw").
        encoder_name (str): Name of the encoder used.
        model_name (str): Name of the model used (e.g., "MLP", "XGBoost").
        coords_for_moran (np.ndarray): Array of shape [n_samples, 2] for Moran's I calculation.
    """
    if true_surface is None or estimated_surface is None:
        print(
            f"Warning: Skipping metrics for {effect_name} ({encoder_name}/{model_name}) due to missing data."
        )
        return None
    if true_surface.shape != estimated_surface.shape:
        print(
            f"Warning: Skipping metrics for {effect_name} ({encoder_name}/{model_name}) due to shape mismatch: True {true_surface.shape}, Est {estimated_surface.shape}"
        )
        return None
    if coords_for_moran is None or coords_for_moran.shape[0] != true_surface.shape[0]:
        print(
            f"Warning: Skipping Moran's I for {effect_name} ({encoder_name}/{model_name}) due to missing or mismatched coords_for_moran. Coords shape: {coords_for_moran.shape if coords_for_moran is not None else 'None'}, Surface shape: {true_surface.shape}"
        )
        # Proceed with other metrics, but Moran's I will be NaN
        moran_calculated_successfully = False
    else:
        moran_calculated_successfully = True

    metrics = {
        "encoder": encoder_name,
        "model": model_name,
        "spatial_effect": effect_name,
    }
    try:
        mask = ~np.isnan(true_surface) & ~np.isnan(estimated_surface)
        if np.sum(mask) < 2:
            print(
                f"Warning: Not enough valid data points ({np.sum(mask)}) for metrics calculation for {effect_name} ({encoder_name}/{model_name})."
            )
            # Initialize all metrics as NaN
            metrics.update(
                {
                    m_key: np.nan
                    for m_key in [
                        "mse",
                        "rmse",
                        "mae",
                        "pearson_r",
                        "pearson_r_squared",
                        "r2_score",
                        "mean_error_bias",
                        "ols_slope",
                        "ols_intercept",
                        "ols_r2",
                        "amplitude_range_ratio",
                        "amplitude_std_ratio",
                        "rmse_normalized_by_range",
                        "rmse_normalized_by_std",
                        "mape_percent",
                        "moran_i_residuals",
                        "moran_i_residuals_p_value",
                    ]
                }
            )
            return metrics

        true_valid = true_surface[mask]
        est_valid = estimated_surface[mask]
        coords_valid = (
            coords_for_moran[mask]
            if moran_calculated_successfully and coords_for_moran is not None
            else None
        )

        # Calculate basic metrics
        metrics["mse"] = mean_squared_error(true_valid, est_valid)
        metrics["rmse"] = np.sqrt(metrics["mse"])
        metrics["mae"] = mean_absolute_error(true_valid, est_valid)

        if np.std(true_valid) > 1e-6 and np.std(est_valid) > 1e-6:
            pearson_r, _ = pearsonr(true_valid, est_valid)
            metrics["pearson_r"] = pearson_r
            metrics["pearson_r_squared"] = pearson_r**2
        else:
            metrics["pearson_r"] = np.nan
            metrics["pearson_r_squared"] = np.nan

        metrics["r2_score"] = r2_score(true_valid, est_valid)
        metrics["mean_error_bias"] = np.mean(est_valid - true_valid)
        
        # ========== ENHANCED METRICS FOR AMPLITUDE DIAGNOSIS ==========
        
        # 1. OLS Regression: est = intercept + slope * true
        # This tells us if the model recovers the right amplitude (slope ≈ 1 is good)
        try:
            if np.std(true_valid) > 1e-6:
                lr = LinearRegression()
                lr.fit(true_valid.reshape(-1, 1), est_valid)
                metrics["ols_slope"] = float(lr.coef_[0])
                metrics["ols_intercept"] = float(lr.intercept_)
                metrics["ols_r2"] = float(lr.score(true_valid.reshape(-1, 1), est_valid))
            else:
                metrics["ols_slope"] = np.nan
                metrics["ols_intercept"] = np.nan
                metrics["ols_r2"] = np.nan
        except Exception as e:
            print(f"  Warning: Could not compute OLS metrics for {effect_name}: {e}")
            metrics["ols_slope"] = np.nan
            metrics["ols_intercept"] = np.nan
            metrics["ols_r2"] = np.nan
        
        # 2. Amplitude Ratios: How much of the signal range/std is recovered?
        true_range = true_valid.max() - true_valid.min()
        est_range = est_valid.max() - est_valid.min()
        metrics["amplitude_range_ratio"] = est_range / true_range if true_range > 1e-6 else np.nan
        
        true_std = np.std(true_valid)
        est_std = np.std(est_valid)
        metrics["amplitude_std_ratio"] = est_std / true_std if true_std > 1e-6 else np.nan
        
        # 3. Normalized RMSE: Scale RMSE by true signal range/std
        metrics["rmse_normalized_by_range"] = metrics["rmse"] / true_range if true_range > 1e-6 else np.nan
        metrics["rmse_normalized_by_std"] = metrics["rmse"] / true_std if true_std > 1e-6 else np.nan
        
        # 4. Mean Absolute Percentage Error (if values not too close to zero)
        if np.all(np.abs(true_valid) > 0.1):  # Avoid division by small numbers
            mape = np.mean(np.abs((true_valid - est_valid) / true_valid)) * 100
            metrics["mape_percent"] = mape
        else:
            metrics["mape_percent"] = np.nan
        
        # ========== END ENHANCED METRICS ==========

        metrics["moran_i_residuals"] = np.nan
        metrics["moran_i_residuals_p_value"] = np.nan
        metrics["moran_i_true_surface"] = np.nan
        metrics["moran_i_true_surface_p_value"] = np.nan
        metrics["moran_i_estimated_surface"] = np.nan
        metrics["moran_i_estimated_surface_p_value"] = np.nan

        if (
            moran_calculated_successfully
            and coords_valid is not None
            and coords_valid.shape[0] > 1
        ):  # Need at least 2 points for weights
            residuals = est_valid - true_valid
            
            if (
                residuals.shape[0] == grid_size * grid_size
            ):  # Ensure it's for the full grid
                w = weights.lat2W(
                    nrows=grid_size, ncols=grid_size, rook=False
                )  # Use 'rook' for Rook contiguity
                w.transform = "r"  # Row-standardize
                
                # Moran's I on residuals (existing)
                if np.std(residuals) > 1e-9:  # Check if residuals are not constant
                    try:
                        moran_result = Moran(residuals, w, permutations=99)
                        metrics["moran_i_residuals"] = moran_result.I
                        metrics["moran_i_residuals_p_value"] = moran_result.p_sim
                    except Exception as e:
                        print(
                            f"Error calculating Moran's I on residuals for {effect_name} ({encoder_name}/{model_name}): {e}"
                        )
                else:
                    print(
                        f"Note: Residuals are constant for {effect_name} ({encoder_name}/{model_name}). Skipping Moran's I on residuals."
                    )
                
                # Moran's I on TRUE surface (NEW)
                if np.std(true_valid) > 1e-9:
                    try:
                        moran_true = Moran(true_valid, w, permutations=99)
                        metrics["moran_i_true_surface"] = moran_true.I
                        metrics["moran_i_true_surface_p_value"] = moran_true.p_sim
                    except Exception as e:
                        print(
                            f"Error calculating Moran's I on true surface for {effect_name} ({encoder_name}/{model_name}): {e}"
                        )
                
                # Moran's I on ESTIMATED surface (NEW)
                if np.std(est_valid) > 1e-9:
                    try:
                        moran_est = Moran(est_valid, w, permutations=99)
                        metrics["moran_i_estimated_surface"] = moran_est.I
                        metrics["moran_i_estimated_surface_p_value"] = moran_est.p_sim
                    except Exception as e:
                        print(
                            f"Error calculating Moran's I on estimated surface for {effect_name} ({encoder_name}/{model_name}): {e}"
                        )
            else:
                print(
                    f"Warning: Moran's I calculation skipped for {effect_name} ({encoder_name}/{model_name}) because the number of points ({residuals.shape[0]}) does not match the expected grid size ({grid_size*grid_size if grid_size else 'unknown'})."
                )

    except Exception as e:
        print(
            f"Error calculating metrics for {effect_name} ({encoder_name}/{model_name}): {e}"
        )
        # Ensure all metric keys exist
        for m_key in [
            "mse",
            "rmse",
            "mae",
            "pearson_r",
            "pearson_r_squared",
            "r2_score",
            "mean_error_bias",
            "ols_slope",
            "ols_intercept",
            "ols_r2",
            "amplitude_range_ratio",
            "amplitude_std_ratio",
            "rmse_normalized_by_range",
            "rmse_normalized_by_std",
            "mape_percent",
            "moran_i_residuals",
            "moran_i_residuals_p_value",
            "moran_i_true_surface",
            "moran_i_true_surface_p_value",
            "moran_i_estimated_surface",
            "moran_i_estimated_surface_p_value",
        ]:
            if m_key not in metrics:
                metrics[m_key] = np.nan
    return metrics


def interpret_metrics(metrics_dict, verbose=True):
    """
    Interpret spatial effect recovery metrics and provide diagnostic summary.
    
    Args:
        metrics_dict: Dictionary of metrics from calculate_spatial_metrics
        verbose: If True, prints interpretation
        
    Returns:
        dict with interpretation categories and scores
    """
    interpretation = {
        "effect_name": metrics_dict.get("spatial_effect", "Unknown"),
        "encoder": metrics_dict.get("encoder", "Unknown"),
        "model": metrics_dict.get("model", "Unknown"),
    }
    
    # Extract key metrics
    pearson_r = metrics_dict.get("pearson_r", np.nan)
    ols_slope = metrics_dict.get("ols_slope", np.nan)
    amplitude_ratio = metrics_dict.get("amplitude_range_ratio", np.nan)
    rmse_norm = metrics_dict.get("rmse_normalized_by_std", np.nan)
    
    # Shape recovery (correlation)
    if pearson_r >= 0.9:
        shape_quality = "Excellent"
        shape_score = 5
    elif pearson_r >= 0.7:
        shape_quality = "Good"
        shape_score = 4
    elif pearson_r >= 0.5:
        shape_quality = "Moderate"
        shape_score = 3
    elif pearson_r >= 0.3:
        shape_quality = "Weak"
        shape_score = 2
    else:
        shape_quality = "Poor/None"
        shape_score = 1
    
    interpretation["shape_quality"] = shape_quality
    interpretation["shape_score"] = shape_score
    
    # Amplitude recovery (OLS slope)
    if ols_slope >= 0.9:
        amplitude_quality = "Excellent"
        amplitude_score = 5
    elif ols_slope >= 0.7:
        amplitude_quality = "Good"
        amplitude_score = 4
    elif ols_slope >= 0.5:
        amplitude_quality = "Moderate"
        amplitude_score = 3
    elif ols_slope >= 0.3:
        amplitude_quality = "Weak"
        amplitude_score = 2
    else:
        amplitude_quality = "Poor/None"
        amplitude_score = 1
    
    interpretation["amplitude_quality"] = amplitude_quality
    interpretation["amplitude_score"] = amplitude_score
    
    # Overall recovery
    overall_score = (shape_score + amplitude_score) / 2
    if overall_score >= 4.5:
        overall_quality = "Excellent - Near-perfect recovery"
    elif overall_score >= 3.5:
        overall_quality = "Good - Useful for interpretation"
    elif overall_score >= 2.5:
        overall_quality = "Moderate - Pattern detected but weak"
    else:
        overall_quality = "Poor - Substantial issues"
    
    interpretation["overall_quality"] = overall_quality
    interpretation["overall_score"] = overall_score
    
    # Diagnosis
    diagnosis = []
    if pearson_r > 0.5 and ols_slope < 0.5:
        diagnosis.append("AMPLITUDE COMPRESSION: Shape captured but magnitude too small")
        diagnosis.append("  → Try: feature scaling, reduce regularization, increase model capacity")
    elif pearson_r < 0.3:
        diagnosis.append("SHAPE MISMATCH: Model not capturing spatial pattern")
        diagnosis.append("  → Try: check encoder extent, increase model capacity, more training data")
    elif ols_slope > 1.3:
        diagnosis.append("AMPLITUDE INFLATION: Magnitude too large")
        diagnosis.append("  → Try: increase regularization, check for overfitting")
    
    if rmse_norm > 0.5 and pearson_r > 0.7:
        diagnosis.append("HIGH NOISE: Good pattern but large errors")
        diagnosis.append("  → Model variance is high, consider ensemble or more data")
    
    if not diagnosis:
        if overall_score >= 4:
            diagnosis.append("✓ Good recovery - both shape and amplitude well captured")
        else:
            diagnosis.append("Multiple issues detected - check individual metrics")
    
    interpretation["diagnosis"] = diagnosis
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"METRIC INTERPRETATION: {interpretation['effect_name']}")
        print(f"Encoder: {interpretation['encoder']} | Model: {interpretation['model']}")
        print(f"{'='*70}")
        print(f"\n  Shape Recovery:     {shape_quality:15s} (r={pearson_r:.3f})")
        print(f"  Amplitude Recovery: {amplitude_quality:15s} (slope={ols_slope:.3f}, ratio={amplitude_ratio:.3f})")
        print(f"  Overall:            {overall_quality}")
        print(f"\n  Diagnosis:")
        for diag in diagnosis:
            print(f"    {diag}")
        print(f"{'='*70}\n")
    
    return interpretation


