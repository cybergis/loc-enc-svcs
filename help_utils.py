"""
Utility functions for location encoding, spatial metrics, and visualization.

Handles TorchSpatial encoder integration and provides metric calculations
for comparing true vs. estimated spatially-varying coefficients.
"""

import os
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from libpysal import weights
from esda.moran import Moran

# Add TorchSpatial/main to path so its modules are importable
_HERE = os.path.dirname(os.path.abspath(__file__))
_TS_DIR = os.path.join(_HERE, "TorchSpatial", "main")
if _TS_DIR not in sys.path:
    sys.path.insert(0, _TS_DIR)

from SpatialRelationEncoder import *  # noqa: E402
from module import *  # noqa: E402
from data_utils import *  # noqa: E402
from utils import *  # noqa: E402

SPA_EMBED_DIM = 4  # Default embedding dimension; overridable via get_loc_embeddings/train_loc_encoder

# Encoders that normalize coordinates internally or convert to 3D — max_radius=1 is correct
_EXTENT_NORMALIZED_ENCODERS = {'tile_ffn', 'wrap_ffn', 'rff', 'NeRF'}


def _get_max_radius(encoder_type, extent):
    """Encoders using raw coordinates need max_radius matching the data span."""
    if encoder_type in _EXTENT_NORMALIZED_ENCODERS or encoder_type is None:
        return 1
    # Space2Vec and Sphere2Vec variants use raw coords with frequency scaling
    x_span = extent[1] - extent[0]
    y_span = extent[3] - extent[2]
    return float(max(x_span, y_span))


def get_loc_embeddings(coords, encoder_type, extent=None, device="cpu",
                       embed_dim=None, f_act="silu"):
    """
    Compute location embeddings for 2D coordinates using the specified spatial encoder type.

    Parameters:
        coords (np.ndarray): Array of shape [batch_size, 2] containing the coordinates.
        encoder_type (str): The string identifier for the spatial encoder.
        extent (tuple): The spatial extent as (x_min, x_max, y_min, y_max).
        device (str): Device to use for the computation.
        embed_dim (int): Embedding dimension. Defaults to SPA_EMBED_DIM.

    Returns:
        torch.Tensor: The location embeddings, shape [batch_size, embed_dim].
    """
    dim = embed_dim if embed_dim is not None else SPA_EMBED_DIM

    if extent is None:
        import warnings
        warnings.warn(
            "No extent provided to get_loc_embeddings(). "
            "Using default (0, 200, 0, 200) which may not match your data.",
            stacklevel=2
        )
        extent = (0, 200, 0, 200)

    params = {
        "spa_enc_type": encoder_type,
        "spa_embed_dim": dim,
        "extent": extent,
        "freq": 16,
        "max_radius": _get_max_radius(encoder_type, extent),
        "min_radius": 0.0001,
        "spa_f_act": f_act,
        "freq_init": "geometric",
        "num_hidden_layer": 1,
        "dropout": 0.5,
        "hidden_dim": 64,
        "use_layn": True,
        "skip_connection": True,
        "spa_enc_use_postmat": True,
        "device": device,
    }

    loc_enc = get_spa_encoder(
        train_locs=[],
        params=params,
        spa_enc_type=params["spa_enc_type"],
        spa_embed_dim=params["spa_embed_dim"],
        extent=params["extent"],
        coord_dim=2,
        frequency_num=params["freq"],
        max_radius=params["max_radius"],
        min_radius=params["min_radius"],
        f_act=params["spa_f_act"],
        freq_init=params["freq_init"],
        use_postmat=params["spa_enc_use_postmat"],
        device=params["device"],
    ).to(params["device"])

    loc_enc.eval()  # Disable dropout for deterministic embeddings

    coords = np.array(coords)
    if coords.ndim == 2:
        coords = np.expand_dims(coords, axis=1)

    with torch.no_grad():
        loc_embeds = torch.squeeze(loc_enc(coords))
    return loc_embeds


def _build_encoder(encoder_type, extent, dim, f_act, device):
    """Build a TorchSpatial location encoder with standard parameters."""
    import torch.nn as nn

    params = {
        "spa_enc_type": encoder_type,
        "spa_embed_dim": dim,
        "extent": extent,
        "freq": 16,
        "max_radius": _get_max_radius(encoder_type, extent),
        "min_radius": 0.0001,
        "spa_f_act": f_act,
        "freq_init": "geometric",
        "num_hidden_layer": 1,
        "dropout": 0.5,
        "hidden_dim": 64,
        "use_layn": True,
        "skip_connection": True,
        "spa_enc_use_postmat": True,
        "device": device,
    }

    return get_spa_encoder(
        train_locs=[],
        params=params,
        spa_enc_type=params["spa_enc_type"],
        spa_embed_dim=params["spa_embed_dim"],
        extent=params["extent"],
        coord_dim=2,
        frequency_num=params["freq"],
        max_radius=params["max_radius"],
        min_radius=params["min_radius"],
        f_act=params["spa_f_act"],
        freq_init=params["freq_init"],
        use_postmat=params["spa_enc_use_postmat"],
        device=params["device"],
    ).to(device)


def train_loc_encoder(coords, X1, X2, y, encoder_type, extent,
                      device="cpu", n_epochs=500, lr=1e-3, random_seed=None,
                      embed_dim=None, f_act="silu"):
    """
    Train a location encoder using spatial contrastive learning.

    Nearby coordinates get similar embeddings, distant ones get pushed apart.
    This preserves spatial structure without overfitting to y.

    Each epoch: sample anchor points, create positive pairs (nearby) and
    negative pairs (distant), optimize InfoNCE contrastive loss.

    Args:
        coords:       np.ndarray [N, 2], (lon, lat) or (x, y) — TRAIN SET ONLY
        X1, X2:      np.ndarray [N], covariates — not used (API compat)
        y:            np.ndarray [N], response — not used (self-supervised)
        encoder_type: TorchSpatial encoder string
        extent:       (xmin, xmax, ymin, ymax)
        device:       'cpu' or 'cuda:0'
        n_epochs:     gradient steps (default 500)
        lr:           initial Adam learning rate (default 1e-3)
        random_seed:  torch manual seed for reproducibility
        embed_dim:    embedding dimension. Defaults to SPA_EMBED_DIM.

    Returns:
        loc_enc: trained nn.Module in eval() mode
    """
    import torch.nn as nn
    import torch.optim as optim

    dim = embed_dim if embed_dim is not None else SPA_EMBED_DIM

    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    loc_enc = _build_encoder(encoder_type, extent, dim, f_act, device)

    optimizer = optim.Adam(loc_enc.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    # Precompute pairwise distances for positive/negative sampling
    coords_arr = np.array(coords)
    lon_min, lon_max, lat_min, lat_max = extent
    coords_norm = np.column_stack([
        (coords_arr[:, 0] - lon_min) / (lon_max - lon_min),
        (coords_arr[:, 1] - lat_min) / (lat_max - lat_min),
    ])

    # Positive threshold: 10th percentile of pairwise distances
    # Negative threshold: 50th percentile
    from scipy.spatial.distance import pdist, squareform
    N = len(coords_norm)
    if N > 1000:
        # Subsample for distance computation
        idx_sub = np.random.choice(N, 1000, replace=False)
        dists_sub = pdist(coords_norm[idx_sub])
        pos_thresh = np.percentile(dists_sub, 10)
        neg_thresh = np.percentile(dists_sub, 50)
    else:
        dists_all = pdist(coords_norm)
        pos_thresh = np.percentile(dists_all, 10)
        neg_thresh = np.percentile(dists_all, 50)

    coords_enc = np.expand_dims(coords_arr, axis=1)  # [N, 1, 2]
    batch_size = min(256, N)
    temperature = 0.1

    loc_enc.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Sample anchor batch
        anchor_idx = np.random.choice(N, batch_size, replace=False)
        anchor_coords = coords_norm[anchor_idx]

        # For each anchor, find a positive (nearby) and negative (distant)
        pos_idx = np.zeros(batch_size, dtype=int)
        neg_idx = np.zeros(batch_size, dtype=int)
        for i, ai in enumerate(anchor_idx):
            dists_i = np.linalg.norm(coords_norm - anchor_coords[i], axis=1)
            # Positive: random nearby point (exclude self)
            pos_mask = (dists_i < pos_thresh) & (dists_i > 0)
            if pos_mask.sum() > 0:
                pos_idx[i] = np.random.choice(np.where(pos_mask)[0])
            else:
                pos_idx[i] = np.argpartition(dists_i, 2)[1]  # nearest neighbor
            # Negative: random distant point
            neg_mask = dists_i > neg_thresh
            if neg_mask.sum() > 0:
                neg_idx[i] = np.random.choice(np.where(neg_mask)[0])
            else:
                neg_idx[i] = np.argmax(dists_i)

        # Get embeddings
        all_idx = np.concatenate([anchor_idx, pos_idx, neg_idx])
        emb_all = loc_enc(coords_enc[all_idx])
        emb_all = torch.squeeze(emb_all)
        if emb_all.ndim == 1:
            emb_all = emb_all.unsqueeze(0)

        emb_anchor = emb_all[:batch_size]
        emb_pos = emb_all[batch_size:2*batch_size]
        emb_neg = emb_all[2*batch_size:]

        # Normalize embeddings for cosine similarity
        emb_anchor = nn.functional.normalize(emb_anchor, dim=1)
        emb_pos = nn.functional.normalize(emb_pos, dim=1)
        emb_neg = nn.functional.normalize(emb_neg, dim=1)

        # InfoNCE loss: push anchors toward positives, away from negatives
        pos_sim = (emb_anchor * emb_pos).sum(dim=1) / temperature
        neg_sim = (emb_anchor * emb_neg).sum(dim=1) / temperature
        logits = torch.stack([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        loss = nn.functional.cross_entropy(logits, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

    loc_enc.eval()
    return loc_enc


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


# --- Spatial Metrics ---
def calculate_spatial_metrics(
    true_surface,
    estimated_surface,
    effect_name,
    encoder_name,
    model_name,
    coords_for_moran,
    grid_size=None,
):
    """
    Compare true and estimated spatial surfaces with multiple metrics.

    Returns a dict with error metrics (MSE, RMSE, MAE), correlation (Pearson r),
    amplitude diagnostics (OLS slope, range/std ratios), and spatial
    autocorrelation (Moran's I) on residuals and surfaces.
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
        
        # OLS regression: est = intercept + slope * true
        # slope ~ 1 means correct amplitude recovery
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
        
        # Amplitude ratios
        true_range = true_valid.max() - true_valid.min()
        est_range = est_valid.max() - est_valid.min()
        metrics["amplitude_range_ratio"] = est_range / true_range if true_range > 1e-6 else np.nan
        
        true_std = np.std(true_valid)
        est_std = np.std(est_valid)
        metrics["amplitude_std_ratio"] = est_std / true_std if true_std > 1e-6 else np.nan
        
        # Normalized RMSE
        metrics["rmse_normalized_by_range"] = metrics["rmse"] / true_range if true_range > 1e-6 else np.nan
        metrics["rmse_normalized_by_std"] = metrics["rmse"] / true_std if true_std > 1e-6 else np.nan
        
        # MAPE (only when values aren't near zero)
        if np.all(np.abs(true_valid) > 0.1):
            mape = np.mean(np.abs((true_valid - est_valid) / true_valid)) * 100
            metrics["mape_percent"] = mape
        else:
            metrics["mape_percent"] = np.nan

        # Moran's I on residuals and surfaces

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
                grid_size is not None and residuals.shape[0] == grid_size * grid_size
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
                # grid_size is None (e.g. counties) or points don't match grid — use KNN weights
                try:
                    w_knn = weights.KNN.from_array(coords_valid, k=min(8, coords_valid.shape[0] - 1))
                    w_knn.transform = "r"

                    if np.std(residuals) > 1e-9:
                        moran_result = Moran(residuals, w_knn, permutations=99)
                        metrics["moran_i_residuals"] = moran_result.I
                        metrics["moran_i_residuals_p_value"] = moran_result.p_sim

                    if np.std(true_valid) > 1e-9:
                        moran_true = Moran(true_valid, w_knn, permutations=99)
                        metrics["moran_i_true_surface"] = moran_true.I
                        metrics["moran_i_true_surface_p_value"] = moran_true.p_sim

                    if np.std(est_valid) > 1e-9:
                        moran_est = Moran(est_valid, w_knn, permutations=99)
                        metrics["moran_i_estimated_surface"] = moran_est.I
                        metrics["moran_i_estimated_surface_p_value"] = moran_est.p_sim
                except Exception as e:
                    print(
                        f"Warning: KNN-based Moran's I failed for {effect_name} "
                        f"({encoder_name}/{model_name}): {e}"
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
            diagnosis.append("Good recovery - both shape and amplitude well captured")
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


