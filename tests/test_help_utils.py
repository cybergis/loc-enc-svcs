"""Tests for help_utils: metrics, embeddings, and interpretation."""

import numpy as np
import pytest

from help_utils import calculate_spatial_metrics, interpret_metrics, get_loc_embeddings


class TestCalculateSpatialMetrics:
    """Tests for the calculate_spatial_metrics function."""

    def _make_coords(self, n):
        """Create a simple grid of coordinates."""
        side = int(np.sqrt(n))
        assert side * side == n, "n must be a perfect square"
        x = np.linspace(0, 1, side)
        xx, yy = np.meshgrid(x, x)
        return np.column_stack([xx.ravel(), yy.ravel()])

    def test_perfect_recovery(self):
        """Identical true/estimated surfaces should give perfect scores."""
        n = 25
        true = np.random.RandomState(0).randn(n)
        coords = self._make_coords(n)
        m = calculate_spatial_metrics(true, true, "test", "enc", "mlp", coords, grid_size=5)
        assert m["pearson_r"] == pytest.approx(1.0, abs=1e-6)
        assert m["rmse"] == pytest.approx(0.0, abs=1e-6)
        assert m["ols_slope"] == pytest.approx(1.0, abs=1e-6)

    def test_scaled_surface(self):
        """Estimated = 2 * true should yield OLS slope ~2."""
        n = 25
        rng = np.random.RandomState(1)
        true = rng.randn(n) * 3 + 5
        est = 2.0 * true
        coords = self._make_coords(n)
        m = calculate_spatial_metrics(true, est, "test", "enc", "mlp", coords, grid_size=5)
        assert m["ols_slope"] == pytest.approx(2.0, abs=0.01)
        assert m["pearson_r"] == pytest.approx(1.0, abs=1e-6)

    def test_returns_none_on_missing_data(self):
        m = calculate_spatial_metrics(None, np.ones(10), "test", "enc", "mlp", None)
        assert m is None

    def test_shape_mismatch_returns_none(self):
        m = calculate_spatial_metrics(np.ones(10), np.ones(5), "test", "enc", "mlp", None)
        assert m is None

    def test_all_expected_keys(self):
        n = 25
        true = np.random.RandomState(2).randn(n)
        est = true + 0.1 * np.random.RandomState(3).randn(n)
        coords = self._make_coords(n)
        m = calculate_spatial_metrics(true, est, "test", "enc", "mlp", coords, grid_size=5)
        expected_keys = [
            "pearson_r", "ols_slope", "rmse", "mae", "mse",
            "amplitude_range_ratio", "amplitude_std_ratio",
            "moran_i_residuals", "encoder", "model", "spatial_effect",
        ]
        for key in expected_keys:
            assert key in m, f"Missing key: {key}"

    def test_bias_metric(self):
        n = 25
        true = np.ones(n) * 3.0
        est = np.ones(n) * 5.0
        coords = self._make_coords(n)
        m = calculate_spatial_metrics(true, est, "test", "enc", "mlp", coords, grid_size=5)
        assert m["mean_error_bias"] == pytest.approx(2.0, abs=1e-6)


class TestInterpretMetrics:
    """Tests for the interpret_metrics function."""

    def test_excellent_recovery(self):
        metrics = {
            "spatial_effect": "b1",
            "encoder": "NeRF",
            "model": "MLP",
            "pearson_r": 0.95,
            "ols_slope": 0.95,
            "amplitude_range_ratio": 1.0,
            "rmse_normalized_by_std": 0.1,
        }
        result = interpret_metrics(metrics, verbose=False)
        assert result["shape_quality"] == "Excellent"
        assert result["amplitude_quality"] == "Excellent"
        assert result["overall_score"] >= 4.5

    def test_amplitude_compression_diagnosis(self):
        metrics = {
            "spatial_effect": "b1",
            "encoder": "rff",
            "model": "MLP",
            "pearson_r": 0.8,
            "ols_slope": 0.2,
            "amplitude_range_ratio": 0.3,
            "rmse_normalized_by_std": 0.5,
        }
        result = interpret_metrics(metrics, verbose=False)
        diag_text = " ".join(result["diagnosis"])
        assert "AMPLITUDE COMPRESSION" in diag_text

    def test_poor_shape(self):
        metrics = {
            "spatial_effect": "b2",
            "encoder": "none",
            "model": "XGBoost",
            "pearson_r": 0.1,
            "ols_slope": 0.1,
            "amplitude_range_ratio": 0.1,
            "rmse_normalized_by_std": 2.0,
        }
        result = interpret_metrics(metrics, verbose=False)
        assert result["shape_score"] == 1


class TestGetLocEmbeddings:
    """Tests for the get_loc_embeddings function."""

    def test_output_shape(self):
        coords = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
        extent = (0, 1, 0, 1)
        emb = get_loc_embeddings(coords, encoder_type="NeRF", extent=extent, device="cpu")
        assert emb.shape[0] == 3
        assert emb.ndim == 2

    def test_different_encoders_differ(self):
        coords = np.array([[0.0, 0.0], [1.0, 1.0]])
        extent = (0, 1, 0, 1)
        e1 = get_loc_embeddings(coords, encoder_type="NeRF", extent=extent, device="cpu")
        e2 = get_loc_embeddings(coords, encoder_type="rff", extent=extent, device="cpu")
        # Different encoders should produce different embeddings
        assert not np.allclose(e1.detach().numpy(), e2.detach().numpy())

    def test_warns_without_extent(self):
        coords = np.array([[0.0, 0.0], [1.0, 1.0]])
        with pytest.warns(UserWarning, match="No extent provided"):
            get_loc_embeddings(coords, encoder_type="NeRF", device="cpu")
