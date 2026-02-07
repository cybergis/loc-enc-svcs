"""Tests for data generating process utilities."""

import numpy as np
import pandas as pd
import pytest

from dgp_utils import GridDGP, create_grid_data


class TestGridDGP:
    """Tests for the GridDGP class."""

    def test_default_grid_shape(self):
        dgp = GridDGP(size=10, coord_system='grid', random_seed=42)
        data, extent = dgp.generate_data()
        assert len(data) == 100  # 10x10

    def test_regional_coords_in_extent(self):
        dgp = GridDGP(size=5, coord_system='regional', random_seed=42)
        data, extent = dgp.generate_data()
        lon_min, lon_max, lat_min, lat_max = extent
        assert (data['lon'] >= lon_min).all()
        assert (data['lon'] <= lon_max).all()
        assert (data['lat'] >= lat_min).all()
        assert (data['lat'] <= lat_max).all()

    def test_grid_coords_nonneg(self):
        dgp = GridDGP(size=5, coord_system='grid', random_seed=42)
        data, _ = dgp.generate_data()
        assert (data['x_coord'] >= 0).all()
        assert (data['y_coord'] >= 0).all()

    def test_coefficients_in_expected_range(self):
        dgp = GridDGP(size=25, coord_system='grid', random_seed=42)
        coeffs = dgp.generate_mgwr_coefficients()
        # b0 clipped to [0.5, 5.5]
        assert coeffs['b0'].min() >= 0.5
        assert coeffs['b0'].max() <= 5.5
        # b1 in [1, 5]
        assert coeffs['b1'].min() >= 1.0 - 1e-6
        assert coeffs['b1'].max() <= 5.0 + 1e-6

    def test_noise_affects_output(self):
        data_noisy, _ = create_grid_data(size=5, noise_std=1.0, random_seed=42)
        data_clean, _ = create_grid_data(size=5, noise_std=0.0, random_seed=42)
        # With noise, y should differ from the noiseless version
        assert not np.allclose(data_noisy['y'].values, data_clean['y'].values)

    def test_seed_reproducibility(self):
        d1, _ = create_grid_data(size=5, random_seed=123)
        d2, _ = create_grid_data(size=5, random_seed=123)
        np.testing.assert_array_equal(d1['y'].values, d2['y'].values)

    def test_output_columns(self):
        data, _ = create_grid_data(size=5, coord_system='regional')
        expected = {'X1', 'X2', 'lon', 'lat', 'y', 'b0', 'b1', 'b2'}
        assert expected.issubset(set(data.columns))

    def test_invalid_coord_system(self):
        with pytest.raises(ValueError):
            GridDGP(size=5, coord_system='invalid')
