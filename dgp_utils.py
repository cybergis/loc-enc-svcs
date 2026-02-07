"""
Data Generating Process (DGP) utilities for spatial experiments.
Supports grid and county-level data with MGWR-style spatially-varying coefficients.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Tuple, Dict, Optional
from pathlib import Path


class GridDGP:
    """Generate MGWR-style synthetic data on regular grids."""
    
    def __init__(self, size: int = 25, coord_system: str = 'regional', 
                 center_coords: Tuple[float, float] = (-87.65, 41.85),
                 km_span: float = 100, random_seed: int = 222):
        """
        Args:
            size: Grid dimension (size × size points)
            coord_system: 'grid' (centered at 0) or 'regional' (geographic coords)
            center_coords: (lon, lat) center for regional coords
            km_span: Geographic span in km for regional coords
            random_seed: Random seed for reproducibility
        """
        self.size = size
        self.coord_system = coord_system
        self.center_coords = center_coords
        self.km_span = km_span
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        self.coords, self.extent = self._generate_coords()
        
    def _generate_coords(self) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """Generate coordinate grid."""
        u = np.linspace(0, self.size-1, self.size)
        uu, vv = np.meshgrid(u, u)
        
        if self.coord_system == 'grid':
            coords = np.column_stack([uu.ravel(), vv.ravel()])
            # IMPORTANT: Keep raw coordinates [0, size-1], don't center at 0
            # Centered coordinates reduce numeric range and hurt amplitude recovery
            extent = (coords[:, 0].min(), coords[:, 0].max(),
                     coords[:, 1].min(), coords[:, 1].max())
            return coords, extent
        
        elif self.coord_system == 'regional':
            center_lon, center_lat = self.center_coords
            lat_span = self.km_span / 111.0
            lon_span = self.km_span / (111.0 * np.cos(np.radians(center_lat)))
            
            lons = np.linspace(center_lon - lon_span/2, center_lon + lon_span/2, self.size)
            lats = np.linspace(center_lat - lat_span/2, center_lat + lat_span/2, self.size)
            lon_grid, lat_grid = np.meshgrid(lons, lats)
            
            coords = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
            extent = (lons[0], lons[-1], lats[0], lats[-1])
            return coords, extent
        
        else:
            raise ValueError(f"Unknown coord_system: {self.coord_system}")
    
    def generate_mgwr_coefficients(self) -> Dict[str, np.ndarray]:
        """Generate MGWR-style spatially-varying coefficients with stronger variation."""
        if self.coord_system == 'grid':
            coords_2d = self.coords.reshape(self.size, self.size, 2)
            u = coords_2d[:, :, 0]  # Range [0, size-1]
            v = coords_2d[:, :, 1]  # Range [0, size-1]
            
            # Normalize to [0, 1] for coefficient generation
            u_norm = u / (self.size - 1)
            v_norm = v / (self.size - 1)
            
            # b0: Parabolic surface - intercept varies spatially [0.5, 5.5]
            b0 = 3.0 + 2.5 * ((u_norm - 0.5)**2 + (v_norm - 0.5)**2)
            b0 = np.clip(b0, 0.5, 5.5)
            
            # b1: Strong linear gradient [1, 5] 
            # Increases left-to-right and bottom-to-top
            b1 = 1.0 + 4.0 * (u_norm + v_norm) / 2.0
            
            # b2: Complex surface [0.5, 5.5] - gradient + oscillation
            # This creates fine-grained spatial variation GeoShapley can detect
            b2 = 1.5 + 3.0 * ((1.0 - u_norm + v_norm) / 2.0) + \
                 0.8 * np.sin(4 * np.pi * u_norm) * np.cos(4 * np.pi * v_norm)
            
            return {'b0': b0.ravel(), 'b1': b1.ravel(), 'b2': b2.ravel()}
        
        else:  # regional
            lon, lat = self.coords[:, 0], self.coords[:, 1]
            lon_min, lon_max, lat_min, lat_max = self.extent
            
            u = (lon - lon_min) / (lon_max - lon_min)
            v = (lat - lat_min) / (lat_max - lat_min)
            
            # Parabolic surface (properly scaled for regional coordinates)
            b0 = 1.5 + 3.5 * (1 - ((u - 0.5)**2 + (v - 0.5)**2) / 0.5)
            b0 = np.clip(b0, 0.5, 5.0)
            
            # East-West gradient
            b1 = 1 + 4 * (u + v) / 2
            
            # More expressive pattern for b2: sinusoidal + gradient
            b2 = 1.5 + 3 * ((1 - u + v) / 2) + 0.5 * np.sin(4 * np.pi * u) * np.cos(4 * np.pi * v)
            
            return {'b0': b0, 'b1': b1, 'b2': b2}
    
    def generate_data(self, noise_std: float = 0.1) -> Tuple[pd.DataFrame, Tuple[float, float, float, float]]:
        """Generate complete dataset."""
        np.random.seed(self.random_seed)
        
        n_points = self.size * self.size
        X1 = np.random.uniform(-2, 2, n_points)
        X2 = np.random.uniform(-2, 2, n_points)
        
        coeffs = self.generate_mgwr_coefficients()
        y = coeffs['b0'] + coeffs['b1'] * X1 + coeffs['b2'] * X2
        
        if noise_std > 0:
            y += np.random.normal(0, noise_std, n_points)
        
        coord_names = ['x_coord', 'y_coord'] if self.coord_system == 'grid' else ['lon', 'lat']
        
        df = pd.DataFrame({
            'X1': X1,
            'X2': X2,
            coord_names[0]: self.coords[:, 0],
            coord_names[1]: self.coords[:, 1],
            'y': y,
            'b0': coeffs['b0'],
            'b1': coeffs['b1'],
            'b2': coeffs['b2']
        })
        
        return df, self.extent


class CountyDGP:
    """Generate synthetic data for US counties."""
    
    def __init__(self, shapefile_path: Optional[str] = None, random_seed: int = 222):
        """
        Args:
            shapefile_path: Path to counties shapefile (downloads if None)
            random_seed: Random seed
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.counties_gdf = self._load_counties(shapefile_path)
        self.n_counties = len(self.counties_gdf)
        
    def _load_counties(self, shapefile_path: Optional[str]) -> gpd.GeoDataFrame:
        """Load or create county geometries."""
        if shapefile_path and Path(shapefile_path).exists():
            gdf = gpd.read_file(shapefile_path)
        else:
            try:
                url = "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_county_500k.zip"
                gdf = gpd.read_file(url)
            except:
                return self._create_synthetic_counties()
        
        # Filter continental US
        continental = [
            '01', '04', '05', '06', '08', '09', '10', '12', '13', '16', '17', '18', '19', '20',
            '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34',
            '35', '36', '37', '38', '39', '40', '41', '42', '44', '45', '46', '47', '48', '49',
            '50', '51', '53', '54', '55', '56'
        ]
        gdf = gdf[gdf['STATEFP'].isin(continental)].copy()
        gdf['centroid'] = gdf.geometry.centroid
        gdf['lon'] = gdf.centroid.x
        gdf['lat'] = gdf.centroid.y
        
        return gdf
    
    def _create_synthetic_counties(self, n_counties: int = 3000) -> gpd.GeoDataFrame:
        """Create synthetic county grid as fallback."""
        lons = np.random.uniform(-125, -66, n_counties)
        lats = np.random.uniform(24, 50, n_counties)
        
        return gpd.GeoDataFrame({
            'GEOID': [f'SYN{i:05d}' for i in range(n_counties)],
            'NAME': [f'County_{i}' for i in range(n_counties)],
            'lon': lons,
            'lat': lats
        }, geometry=gpd.points_from_xy(lons, lats))
    
    def generate_mgwr_coefficients(self) -> Dict[str, np.ndarray]:
        """Generate MGWR-style coefficients for counties."""
        lons = self.counties_gdf['lon'].values
        lats = self.counties_gdf['lat'].values
        
        lon_norm = (lons - lons.min()) / (lons.max() - lons.min())
        lat_norm = (lats - lats.min()) / (lats.max() - lats.min())
        
        # Parabolic surface (properly scaled for county coordinates)
        b0 = 1.5 + 3.5 * (1 - ((lon_norm - 0.5)**2 + (lat_norm - 0.5)**2) / 0.5)
        b0 = np.clip(b0, 0.5, 5.0)
        
        # East-West gradient
        b1 = 1 + 4 * (lon_norm + lat_norm) / 2
        
        # More expressive pattern for b2: sinusoidal + gradient
        b2 = 1.5 + 3 * ((1 - lon_norm + lat_norm) / 2) + 0.5 * np.sin(4 * np.pi * lon_norm) * np.cos(4 * np.pi * lat_norm)
        
        return {'b0': b0, 'b1': b1, 'b2': b2}
    
    def generate_data(self, noise_std: float = 0.1) -> Tuple[pd.DataFrame, Tuple[float, float, float, float], gpd.GeoDataFrame]:
        """Generate complete dataset."""
        np.random.seed(self.random_seed)
        
        X1 = np.random.uniform(-2, 2, self.n_counties)
        X2 = np.random.uniform(-2, 2, self.n_counties)
        
        coeffs = self.generate_mgwr_coefficients()
        y = coeffs['b0'] + coeffs['b1'] * X1 + coeffs['b2'] * X2
        
        if noise_std > 0:
            y += np.random.normal(0, noise_std, self.n_counties)
        
        df = pd.DataFrame({
            'X1': X1,
            'X2': X2,
            'lon': self.counties_gdf['lon'].values,
            'lat': self.counties_gdf['lat'].values,
            'y': y,
            'b0': coeffs['b0'],
            'b1': coeffs['b1'],
            'b2': coeffs['b2'],
            'GEOID': self.counties_gdf['GEOID'].values,
            'NAME': self.counties_gdf['NAME'].values
        })
        
        extent = (df['lon'].min(), df['lon'].max(), df['lat'].min(), df['lat'].max())
        
        return df, extent, self.counties_gdf


# Convenience functions
def create_grid_data(size: int = 25, coord_system: str = 'regional', 
                    noise_std: float = 0.1, random_seed: int = 222) -> Tuple[pd.DataFrame, Tuple]:
    """Quick grid data generation."""
    dgp = GridDGP(size=size, coord_system=coord_system, random_seed=random_seed)
    return dgp.generate_data(noise_std=noise_std)


def create_county_data(shapefile_path: Optional[str] = None, noise_std: float = 0.1,
                       random_seed: int = 222) -> Tuple[pd.DataFrame, Tuple, gpd.GeoDataFrame]:
    """Quick county data generation."""
    dgp = CountyDGP(shapefile_path=shapefile_path, random_seed=random_seed)
    return dgp.generate_data(noise_std=noise_std)
