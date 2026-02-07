"""
Visualization Script for US Counties Location Encoder Experiments
Creates choropleth maps comparing true vs estimated spatially-varying coefficients
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from pathlib import Path
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')


class CountyVisualization:
    """Create publication-quality choropleth maps for county-level experiments."""
    
    def __init__(self, results_dir: str, counties_gdf: gpd.GeoDataFrame):
        """
        Initialize visualization.
        
        Args:
            results_dir: Directory with experiment results
            counties_gdf: GeoDataFrame with county geometries
        """
        self.results_dir = Path(results_dir)
        self.counties_gdf = counties_gdf
        
    def load_results(self, encoder_name: str, model_type: str = 'MLP') -> pd.DataFrame:
        """
        Load aggregated results for an encoder.
        
        Args:
            encoder_name: Name of location encoder
            model_type: 'MLP' or 'XGBoost'
            
        Returns:
            DataFrame with mean estimates and std across repetitions
        """
        # Pattern: {encoder}_{model}_rep{N}_spatial_effects.csv
        rep_files = list(self.results_dir.glob(f"{encoder_name}_{model_type}_rep*_spatial_effects.csv"))
        
        if not rep_files:
            raise FileNotFoundError(f"No results found for {encoder_name} {model_type}")
        
        # Load all repetitions
        all_reps = []
        for f in rep_files:
            df = pd.read_csv(f)
            all_reps.append(df)
        
        # Compute mean and std
        concat = pd.concat(all_reps)
        summary = concat.groupby('GEOID').agg({
            'b0_estimated': ['mean', 'std'],
            'b1_estimated': ['mean', 'std'],
            'b2_estimated': ['mean', 'std'],
            'b0_true': 'first',  # True values same across reps
            'b1_true': 'first',
            'b2_true': 'first'
        })
        
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary = summary.reset_index()
        
        print(f"✓ Loaded {len(rep_files)} repetitions for {encoder_name} {model_type}")
        return summary
    
    def create_comparison_map(
        self,
        encoder_name: str,
        coefficient: str = 'b1',
        model_type: str = 'MLP',
        output_path: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        figsize: Tuple[int, int] = (18, 6)
    ):
        """
        Create 3-panel comparison: True | Mean Estimate | Std Dev
        
        Args:
            encoder_name: Name of encoder
            coefficient: 'b0', 'b1', or 'b2'
            model_type: 'MLP' or 'XGBoost'
            output_path: Where to save figure
            vmin, vmax: Color scale limits (auto if None)
            figsize: Figure size
        """
        # Load results
        results = self.load_results(encoder_name, model_type)
        
        # Merge with geometries
        plot_data = self.counties_gdf.merge(results, on='GEOID', how='left')
        
        # Column names
        true_col = f'{coefficient}_true_first'
        mean_col = f'{coefficient}_estimated_mean'
        std_col = f'{coefficient}_estimated_std'
        
        # Auto range if not specified
        if vmin is None or vmax is None:
            vmin = plot_data[true_col].min()
            vmax = plot_data[true_col].max()
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Panel 1: True coefficients
        plot_data.plot(
            column=true_col,
            ax=axes[0],
            cmap='RdYlBu_r',
            vmin=vmin,
            vmax=vmax,
            legend=True,
            legend_kwds={'label': f'True {coefficient}', 'shrink': 0.7}
        )
        axes[0].set_title(f'True {coefficient}', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Panel 2: Mean estimate
        plot_data.plot(
            column=mean_col,
            ax=axes[1],
            cmap='RdYlBu_r',
            vmin=vmin,
            vmax=vmax,
            legend=True,
            legend_kwds={'label': f'Estimated {coefficient} (mean)', 'shrink': 0.7}
        )
        axes[1].set_title(f'Mean Estimate {coefficient}', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Panel 3: Std dev
        plot_data.plot(
            column=std_col,
            ax=axes[2],
            cmap='Reds',
            legend=True,
            legend_kwds={'label': f'Std Dev', 'shrink': 0.7}
        )
        axes[2].set_title(f'Uncertainty (Std Dev)', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        # Overall title
        fig.suptitle(
            f'{encoder_name} - {model_type} - Coefficient {coefficient}',
            fontsize=16,
            fontweight='bold',
            y=0.95
        )
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved map to: {output_path}")
        
        return fig, axes
    
    def create_multi_encoder_comparison(
        self,
        encoders: List[str],
        coefficient: str = 'b1',
        model_type: str = 'MLP',
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (20, 12)
    ):
        """
        Compare multiple encoders side-by-side.
        
        Args:
            encoders: List of encoder names
            coefficient: 'b0', 'b1', or 'b2'
            model_type: 'MLP' or 'XGBoost'
            output_path: Where to save figure
            figsize: Figure size
        """
        n_encoders = len(encoders)
        fig, axes = plt.subplots(n_encoders, 3, figsize=figsize)
        
        if n_encoders == 1:
            axes = axes.reshape(1, -1)
        
        # Load true values from first encoder
        results_first = self.load_results(encoders[0], model_type)
        plot_data_first = self.counties_gdf.merge(results_first, on='GEOID', how='left')
        true_col = f'{coefficient}_true_first'
        vmin = plot_data_first[true_col].min()
        vmax = plot_data_first[true_col].max()
        
        for i, encoder in enumerate(encoders):
            # Load results
            results = self.load_results(encoder, model_type)
            plot_data = self.counties_gdf.merge(results, on='GEOID', how='left')
            
            mean_col = f'{coefficient}_estimated_mean'
            std_col = f'{coefficient}_estimated_std'
            
            # True (col 0)
            plot_data.plot(
                column=true_col,
                ax=axes[i, 0],
                cmap='RdYlBu_r',
                vmin=vmin,
                vmax=vmax,
                legend=(i == 0),
                legend_kwds={'label': f'True {coefficient}', 'shrink': 0.5}
            )
            if i == 0:
                axes[i, 0].set_title('True', fontsize=12, fontweight='bold')
            axes[i, 0].axis('off')
            axes[i, 0].text(-0.15, 0.5, encoder, 
                           transform=axes[i, 0].transAxes,
                           fontsize=11, fontweight='bold',
                           va='center', ha='right', rotation=90)
            
            # Mean estimate (col 1)
            plot_data.plot(
                column=mean_col,
                ax=axes[i, 1],
                cmap='RdYlBu_r',
                vmin=vmin,
                vmax=vmax,
                legend=(i == 0),
                legend_kwds={'label': f'Estimate', 'shrink': 0.5}
            )
            if i == 0:
                axes[i, 1].set_title('Mean Estimate', fontsize=12, fontweight='bold')
            axes[i, 1].axis('off')
            
            # Std dev (col 2)
            plot_data.plot(
                column=std_col,
                ax=axes[i, 2],
                cmap='Reds',
                legend=(i == 0),
                legend_kwds={'label': 'Std Dev', 'shrink': 0.5}
            )
            if i == 0:
                axes[i, 2].set_title('Uncertainty', fontsize=12, fontweight='bold')
            axes[i, 2].axis('off')
        
        fig.suptitle(
            f'Encoder Comparison - {model_type} - Coefficient {coefficient}',
            fontsize=16,
            fontweight='bold',
            y=0.98
        )
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved comparison to: {output_path}")
        
        return fig, axes


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize county-level results')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory with experiment results')
    parser.add_argument('--shapefile', type=str, default=None,
                       help='Path to counties shapefile (will download if None)')
    parser.add_argument('--encoders', nargs='+', 
                       default=['space2vec_rbf', 'tile_ffn', 'wrap_ffn'],
                       help='Encoder names to visualize')
    parser.add_argument('--coefficient', type=str, default='b1',
                       choices=['b0', 'b1', 'b2'],
                       help='Which coefficient to visualize')
    parser.add_argument('--model', type=str, default='MLP',
                       choices=['MLP', 'XGBoost'],
                       help='Model type')
    parser.add_argument('--output_dir', type=str, default='./county_figs',
                       help='Output directory for figures')
    
    args = parser.parse_args()
    
    # Load counties
    print("Loading US counties...")
    if args.shapefile:
        counties = gpd.read_file(args.shapefile)
    else:
        # Try to download
        url = "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_county_500k.zip"
        counties = gpd.read_file(url)
        # Filter continental US
        continental = [
            '01', '04', '05', '06', '08', '09', '10', '12', '13', '16', '17', '18', '19', '20',
            '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34',
            '35', '36', '37', '38', '39', '40', '41', '42', '44', '45', '46', '47', '48', '49',
            '50', '51', '53', '54', '55', '56'
        ]
        counties = counties[counties['STATEFP'].isin(continental)]
    
    print(f"✓ Loaded {len(counties)} counties")
    
    # Create visualizations
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    vis = CountyVisualization(args.results_dir, counties)
    
    # Single encoder comparison (3 panels)
    for encoder in args.encoders:
        output_path = output_dir / f'{encoder}_{args.model}_{args.coefficient}.pdf'
        vis.create_comparison_map(
            encoder_name=encoder,
            coefficient=args.coefficient,
            model_type=args.model,
            output_path=str(output_path)
        )
    
    # Multi-encoder comparison
    if len(args.encoders) > 1:
        output_path = output_dir / f'comparison_{args.model}_{args.coefficient}.pdf'
        vis.create_multi_encoder_comparison(
            encoders=args.encoders,
            coefficient=args.coefficient,
            model_type=args.model,
            output_path=str(output_path)
        )
    
    print(f"\n✓ All visualizations saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
