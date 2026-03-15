"""
County-level location encoder experiments with GeoShapley.

Generates synthetic data on US county geometries with MGWR-style
spatially-varying coefficients, trains an ML model augmented with location
embeddings, and extracts spatial effects via GeoShapley.
"""

# NOTE: Parsing at module level (not under __main__) is required for joblib pickling.
from run_utils import build_base_parser, run_experiment_loop
from dgp_utils import create_county_data

parser = build_base_parser('County location encoder experiments')
parser.add_argument('--shapefile_path', type=str, default=None)
args = parser.parse_args()


def _county_data_fn(seed):
    data, extent, _ = create_county_data(
        shapefile_path=args.shapefile_path,
        noise_std=args.noise_std,
        random_seed=seed,
        simple_dgp=args.simple_dgp,
    )
    return data, extent


run_experiment_loop(
    args,
    data_fn=_county_data_fn,
    experiment_label='COUNTY',
    extra_spatial_cols_fn=lambda data: {'GEOID': data['GEOID'], 'NAME': data['NAME']},
)
