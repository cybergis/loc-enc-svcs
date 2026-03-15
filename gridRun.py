"""
Grid-based location encoder experiments with GeoShapley.

Generates synthetic data on a regular regional grid with MGWR-style
spatially-varying coefficients, trains an ML model augmented with location
embeddings, and extracts spatial effects via GeoShapley.
"""

# NOTE: Parsing at module level (not under __main__) is required for joblib pickling.
from run_utils import build_base_parser, run_experiment_loop
from dgp_utils import create_grid_data

parser = build_base_parser('Grid location encoder experiments')
parser.add_argument('--grid_size', type=int, default=25)
args = parser.parse_args()

run_experiment_loop(
    args,
    data_fn=lambda seed: create_grid_data(
        size=args.grid_size, coord_system='regional',
        noise_std=args.noise_std, random_seed=seed,
        simple_dgp=args.simple_dgp,
    ),
    experiment_label='GRID',
    grid_size=args.grid_size,
)
