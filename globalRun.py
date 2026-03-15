"""
Global-scale location encoder experiments with GeoShapley.

Generates synthetic data on random global coordinates with MGWR-style
spatially-varying coefficients, trains an ML model augmented with location
embeddings, and extracts spatial effects via GeoShapley.
"""

# NOTE: Parsing at module level (not under __main__) is required for joblib pickling.
from run_utils import build_base_parser, run_experiment_loop
from dgp_utils import create_global_data

parser = build_base_parser('Global location encoder experiments')
parser.add_argument('--n_points', type=int, default=3000)
args = parser.parse_args()

run_experiment_loop(
    args,
    data_fn=lambda seed: create_global_data(
        n_points=args.n_points,
        noise_std=args.noise_std,
        random_seed=seed,
        simple_dgp=args.simple_dgp,
    ),
    experiment_label='GLOBAL',
)
