#!/bin/bash
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-11

# Unified experiment runner for grid, counties, and global experiments
# Usage: sbatch --job-name=grid    run_experiments.bash grid    [output_dir] [model_type] [num_reps] [extra_flags]
#        sbatch --job-name=county  run_experiments.bash county  [output_dir] [model_type] [num_reps] [extra_flags]
#        sbatch --job-name=global  run_experiments.bash global  [output_dir] [model_type] [num_reps] [extra_flags]

EXPERIMENT=${1:?"Usage: run_experiments.bash <grid|county|global> [output_dir] [model_type] [num_reps] [extra_flags]"}
OUTPUT_DIR=${2:-"./results/${EXPERIMENT}"}
MODEL_TYPE=${3:-"MLP"}
NUM_REPS=${4:-25}
EXTRA_FLAGS=${5:-""}    # e.g. "--train_encoder --encoder_epochs 500"

ENCODER_IDX=${SLURM_ARRAY_TASK_ID}

# Map experiment type to Python script
case "$EXPERIMENT" in
    grid)    SCRIPT="gridRun.py" ;;
    county)  SCRIPT="countiesRun.py" ;;
    global)  SCRIPT="globalRun.py" ;;
    *)       echo "Unknown experiment: $EXPERIMENT"; exit 1 ;;
esac

# Encoder names (must match ENCODER_CONFIGS in Python scripts)
ENCODER_NAMES=(
    "Space2Vec-theory" "tile_ffn" "wrap_ffn"
    "Sphere2Vec-sphereM" "Sphere2Vec-sphereM+" "rff"
    "Sphere2Vec-sphereC" "Sphere2Vec-sphereC+" "NeRF"
    "Sphere2Vec-dfs" "Space2Vec-grid" "none"
)

echo "========================================"
echo "${EXPERIMENT^^} EXPERIMENT"
echo "Encoder: $ENCODER_IDX (${ENCODER_NAMES[$ENCODER_IDX]})"
echo "Model: $MODEL_TYPE | Reps: $NUM_REPS"
echo "Output: $OUTPUT_DIR"
echo "Extra: $EXTRA_FLAGS"
echo "========================================"

# Conda setup — find conda from standard locations, no hardcoded home dir
SCRIPT_DIR="/u/dkiv2/group_dkiv2/active/loc-enc-svcs"
if command -v conda &>/dev/null; then
    CONDA_BASE=$(conda info --base 2>/dev/null)
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="$HOME/miniconda3"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="$HOME/anaconda3"
else
    echo "ERROR: conda not found. Activate your environment manually." >&2
    exit 1
fi

__conda_setup="$("${CONDA_BASE}/bin/conda" 'shell.bash' 'hook' 2>/dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
fi
unset __conda_setup

conda activate e

mkdir -p "$OUTPUT_DIR" slurm_logs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$SCRIPT_DIR"
python $SCRIPT \
    --encoder_index $ENCODER_IDX \
    --model_type $MODEL_TYPE \
    --num_repetitions $NUM_REPS \
    --noise_std 0.1 \
    --output_dir $OUTPUT_DIR \
    --random_seed 222 \
    $EXTRA_FLAGS

echo ""
echo "Completed: ${ENCODER_NAMES[$ENCODER_IDX]}"
