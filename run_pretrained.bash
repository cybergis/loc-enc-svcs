#!/bin/bash
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-10

# Run experiments with TorchSpatial pretrained encoder weights.
# Encoder index 11 (none/baseline) is excluded — no pretrained weights apply.
#
# Best-available checkpoints (preferring inat_2018 > inat_2017 > birdsnap_orig > fmow):
#   idx 0  Space2Vec-theory   inat_2018
#   idx 1  tile_ffn           birdsnap_orig
#   idx 2  wrap_ffn           birdsnap_orig
#   idx 3  Sphere2Vec-sphereM birdsnap_orig
#   idx 4  Sphere2Vec-sphereM+birdsnap_orig
#   idx 5  rff                birdsnap_orig
#   idx 6  Sphere2Vec-sphereC fmow
#   idx 7  Sphere2Vec-sphereC+fmow
#   idx 8  NeRF               nabirds_ebird
#   idx 9  Sphere2Vec-dfs     inat_2018
#   idx 10 Space2Vec-grid     inat_2018
#
# The inat_2018 checkpoints for the other encoders appear corrupted in the
# TorchSpatial release; this script uses the best loadable alternative.
#
# Usage:
#   sbatch --job-name=grid_pretrained   run_pretrained.bash grid   [output_dir] [model_type] [num_reps]
#   sbatch --job-name=county_pretrained run_pretrained.bash county [output_dir] [model_type] [num_reps]
#   sbatch --job-name=global_pretrained run_pretrained.bash global [output_dir] [model_type] [num_reps]

EXPERIMENT=${1:?"Usage: run_pretrained.bash <grid|county|global> [output_dir] [model_type] [num_reps]"}
OUTPUT_DIR=${2:-"./results/${EXPERIMENT}_simple_pretrained_dim8"}
MODEL_TYPE=${3:-"MLP"}
NUM_REPS=${4:-25}

ENCODER_IDX=${SLURM_ARRAY_TASK_ID}

case "$EXPERIMENT" in
    grid)    SCRIPT="gridRun.py" ;;
    county)  SCRIPT="countiesRun.py" ;;
    global)  SCRIPT="globalRun.py" ;;
    *)       echo "Unknown experiment: $EXPERIMENT"; exit 1 ;;
esac

# Encoder names (must match ENCODER_CONFIGS indices 0-10 in run_utils.py)
ENCODER_NAMES=(
    "Space2Vec-theory"    # 0
    "tile_ffn"            # 1
    "wrap_ffn"            # 2
    "Sphere2Vec-sphereM"  # 3
    "Sphere2Vec-sphereM+" # 4
    "rff"                 # 5
    "Sphere2Vec-sphereC"  # 6
    "Sphere2Vec-sphereC+" # 7
    "NeRF"                # 8
    "Sphere2Vec-dfs"      # 9
    "Space2Vec-grid"      # 10
)

PT_DIR="/u/dkiv2/group_dkiv2/active/TorchSpatial/pre_trained_models"
SCRIPT_DIR="/u/dkiv2/group_dkiv2/active/effectsExplainableEmbeddings"

CHECKPOINTS=(
    "${PT_DIR}/space2vec_theory/model_inat_2018_Space2Vec-theory_0.0200_64_0.0500000_360.000_1_512_BATCH4096_leakyrelu.pth.tar"                     # 0 inat_2018
    "${PT_DIR}/tile/model_birdsnap_orig_meta_tile_ffn_inception_v3_0.00018903_32_0.0000019_1_512_leakyrelu.pth.tar"                                   # 1 birdsnap_orig
    "${PT_DIR}/wrap_ffn/model_birdsnap_orig_meta_wrap_ffn_inception_v3_0.0020_64_0.0100000_1_512.pth.tar"                                             # 2 birdsnap_orig
    "${PT_DIR}/sphere2vec_sphereM/model_birdsnap_orig_meta_Sphere2Vec-sphereM_inception_v3_0.0010_64_0.0005000_1_512.pth.tar"                         # 3 birdsnap_orig
    "${PT_DIR}/sphere2vec_sphereMplus/model_birdsnap_orig_meta_Sphere2Vec-sphereM+_inception_v3_0.00166576_16_0.0029816_3_256.pth.tar"                # 4 birdsnap_orig
    "${PT_DIR}/rff/model_birdsnap_orig_meta_rff_inception_v3_0.0020_64_0.1000000_1_512_1.0.pth.tar"                                                   # 5 birdsnap_orig
    "${PT_DIR}/sphere2vec_sphereC/model_fmow_Sphere2Vec-sphereC_inception_v3_0.0050_64_0.0005000_1_512.pth.tar"                                       # 6 fmow
    "${PT_DIR}/sphere2vec_sphereCplus/model_fmow_Sphere2Vec-sphereC+_inception_v3_0.0050_64_0.0001000_1_512.pth.tar"                                  # 7 fmow
    "${PT_DIR}/nerf/model_nabirds_ebird_meta_NeRF_inception_v3_0.0100_16_0.1000000_1_256_sigmoid.pth.tar"                                             # 8 nabirds_ebird
    "${PT_DIR}/sphere2vec_dfs/model_inat_2018_Sphere2Vec-dfs_0.0100_8_0.0001000_1.000_1_512_BATCH4096_leakyrelu.pth.tar"                             # 9 inat_2018
    "${PT_DIR}/space2vec_grid/model_inat_2018_Space2Vec-grid_0.0100_32_0.0001000_360.000_1_512_BATCH4096_leakyrelu.pth.tar"                           # 10 inat_2018
)

CHECKPOINT="${CHECKPOINTS[$ENCODER_IDX]}"

echo "========================================"
echo "${EXPERIMENT^^} PRETRAINED EXPERIMENT"
echo "Encoder: $ENCODER_IDX (${ENCODER_NAMES[$ENCODER_IDX]})"
echo "Checkpoint: $(basename $CHECKPOINT)"
echo "Model: $MODEL_TYPE | Reps: $NUM_REPS"
echo "Output: $OUTPUT_DIR"
echo "========================================"

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

# Conda setup — find conda from standard locations, no hardcoded home dir
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
    --pretrained_weights "$CHECKPOINT"

echo ""
echo "Completed: ${ENCODER_NAMES[$ENCODER_IDX]}"
