#!/bin/bash

#SBATCH --job-name=encoder_job_%A_%a
#SBATCH --output=slurm_logs/encoder_out_%A_%a.txt
#SBATCH --error=slurm_logs/encoder_err_%A_%a.txt

#SBATCH --time=22:00:00
#SBATCH --mem=192g
#SBATCH --ntasks=1          # Request a single task
#SBATCH --cpus-per-task=24  # Assign 24 CPU cores to that task
#SBATCH --mail-user=dkiv2@illinois.edu

# --- User Configuration ---
NUM_REPETITIONS=25
ENCODER_INDEX=${SLURM_ARRAY_TASK_ID}

echo "=========================================================="
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================================="

# --- Setup Conda Environment ---
__conda_setup="$('/u/dkiv2/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/u/dkiv2/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/u/dkiv2/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/u/dkiv2/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

# --- Activate Your Environment ---
conda activate e

# --- Directory Setup ---
PROJECT_DIR=$(pwd)
RESULTS_DIR_BASE="${PROJECT_DIR}/results/AUTO_FINAL_DRAFT_NOV17"
mkdir -p "${PROJECT_DIR}/slurm_logs"
mkdir -p "${RESULTS_DIR_BASE}"

echo "Starting job for Encoder Index: ${ENCODER_INDEX}"

# --- Set Environment Variable for Threading ---
# This helps libraries like NumPy, Scikit-learn, etc., know how many threads to use.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- Execute Python Script ---
python "${PROJECT_DIR}/embeddingsRun.py" --encoder_index "${ENCODER_INDEX}" --num_repetitions "${NUM_REPETITIONS}" --base_experiment_dir "${RESULTS_DIR_BASE}"

echo "Job finished for Encoder Index: ${ENCODER_INDEX}"

exit