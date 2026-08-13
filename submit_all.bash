#!/bin/bash

#SBATCH --job-name=all_job_%A_%a # Job name with array task ID
#SBATCH --output=slurm_logs/all_job_%A_%a.txt   # Standard output and error log
#SBATCH --error=slurm_logs/all_job_%A_%a.txt    # Standard error log

# Submit the sbatch script as a job array
# The %A placeholder will be replaced by the job ID
# The %a placeholder will be replaced by the array task ID (encoder index)
# sbatch --array=0-${LAST_ENCODER_INDEX} run_single_encoder.sbatch
sbatch --array=0-10 run_single.bash

echo "Jobs submitted."

# End the script
exit