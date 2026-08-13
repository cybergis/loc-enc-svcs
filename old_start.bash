#!/bin/bash

#SBATCH --job-name=svcr
#SBATCH --time=04:00:00
#SBATCH --mem=1024G
#SBATCH --nodes=1                # Request one node
#SBATCH --ntasks=1               # Run a single task
#SBATCH --cpus-per-task=96       # Number of CPU cores per task

# Load the pyton environment
echo "Loading conda environment"
source /u/dkiv2/miniconda3/bin/activate 
conda activate night

cmds=(
    # "python run_ml_exps.py"
    # "python voting_start.py"
    # "python explainable_spatialeffects_embedding_test.py"
    "python dumb.py"
)

for ((i = 0; i < ${#cmds[@]}; i++))
do
    # Print the command and run it
    echo "Running command: ${cmds[$i]}"
    ${cmds[$i]}
done

echo "Experimental run complete. Script closing."

# End the script
exit