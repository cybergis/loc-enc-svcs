#!/bin/bash
#SBATCH --job-name=regen_heatmap
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
source /u/dkiv2/miniconda3/etc/profile.d/conda.sh
conda activate e
cd /u/dkiv2/group_dkiv2/active/loc-enc-svcs
python -c "import seaborn" 2>/dev/null || pip install --quiet seaborn
python regen_heatmap.py
