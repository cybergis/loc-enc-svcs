#!/bin/bash
#SBATCH --job-name=full_dgp_report
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# Aggregates simple+full results and writes results/REPORT_full_dgp.txt.
# Intended to run after the wrap_ffn reruns via --dependency=afterok.

source /u/dkiv2/miniconda3/etc/profile.d/conda.sh
conda activate e
cd /u/dkiv2/group_dkiv2/active/loc-enc-svcs

echo "Aggregating..."
python aggregate_metrics.py --results_root ./results --prefix "grid_,county_,global_"
echo "Building full-DGP report..."
python report_full_dgp.py
echo "Done. See results/REPORT_full_dgp.txt and results/statistical_tests.csv"
