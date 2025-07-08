#!/bin/bash

#SBATCH --job-name=BaselineEvaluation
#SBATCH --mail-user=rongero@tnt.uni-hannover.de
#SBATCH --mail-type=ALL
#SBATCH --partition=cpu_normal_stud

#SBATCH --array=0-9
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=4G
#SBATCH --time=0-5 # Days_hours
#SBATCH --output=BaselineEvaluation_%A_%a-out.txt   # Logdatei für den merged STDOUT/STDERR output (%A wird durch slurm Job-ID ersetzt und %a durch den Array Index)


# setup conda, and a conda-environment like environment.txt :
#source setup_on_my_laptop.sh
source setup_on_cluster.sh


echo "RUNNING ON FOLLOWING DEVICE:"
echo $SLURMD_NODENAME
echo ""



cd $SLURM_SUBMIT_DIR

python Keke_PY/tree_vs_linear__time_based__single_levels/evaluation/evaluate_baselines.py
