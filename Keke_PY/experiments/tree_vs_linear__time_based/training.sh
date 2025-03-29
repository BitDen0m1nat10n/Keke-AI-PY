#!/bin/bash

#SBATCH --job-name=TreeVsLinear_TimeBased_40Gens_2sec
#SBATCH --mail-user=rongero@tnt.uni-hannover.de
#SBATCH --mail-type=ALL
#SBATCH --partition=cpu_long_stud

#SBATCH --array=0-1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=4G
#SBATCH --time=8-0 # 8Days + 0hours
#SBATCH --output=TreeVsLinear_TimeBased_40Gens_2sec_%A_%a-out.txt   # Logdatei für den merged STDOUT/STDERR output (%A wird durch slurm Job-ID ersetzt und %a durch den Array Index)


# setup conda, and a conda-environment like environment.txt :
#source setup_on_my_laptop.sh
source setup_on_cluster.sh



# Change to my work dir
# SLURM_SUBMIT_DIR is an environment variable that automatically gets
# assigned the directory from which you did submit the job. A batch job
# is like a new login, so you'll initially be in your HOME directory.
# So it's usually a good idea to first change into the directory you did
# submit your job from.
cd $SLURM_SUBMIT_DIR

python Keke_PY/experiments/tree_vs_linear__time_based/training.py --argument $SLURM_ARRAY_TASK_ID
