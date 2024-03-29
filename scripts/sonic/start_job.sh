#!/bin/bash -l
#SBATCH --job-name=adhocSL-cpu

# speficity number of nodes
#SBATCH -N 1
#SBATCH --ntasks-per-node 2
# specify the walltime e.g 20 mins
#SBATCH -t 90:00:00

# set to email at start,end and failed jobs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=joana.tirana@ucdconnect.ie

# run from current directory
cd $SLURM_SUBMIT_DIR
#=/home/people/21211297/scratch/NIID-Bench-adhocSL

# command to use

conda activate myenv
python --version
./command.sh
