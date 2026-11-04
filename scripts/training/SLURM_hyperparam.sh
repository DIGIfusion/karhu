#!/bin/bash
# Call with `sbatch SLURMrun.bash` and modify below with your relevant SLURM config
#SBATCH --job-name=SURROGATE_WORKFLOW
#SBATCH --account=project_2009007
#SBATCH --time=06:00:00
#SBATCH --partition=small
#SBATCH --mem=32GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --out=./run_%j.out

export RAY_DISABLE_DASHBOARD=1
source /scratch/project_2009007/.venv/bin/activate
python3 -u /scratch/project_2009007/ambrunc/dev_karhu/karhu-training/src/run_training_hyperparam.py
