#!/bin/bash
# Call with `sbatch SLURMrun.bash` and modify below with your relevant SLURM config
#SBATCH --job-name=ML_TRAINING
#SBATCH --account=project_2009007
#SBATCH --time=02:00:00
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --out=./run_%j.out

module load python-pytorch
source /scratch/project_2009007/ambrunc/dev_karhu/.venv/bin/activate

cd /scratch/project_2009007/ambrunc/dev_karhu/karhu

python scripts/training/run_training.py  \
    --db_dirs /scratch/project_2009007/ambrunc/dev_karhu/karhu-training/db/db_jet \
    --epochs 400   \
    --seed 42 \
    --lr 0.001 \
    --batch_size 256 \
    --model_name "KARHU_JET"

