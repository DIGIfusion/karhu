#!/bin/bash
# Call with `sbatch SLURMrun.bash` and modify below with your relevant SLURM config
#SBATCH --job-name=ML_TRAINING
#SBATCH --account=project_2009007
#SBATCH --time=10:00:00
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --out=./run_%j.out

source /scratch/project_2009007/.venv/bin/activate
cd /scratch/project_2009007/ambrunc/dev_karhu/karhu-training

python src/run_training_ensamble.py  \
    --dataset_dir db/all/interpolated64.h5   \
    --epochs 500   \
    --ensemble_size 10   \
    --seed 42 \
    --lr 0.001 \
    --beta1 0.91 \
    --beta2 0.97 \
    --weight_decay 0.0 \
    --batch_size 128
