#!/bin/bash
#SBATCH --account=project_2009007
#SBATCH --job-name=karhu_ens
#SBATCH --partition=small
#SBATCH --array=0-20
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=12G
#SBATCH --output=logs/%x_%A_%a.out

module load cuda

source /scratch/project_2009007/.venv/bin/activate
cd /scratch/project_2009007/ambrunc/dev_karhu/karhu-training

python src/run_training_ensemble_parallel.py  \
    --dataset_dir db/jet_2H/interpolated64.h5   \
    --ensemble_id ${SLURM_ARRAY_TASK_ID} \
    --epochs 500   \
    --seed 42 \
    --lr 0.001 \
    --batch_size 256 \
    --training_split 0.7

    # --beta1 0.91 \
    # --beta2 0.99 \
    # --weight_decay 0.0 \