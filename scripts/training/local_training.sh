module load python-pytorch
source "/scratch/project_2009007/ambrunc/dev_karhu/.venv/bin/activate"

python scripts/training/run_training.py  \
    --dataset_dirs  /dataset/project_2020669/KARHU_data/JET_EXTENDED  \
    --epochs 400   \
    --seed 42 \
    --lr 0.001 \
    --batch_size 256 \
    --model_name "KARHU_JET_EXT" \
    # --castor
#     # --beta1 0.91 \
#     # --beta2 0.97 \
#     # --weight_decay 5.5E-5 \

# python scripts/training/run_training_ensemble.py  \
#     --dataset_dir /scratch/project_2009007/ambrunc/dev_karhu/karhu-training/db/JET_1H/interpolated64.h5   \
#     --epochs 10   \
#     --seed 42 \
#     --lr 0.001 \
#     --batch_size 64 \
#     --ensemble_size 3 \
#     --training_split 0.7
    # --beta1 0.91 \
    # --beta2 0.97 \
    # --weight_decay 5.5E-5 \
