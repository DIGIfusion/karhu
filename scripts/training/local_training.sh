source "/scratch/project_2009007/.venv/bin/activate"

python src/run_training_ensemble.py  \
    --dataset_dir db/JET_1H/interpolated64.h5   \
    --epochs 200   \
    --seed 42 \
    --lr 0.001 \
    --batch_size 64 \
    --ensemble_size 20 \
    --training_split 0.7
    # --beta1 0.91 \
    # --beta2 0.97 \
    # --weight_decay 5.5E-5 \
