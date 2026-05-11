#!/bin/bash
# Call with `sbatch SLURMrun.bash` and modify below with your relevant SLURM config
#SBATCH --job-name=SURROGATE_WORKFLOW
#SBATCH --account=project_2009007
#SBATCH --time=10:00:00
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --out=./run_%j.out


cd /scratch/project_2009007/ambrunc/dev_karhu/karhu-training

# deactivate
source /scratch/project_2009007/ambrunc/develop2/.venv/bin/activate

# Postprocess MISHKA and CASTOR runs
python -u scripts/enchanted_postprocess_mishka_castor.py \
    -d /scratch/project_2009007/data_DIIID/success \
    -s 0

# Postprocess HELENA runs
python -u scripts/enchanted_postprocess_helena.py \
    -d /scratch/project_2009007/data_DIIID/success \
    -s 0

# Collect data
python -u scripts/enchanted_collect_data_to_hdf5.py \
    --simulation_dirs \
        "/scratch/project_2009007/data_JET_1H/success" \
        "/scratch/project_2009007/data_JET_2H/success" \
        "/scratch/project_2009007/data_JET_3H/success" \
        "/scratch/project_2009007/data_DIIID/success" \
    --db_dir /scratch/project_2009007/data_KARHU/v2.0

# Restructure to KARHU axes
python -u scripts/enchanted_fullprofiles_to_karhu_grid.py \
        --db_fullprofiles /scratch/project_2009007/data_KARHU/v2.0/fullprofiles.h5 \
        --db_dir /scratch/project_2009007/data_KARHU/v2.0

