#!/bin/bash

set -euox pipefail 

#unset PYTHONPATH
#unset LD_LIBRARY_PATH

# 2. Try module purge, but silence errors because we don't care if 
# it complains about unloading the parent's complex modules.
#module purge > /dev/null 2>&1

# module purge
#module load python/3.11-base
#module load gcc/13.2.0
source /home/mn2596/JETPEDESTAL_ANALYSIS/karhu/.venv37/bin/activate
INFERENCE=/home/mn2596/JETPEDESTAL_ANALYSIS/karhu/src/run.py
INFERENCE=/home/mn2596/JETPEDESTAL_ANALYSIS/karhu/src/karhu/cli/inference_from_helena.py
HELDIR=$1
MODEL=/home/mn2596/JETPEDESTAL_ANALYSIS/karhu/model/jet_2H
WRITEOUT=$2
python3 $INFERENCE -hd=$HELDIR -m=$MODEL -w=$WRITEOUT
