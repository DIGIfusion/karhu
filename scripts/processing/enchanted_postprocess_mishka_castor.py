"""
Run the enchanted-surrogates postprocessing part for MISHKA and CASTOR runs.
Useful when mishka/castor runs have been generated outside enchanted-surrogates.

Example:
python -u scripts/enchanted_postprocess_mishka_castor.py \
    -d /scratch/project_2009007/data_JET_2H/7d_scan/simulations_success_test \
    -s 0
"""
import os
import json
import argparse
from datetime import datetime

from enchanted_plugin_mishka.mishka_parser import MishkaParser
from enchanted_plugin_castor.castor_parser import CastorParser


def main(main_dir: str, n: int = None, start: int = 0):
    """
    Loop through HELENA runs, get the mishka runs in the subfolder "mishka/"
    and CASTOR runs in subfolder "castor/". Trigger postprocessing.
    """
    print(f"main_dir: {main_dir}")
    m_parser = MishkaParser(default_namelist="")
    c_parser = CastorParser(default_namelist="")
    helena_dirs = sorted([f.path for f in os.scandir(main_dir) if f.is_dir()])
    if n is None:
        helena_dirs = sorted(helena_dirs)[start:]
    else:
        helena_dirs = sorted(helena_dirs)[start:start+n]

    for _i, h_dir in enumerate(helena_dirs):
        if _i % 100 == 0:
            print(f"{datetime.now()} - {_i:>8}/{len(helena_dirs)} ({_i/len(helena_dirs)*100.0:.2f}%)")

        if os.path.exists(os.path.join(h_dir, "mishka")):
            mishka_dirs = sorted([f.path for f in os.scandir(os.path.join(h_dir, "mishka")) if f.is_dir()])
            for m_dir in mishka_dirs:
                x_postprocessing(m_dir, m_parser)
        if os.path.exists(os.path.join(h_dir, "castor")):
            castor_dirs = sorted([f.path for f in os.scandir(os.path.join(h_dir, "castor")) if f.is_dir()])
            for c_dir in castor_dirs:
                x_postprocessing(c_dir, c_parser)
    return

def x_postprocessing(x_dir: str, x_parser):
    """
    Run postprocessing using a Parser from enchanted-surrogaes.
    """
    summary_path = os.path.join(x_dir, 'summary.json')

    # Retrieve params and run_dir from old summary.json file
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding='utf-8') as f:
            summary = json.load(f)
        params = summary.get("params", {})
        run_dir = summary.get("run_dir", None)
        if x_dir != run_dir:
            # print(f"Warning! Replacing run_dir {run_dir} with x_dir {x_dir}")
            run_dir = x_dir

        # Rerun postprocessing
        summary = x_parser.write_summary(run_dir=run_dir, params=params, mpol=71)
    else:
        params = {}
        run_dir = x_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runner")
    parser.add_argument("-i", "--input_number", type=int, default=None, required=False, help="")
    parser.add_argument("-s", "--start_index", type=int, default=0, required=False, help="")
    parser.add_argument("-d", "--main_dir", type=str, default="base", help="")
    config_args = parser.parse_args()
    main(config_args.main_dir, config_args.input_number, start=config_args.start_index)
