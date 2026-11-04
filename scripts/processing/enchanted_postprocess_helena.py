"""
Run postprocessing routine from Enchanted surrogates.

Example:
    source /scratch/project_2009007/ambrunc/develop2/.venv/bin/activate
    python -u scripts/enchanted_postprocess_helena.py \
        -d /scratch/project_2009007/data_DIIID/success \
        -s 0

"""
import os
import json
import argparse
from datetime import datetime

from enchanted_plugin_helena.helena_parser import HelenaParser


def main(main_dir: str, n: int = None, start: int = 0):
    print(f"main_dir: {main_dir}")
    h_parser = HelenaParser()
    helena_dirs = [f.path for f in os.scandir(main_dir) if f.is_dir()]
    if n is None:
        helena_dirs = sorted(helena_dirs)[start:]
    else:
        helena_dirs = sorted(helena_dirs)[start:start+n]

    for _i, h_dir in enumerate(helena_dirs):
        if _i % 100 == 0:
            print(f"{datetime.now()} - {_i:>8}/{len(helena_dirs)} ({_i/len(helena_dirs)*100.0:.2f}%)")
        helena_postprocessing(h_dir, h_parser)
    return

def helena_postprocessing(h_dir: str, h_parser):
    summary_path = os.path.join(h_dir, 'summary.json')

    # Retrieve params and run_dir from old summary.json file
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding='utf-8') as f:
            summary = json.load(f)
        params = summary.get("params", {})
        run_dir = summary.get("run_dir", None)
        if h_dir != run_dir:
            # print(f"Warning! Replacing run_dir {run_dir} with h_dir {h_dir}")
            run_dir = h_dir
    else:
        params = {}
        run_dir = h_dir

    # Rerun postprocessing
    summary = h_parser.write_summary(run_dir=run_dir, params=params)
    h_parser.collect_growthrates_from_mishka(h_dir=run_dir, save=True)
    h_parser.collect_growthrates_from_castor(h_dir=run_dir, save=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runner")
    parser.add_argument("-n", "--number_of_dirs", type=int, default=None, required=False, help="")
    parser.add_argument("-s", "--start_index", type=int, default=0, required=False, help="")
    parser.add_argument("-d", "--main_dir", type=str, default="base", help="")
    config_args = parser.parse_args()
    main(config_args.main_dir, config_args.number_of_dirs, start=config_args.start_index)
