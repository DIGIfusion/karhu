""" Script for merging data sets.

Example:

python scripts/merge_hdf5_datasets.py \
    --dataset_paths \
        /scratch/project_2009007/ambrunc/dev_karhu/karhu-training/db/DIIID/fullprofiles.h5 \
        /scratch/project_2009007/ambrunc/dev_karhu/karhu-training/db/JET_1H/fullprofiles.h5 \
        /scratch/project_2009007/ambrunc/dev_karhu/karhu-training/db/JET_2H/fullprofiles.h5 \
        /scratch/project_2009007/ambrunc/dev_karhu/karhu-training/db/JET_3H/fullprofiles.h5 \
    --new_dataset_path \
        /scratch/project_2009007/ambrunc/dev_karhu/karhu-training/db/all/fullprofiles.h5 
        
"""
from pathlib import Path
from datetime import datetime
import argparse
import h5py
import numpy as np


def merge_hdf5_datasets(input_files, output_file):
    """
    Merge multiple HDF5 datasets created by get_profiles_from_simdir_hdf5
    into a single HDF5 file with identical structure.
    """

    input_files = [Path(f) for f in input_files]

    with h5py.File(output_file, "w") as fout:
        initialized = False
        offset = 0

        for fpath in input_files:
            print(f"Merging {fpath}")
            with h5py.File(fpath, "r") as fin:

                # Number of samples in this file
                n = fin["meta/h_id"].shape[0]

                # -------------------------
                # First file: create layout
                # -------------------------
                if not initialized:
                    def clone_group(name):
                        g_in = fin[name]
                        g_out = fout.create_group(name)

                        for dname, dset in g_in.items():
                            full_shape = dset.shape
                            if len(full_shape) < 1:
                                raise ValueError(f"Dataset {name}/{dname} has invalid shape {full_shape}")

                            g_out.create_dataset(
                                dname,
                                shape=(0,) + full_shape[1:],
                                maxshape=(None,) + full_shape[1:],
                                dtype=dset.dtype,
                                chunks=True,
                                compression="gzip",
                            )

                            for k, v in dset.attrs.items():
                                g_out[dname].attrs[k] = v


                    clone_group("profiles")
                    clone_group("scalars")
                    clone_group("growthrates_mishka")
                    # clone_group("growthrates_castor")

                    # meta (strings)
                    dt = h5py.string_dtype("utf-8")
                    g_meta = fout.create_group("meta")
                    g_meta.create_dataset("h_id", shape=(0,), maxshape=(None,), dtype=dt)
                    g_meta.create_dataset("h_dir", shape=(0,), maxshape=(None,), dtype=dt)

                    initialized = True

                # -------------------------
                # Append data
                # -------------------------
                def append_group(group_name):
                    print(f"Appending group {group_name}")
                    for name, dset_out in fout[group_name].items():
                        dset_in = fin[group_name][name]

                        # --- SHAPE VALIDATION ---
                        if dset_in.shape[1:] != dset_out.shape[1:]:
                            raise ValueError(
                                f"Shape mismatch in {group_name}/{name}: "
                                f"input {dset_in.shape}, output {dset_out.shape}"
                            )

                        dset_out.resize(offset + n, axis=0)
                        dset_out[offset:offset+n, ...] = dset_in[:]


                append_group("profiles")
                append_group("scalars")
                append_group("growthrates_mishka")
                # append_group("growthrates_castor")

                # meta
                for key in ["h_id", "h_dir"]:
                    d_out = fout["meta"][key]
                    d_in  = fin["meta"][key]
                    d_out.resize(offset + n, axis=0)
                    d_out[offset:offset+n] = d_in[:]

                offset += n

    print(f"Final dataset size: {offset}")


def main():
    print(f"Start time: {datetime.now()}")

    # Create argument parser
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset_paths", nargs='+', default=[])
    parser.add_argument("--new_dataset_path", type=str, help="")
    args = parser.parse_args()

    merge_hdf5_datasets(args.dataset_paths, args.new_dataset_path)

    print(f"End time: {datetime.now()}")

if __name__ == "__main__":
    main()
