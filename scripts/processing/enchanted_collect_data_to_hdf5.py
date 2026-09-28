"""
Collect data from HELENA, MISHKA and CASTOR simulation directories created by
the Enchanted-surrogates framework. The expected directory structure:

base_run_dir
    - helena_dir
        - mishka
            - 5
            ...
        - castor
            - 5
            ...
    ...

Example
    python -u scripts/enchanted_collect_data_to_hdf5.py \
        --simulation_dirs "/scratch/project_2009007/data_JET_1H/success" \
             "/scratch/project_2009007/data_JET_2H/success" \
                 "/scratch/project_2009007/data_JET_3H/success" \
                     "/scratch/project_2009007/data_DIIID/success" \
        --db_dir /scratch/project_2009007/data_KARHU/v2.0

"""
import os
import json
from datetime import datetime
import argparse

import numpy as np
import f90nml
import h5py


def get_profiles_from_simdir_hdf5(H_DIRS, SAVE_PATH_FULL_PROFILES):
    """
    Collect profiles, scalars, and growthrates from simulation directories
    and store them in a structured HDF5 file.
    """
    print("Collecting simulations to HDF5")

    h5_path = SAVE_PATH_FULL_PROFILES

    with h5py.File(h5_path, "w") as h5f:
        grp_profiles = h5f.create_group("profiles")
        grp_scalars = h5f.create_group("scalars")
        grp_meta = h5f.create_group("meta")
        grp_mishka = h5f.create_group("growthrates_mishka")
        # grp_castor = h5f.create_group("growthrates_castor")

        profile_dsets = {}
        scalar_dsets = {}
        mishka_dsets = {}
        # castor_dsets = {}
        h_ids = []
        h_dirs = []

        for i, h_dir in enumerate(H_DIRS):
            if i % 100 == 0:
                print(datetime.now(), i)

            # The sample has failed or not been postprocessed if summary files doesn't exist
            summary_path = os.path.join(h_dir, "summary.json")
            if not os.path.isfile(summary_path):
                print(f"({i}) {h_dir} missing summary")
                continue

            # Load summary
            with open(summary_path) as f:
                summary = json.load(f)

            # Load MISHKA growthrates
            all_gr_mishka = np.load(os.path.join(h_dir, "growthrates_mishka.npy"))
            growthrates_mishka = {
                "ntor": all_gr_mishka[:, 0].astype(np.int32),
                "gamma2": all_gr_mishka[:, 1].astype(np.float32),
                "iterations": all_gr_mishka[:, 3].astype(np.float32),
            }
            idx_max_mishka = np.argmax(all_gr_mishka[:, 1])
            max_gr_mishka = np.sqrt(np.max((0.0, all_gr_mishka[idx_max_mishka, 1])))
            max_gr_ntor_mishka = all_gr_mishka[idx_max_mishka, 0]

            if max_gr_mishka is None:
                print(f"Growthrate none: {h_dir}")
                continue

            # # Load CASTOR growthrates (if exists)
            # if os.path.exists(os.path.join(h_dir, "growthrates_castor.npy")):
            #     all_gr_castor = np.load(os.path.join(h_dir, "growthrates_castor.npy"))
            #     if len(all_gr_castor) > 0:
            #         idx_max_castor = np.argmax(all_gr_castor[:, 1])
            #         max_gr_castor = np.sqrt(np.max((0.0, all_gr_castor[idx_max_castor, 1])))
            #         max_gr_ntor_castor = all_gr_castor[idx_max_castor, 0]
            #     else:
            #         all_gr_castor = np.empty(shape=(0,4))
            #         max_gr_castor = np.nan
            #         max_gr_ntor_castor = np.nan
            # else:
            #     all_gr_castor = np.empty(shape=(0,4))
            #     max_gr_castor = np.nan
            #     max_gr_ntor_castor = np.nan

            # growthrates_castor = {
            #     "ntor": all_gr_castor[:, 0].astype(np.int32),
            #     "gamma2": all_gr_castor[:, 1].astype(np.float32),
            #     "iterations": all_gr_castor[:, 3].astype(np.float32),
            # }

            scalars = {
                "betan": summary["betan"],
                # "betap": summary["betap"],
                # "total_current": summary["total_current"],
                "radius": summary["radius"],
                # "b0": summary["b0"],
                # "bt": summary["bt"],
                "ip": summary["ip"],
                "q_on_axis": summary["q_on_axis"],
                "q_at_boundary": summary["q_at_boundary"],
                "rvac": summary["rvac"],
                "bvac": summary["bvac"],
                "rmag": summary["rmag"],
                "bmag": summary["bmag"],
                # "mercier_stable": summary["mercier_stable"],
                # "ballooning_stable": summary["ballooning_stable"],
                "max_gr_mishka": max_gr_mishka,
                "max_gr_ntor_mishka": max_gr_ntor_mishka,
                # "max_gr_castor": max_gr_castor,
                # "max_gr_ntor_castor": max_gr_ntor_castor,
            }

            h_ids.append(str(summary["h_id"]))
            h_dirs.append(str(h_dir))

            # Load profiles
            profiles = {
                "cs": np.load(os.path.join(h_dir, "cs.npy")),
                "qs": np.load(os.path.join(h_dir, "qs.npy")),
                "p0": np.load(os.path.join(h_dir, "p0.npy")),
                "rbphi": np.load(os.path.join(h_dir, "rbphi.npy")),
                "vxvy": np.load(os.path.join(h_dir, "vxvy.npy")),
                "boundary_polar": np.load(os.path.join(h_dir, "boundary_polar.npy")),
                "resistivity": np.load(os.path.join(h_dir, "resistivity.npy")),
            }

            # Datasets has to be created before writing to them
            if not profile_dsets:
                for name, arr in profiles.items():
                    dset = grp_profiles.create_dataset(
                        name,
                        shape=(0,) + arr.shape,
                        maxshape=(None,) + arr.shape,
                        dtype=arr.dtype,
                        chunks=True,
                        compression="gzip",
                    )
                    dset.attrs["description"] = f"{name} profile"
                    profile_dsets[name] = dset

            if not scalar_dsets:
                for name, val in scalars.items():
                    dset = grp_scalars.create_dataset(
                        name,
                        shape=(0,),
                        maxshape=(None,),
                        dtype=np.asarray(val).dtype,
                        chunks=True,
                        compression="gzip",
                    )
                    dset.attrs["description"] = name
                    scalar_dsets[name] = dset

            # Variable length data types for growth rates
            vlen_int   = h5py.vlen_dtype(np.int32)
            vlen_float = h5py.vlen_dtype(np.float32)
            if not mishka_dsets:
                dset = grp_mishka.create_dataset(
                    "ntor",
                    shape=(0,),
                    maxshape=(None,),
                    dtype=vlen_int,
                    chunks=True,
                    compression="gzip",
                )
                dset.attrs["description"] = "ntor"
                mishka_dsets["ntor"] = dset
                dset = grp_mishka.create_dataset(
                    "gamma2",
                    shape=(0,),
                    maxshape=(None,),
                    dtype=vlen_float,
                    chunks=True,
                    compression="gzip",
                )
                dset.attrs["description"] = "gamma2"
                mishka_dsets["gamma2"] = dset
                dset = grp_mishka.create_dataset(
                    "iterations",
                    shape=(0,),
                    maxshape=(None,),
                    dtype=vlen_int,
                    chunks=True,
                    compression="gzip",
                )
                dset.attrs["description"] = "iterations"
                mishka_dsets["iterations"] = dset

            # if not castor_dsets:
            #     dset = grp_castor.create_dataset(
            #         "ntor",
            #         shape=(0,),
            #         maxshape=(None,),
            #         dtype=vlen_int,
            #         chunks=True,
            #         compression="gzip",
            #     )
            #     dset.attrs["description"] = "ntor"
            #     castor_dsets["ntor"] = dset
            #     dset = grp_castor.create_dataset(
            #         "gamma2",
            #         shape=(0,),
            #         maxshape=(None,),
            #         dtype=vlen_float,
            #         chunks=True,
            #         compression="gzip",
            #     )
            #     dset.attrs["description"] = "gamma2"
            #     castor_dsets["gamma2"] = dset
            #     dset = grp_castor.create_dataset(
            #         "iterations",
            #         shape=(0,),
            #         maxshape=(None,),
            #         dtype=vlen_int,
            #         chunks=True,
            #         compression="gzip",
            #     )
            #     dset.attrs["description"] = "iterations"
            #     castor_dsets["iterations"] = dset

            # Add profiles to datasets
            for name, arr in profiles.items():
                dset = profile_dsets[name]
                dset.resize(dset.shape[0] + 1, axis=0)
                dset[-1] = arr

            # Add scalar to datasets
            for name, val in scalars.items():
                dset = scalar_dsets[name]
                dset.resize(dset.shape[0] + 1, axis=0)
                dset[-1] = val

            # Add growthrates to datasets
            for key, arr in growthrates_mishka.items():
                dset = mishka_dsets[key]
                dset.resize(dset.shape[0] + 1, axis=0)
                dset[-1] = arr

            # Add growthrates to datasets
            # for key, arr in growthrates_castor.items():
            #     dset = castor_dsets[key]
            #     dset.resize(dset.shape[0] + 1, axis=0)
            #     dset[-1] = arr


        # Save meta info as string datasets
        dt = h5py.string_dtype(encoding='utf-8')
        grp_meta.create_dataset("h_id", data=np.array(h_ids, dtype=dt))
        # grp_meta.create_dataset("h_dir", data=np.array(h_dirs, dtype=dt))

    return


def main():
    print(f"Start time: {datetime.now()}")

    # Create argument parser
    parser = argparse.ArgumentParser(description="Collect data into npy array data sets.")
    parser.add_argument("--simulation_dirs", nargs='+', default=[])
    parser.add_argument("--db_dir", type=str, help="Path to directory where the final database is saved")
    args = parser.parse_args()

    helena_simulation_base_dirs = args.simulation_dirs
    DB_DIR = args.db_dir
    os.makedirs(DB_DIR, exist_ok=True)

    SAVE_PATH_FULL_PROFILES = os.path.join(DB_DIR, "fullprofiles.h5")

    # Loop through run directory and collect the postprocessed .npy profiles
    helena_directories = []
    for h_dir in helena_simulation_base_dirs:
        helena_directories = (
            helena_directories
            + sorted([f.path for f in os.scandir(h_dir) if f.is_dir()])
        )
    print("Number of HELENA directories in processing:", len(helena_directories))

    # Fetch and save data
    get_profiles_from_simdir_hdf5(helena_directories, SAVE_PATH_FULL_PROFILES)

    print(f"End time: {datetime.now()}")


if __name__ == "__main__":
    main()
