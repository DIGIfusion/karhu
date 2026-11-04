"""


Example
    source /scratch/project_2009007/ambrunc/develop2/.venv/bin/activate
    python -u scripts/enchanted_fullprofiles_to_karhu_grid.py \
        --db_fullprofiles db/JET_1H/fullprofiles.h5 \
        --db_dir db/JET_1H


"""
import argparse
import numpy as np
import os
import json
from datetime import datetime
from scipy.interpolate import interp1d
import h5py


def interpolate_psi_profile(x_0, y_0, x_1):
    """
    y_0 is the values at positions x_0
    x_0 is the corresponding locations between 0 and 1
    x_1 is the new locations where you want to interpolate y_0
    """

    interpolation_function = interp1d(x_0, y_0, kind="linear", fill_value='extrapolate')

    # Use the interpolation function to find y_1 at new x_1 locations
    y_1 = interpolation_function(x_1)

    # y_1 now contains the scaled down values corresponding to the new locations x_1
    return y_1

def create_data_new_grid_h5(
    in_path,
    out_path,
    x_1,
    theta_1,
):
    """
    Read KARHU HDF5 dataset, interpolate profiles to a new grid,
    and write a new HDF5 dataset.

    Parameters
    ----------
    data_dir : str
        Directory containing input profiles.h5
    new_data_dir : str
        Output directory
    x_1 : ndarray
        New psi grid
    theta_1 : ndarray
        New theta grid
    """
    print(f"Creating new HDF5 dataset: {out_path}")

    n_profile_points = x_1.shape[0]
    n_boundary_points = theta_1.shape[0]

    with h5py.File(in_path, "r") as fin:
        # --------------------------------------------------
        # Select indices
        # --------------------------------------------------
        idx = np.arange(fin["profiles/cs"].shape[0])
        i_max = len(idx)
        print(f"Using {i_max} simulations")

        # --------------------------------------------------
        # Allocate new arrays
        # --------------------------------------------------
        p0_new = np.empty((i_max, n_profile_points))
        qs_new = np.empty((i_max, n_profile_points))
        rbphi_new = np.empty((i_max, n_profile_points))

        # vxvy_new = np.empty((i_max, 2, n_profile_points))
        boundary_polar_new = np.empty((i_max, 2, n_boundary_points))

        # --------------------------------------------------
        # Interpolation loop
        # --------------------------------------------------
        for j, i in enumerate(idx):
            # ---- vxvy (cartesian shape)
            # vx_0 = fin["profiles/vxvy"][i, 0, :]
            # vy_0 = fin["profiles/vxvy"][i, 1, :]
            # vy_1 = interpolate_psi_profile(vx_0, vy_0, vx_1)

            # vxvy_new[j, 0, :] = vx_1
            # vxvy_new[j, 1, :] = vy_1

            # ---- boundary (polar)
            theta_0 = fin["profiles/boundary_polar"][i, 1, :]
            rho_0 = fin["profiles/boundary_polar"][i, 0, :]
            rho_1 = interpolate_psi_profile(theta_0, rho_0, theta_1)

            boundary_polar_new[j, 1, :] = theta_1
            boundary_polar_new[j, 0, :] = rho_1

            # ---- profiles
            cs_0 = fin["profiles/cs"][i, :]

            p0_new[j, :] = interpolate_psi_profile(
                cs_0, fin["profiles/p0"][i, :], x_1
            )
            qs_new[j, :] = interpolate_psi_profile(
                cs_0, fin["profiles/qs"][i, :], x_1
            )
            rbphi_new[j, :] = interpolate_psi_profile(
                cs_0, fin["profiles/rbphi"][i, :], x_1
            )

        # --------------------------------------------------
        # Write new HDF5
        # --------------------------------------------------
        with h5py.File(out_path, "w") as fout:
            grp_profiles = fout.create_group("profiles")
            grp_scalars = fout.create_group("scalars")
            grp_mishka = fout.create_group("growthrates_mishka")
            grp_castor = fout.create_group("growthrates_castor")
            grp_karhu = fout.create_group("karhu")

            # Profiles
            # grp_profiles.create_dataset("cs", data=np.tile(x_1, (i_max, 1)))
            grp_profiles.create_dataset("p0", data=p0_new)
            grp_profiles.create_dataset("qs", data=qs_new)
            grp_profiles.create_dataset("rbphi", data=rbphi_new)
            # grp_profiles.create_dataset("vxvy", data=vxvy_new)
            grp_profiles.create_dataset(
                "boundary_polar", data=boundary_polar_new
            )
            grp_karhu.create_dataset("psin_axis", data=x_1)
            grp_karhu.create_dataset("theta_axis", data=theta_1)

            # Scalars (copied, filtered)
            for name in fin["scalars"].keys():
                grp_scalars.create_dataset(
                    name, data=fin["scalars"][name][idx]
                )

            # Growthrates (copied, filtered)
            gr_in = fin["growthrates_mishka/gamma"]
            gr_out = grp_mishka.create_dataset(
                "gamma2",
                shape=(i_max,),
                dtype=gr_in.dtype,
            )
            for j, i in enumerate(idx):
                gr_out[j] = gr_in[i]
            gr_in = fin["growthrates_mishka/ntor"]
            gr_out = grp_mishka.create_dataset(
                "ntor",
                shape=(i_max,),
                dtype=gr_in.dtype,
            )
            for j, i in enumerate(idx):
                gr_out[j] = gr_in[i]
            gr_in = fin["growthrates_castor/gamma"]
            gr_out = grp_castor.create_dataset(
                "gamma2",
                shape=(i_max,),
                dtype=gr_in.dtype,
            )
            for j, i in enumerate(idx):
                gr_out[j] = gr_in[i]
            gr_in = fin["growthrates_castor/ntor"]
            gr_out = grp_castor.create_dataset(
                "ntor",
                shape=(i_max,),
                dtype=gr_in.dtype,
            )
            for j, i in enumerate(idx):
                gr_out[j] = gr_in[i]

    print(f"New dataset written to: {out_path}")


def main():
    print(f"Start time: {datetime.now()}")
    # Create argument parser
    parser = argparse.ArgumentParser(description="Collect data into npy array data sets.")  # fmt: skip
    parser.add_argument("--db_fullprofiles", type=str, help="Path to the database .h5 containing the full profiles.")  # fmt: skip
    parser.add_argument("--db_dir", type=str, help="Path to the target database.")  # fmt: skip
    args = parser.parse_args()

    SAVE_PATH_FULL_PROFILES = args.db_fullprofiles
    DB_DIR = args.db_dir
    os.makedirs(DB_DIR, exist_ok=True)

    # Create scaled down data set of _ number of points
    SAVE_PATH_INTERPOLATED =  os.path.join(DB_DIR, "interpolated64.h5")
    n_profile_points = 64
    x_1 = np.linspace(1e-5, 1, n_profile_points) ** (1 / 4) # S=sqrt(psi)
    theta_1 = np.linspace(1e-5, 2*np.pi, n_profile_points*2)
    create_data_new_grid_h5(
        in_path=SAVE_PATH_FULL_PROFILES,
        out_path=SAVE_PATH_INTERPOLATED,
        x_1=x_1,
        theta_1=theta_1,
    )

    SAVE_PATH_INTERPOLATED =  os.path.join(DB_DIR, "interpolated128.h5")
    n_profile_points = 128
    x_1 = np.linspace(1e-5, 1, n_profile_points) ** (1 / 2) # S=sqrt(psi)
    theta_1 = np.linspace(1e-5, 2*np.pi, n_profile_points*2)
    create_data_new_grid_h5(
        in_path=SAVE_PATH_FULL_PROFILES,
        out_path=SAVE_PATH_INTERPOLATED,
        x_1=x_1,
        theta_1=theta_1,
    )

    print(f"End time: {datetime.now()}")


if __name__ == "__main__":
    main()
