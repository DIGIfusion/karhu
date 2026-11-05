import os
import logging
import numpy as np
from pathlib import Path
import h5py
import torch
from torch.utils.data import Dataset, random_split
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


class DatasetEquilibriumGmax(Dataset):
    """ Extending the Dataset class. """

    def __init__(self, features):
        """Initialize the dataset with features.
        Args:
            features (dict): Dictionary of features where keys are feature
                names and values are numpy arrays.
        """
        self.features = features
        self.scaling_params = {}
        self.scaled = False
        self.transformed = False
        self.c = 0.1

    def set_scaling_params(self, scaling_params):
        """
        Set the scaling parameters from a dict.
        """
        self.scaling_params = scaling_params
        return

    def set_scaling_params_from_data(self):
        """
        Set the scaling parameters for each feature based on the data.
        This method calculates the min and max values for each feature
        and stores them in the scalers dictionary.
        """
        logger.info("Setting scaling params from data...")
        if self.scaled:
            logger.warning("WARNING: Profiles have already been scaled.")
        self.scaling_params = {
            name: (torch.min(data).item(), torch.max(data).item())
            for name, data in self.features.items()
        }
        logger.info(f"Scaling params: {self.scaling_params}")
        return

    def set_scaling_params_from_dataset(self, other):
        """Set the scaling parameters from another dataset.
        This method copies the scaling parameters from another dataset
        instance.
        Args:
            other (DatasetEquilibriumGmax): Another dataset instance
            from which to copy scaling parameters.
        """
        logger.info("Setting scaling params from another dataset...")
        self.scaling_params = other.scalers.copy()
        logger.info(f"Scaling params: {self.scaling_params}")

    def get_scaling_params(self):
        """Get the scaling parameters for each feature.
        Returns:
            dict: Dictionary of scaling parameters where keys are feature
            names and values are tuples of (min, max).
        """
        scaling_params = {}
        for name, scaler in self.scaling_params.items():
            scaling_params[f"{name}_min"] = scaler[0]
            scaling_params[f"{name}_max"] = scaler[1]
        return scaling_params

    def scale_data(self):
        """Perform min-max scaling on the data"""
        logger.info("Scaling data...")
        if not self.scaled:
            for name, data in self.features.items():
                min_, max_ = self.scaling_params[name]
                if max_ != min_:
                    self.features[name] = (data - min_) / (max_ - min_)
            self.scaled = True
        else:
            print("Data already scaled.")

    def descale_data(self):
        """Undo the minmax scaling on the data"""
        if self.scaled:
            for name, data in self.features.items():
                min_, max_ = self.scaling_params[name]
                if max_ != min_:
                    self.features[name] = data * (max_ - min_) + min_
            self.scaled = False
        else:
            print("Data is not scaled.")

    def set_zeros_to_negative(self, negative_value: float = -1.0):
        """Set all zeros in the labels of the dataset to a negative value"""
        self.features["gamma"][self.features["gamma"] == 0.0] = negative_value

    def _transform(self, y, c):
        return np.log1p(y/c)

    def _detransform(self, y, c):
        return np.expm1(y)*c

    def transform_labels(self, c=0.1):
        if self.transformed:
            c = self.c
            self.features["gamma"] = self._detransform(
                self.features["gamma"], c)
            self.transformed = False
        else:
            self.c = c
            self.features["gamma"] = self._transform(
                self.features["gamma"], c)
            self.transformed = True

    def minmax(self, data, scaler_min, scaler_max):
        """Perform min-max scaling on the data.
        Args:
            data (torch.Tensor): Data to be scaled.
            scaler_min (float): Minimum value for scaling.
            scaler_max (float): Maximum value for scaling.
        Returns:
            torch.Tensor: Scaled data.
        """
        return (data - scaler_min) / (scaler_max - scaler_min)

    def descale_minmax(self, scaled_data, scaler_min, scaler_max):
        """Undo the min-max scaling on the data.
        Args:
            scaled_data (torch.Tensor): Scaled data to be descaled.
            scaler_min (float): Minimum value used for scaling.
            scaler_max (float): Maximum value used for scaling.
        Returns:
            torch.Tensor: Descaled data.
        """
        return scaled_data * (scaler_max - scaler_min) + scaler_min

    def __getitem__(self, index):
        return {name: data[index] for name, data in self.features.items()}

    def __len__(self):
        return len(next(iter(self.features.values())))

    def get_features_description(self):
        """Get a description of the features in the dataset.
        Returns:
            str: Description of the features and their shapes.
        """
        desc = []
        desc.append("Features in dataset:")
        for key, value in self.features.items():
            desc.append(f"  {key}: {value.shape}")
        return "\n".join(desc)


def split_datasets(
    datasets,
    train_ratio=0.70,
    val_ratio=0.20,
    test_ratio=0.10,
    base_seed=42,
    ensemble_seed=0,
):
    train_sets = []
    val_sets = []
    test_sets = []

    for i, (dataset, _) in enumerate(datasets):
        train_data, val_data, test_data = split_dataset(
            dataset,
            train_ratio,
            val_ratio,
            test_ratio,
            base_seed,
            ensemble_seed + i
        )

        train_sets.append(train_data)
        val_sets.append(val_data)
        test_sets.append(test_data)

    return train_sets, val_sets, test_sets


def split_dataset(
    dataset,
    train_ratio=0.70,
    val_ratio=0.20,
    test_ratio=0.10,
    base_seed=42,
    ensemble_seed=None
):

    if ensemble_seed is None:
        ensemble_seed = base_seed

    n_total = len(dataset)
    n_test = int(test_ratio * n_total)
    n_trainval = n_total - n_test

    # --- Fixed test split (dataset-specific but reproducible) ---
    g_test = torch.Generator().manual_seed(base_seed)
    trainval_data, test_data = random_split(
        dataset,
        [n_trainval, n_test],
        generator=g_test,
    )

    # --- Train / val split (varies per ensemble member) ---
    n_train = int(train_ratio * n_total)
    n_val = n_trainval - n_train

    g_tv = torch.Generator().manual_seed(ensemble_seed)
    train_data, val_data = random_split(
        trainval_data,
        [n_train, n_val],
        generator=g_tv,
    )

    return train_data, val_data, test_data


def setup_dataset(
        dir_paths: list, dataset_mappings: dict, log_transformation_c=None):
    """Creates a DatasetEquilibriumGmax with the files from the given path.
    Detects and removes samples containing NaNs before scaling.
    """
    logger.debug(f"Number of datasets: {len(dir_paths)}")
    data = {key: [] for key in dataset_mappings}

    for dir_path in dir_paths:
        h5_paths = sorted(
            os.path.join(dir_path, filename)
            for filename in os.listdir(dir_path)
            if filename.endswith((".h5", ".hdf5"))
            and os.path.isfile(os.path.join(dir_path, filename))
        )
        logger.debug("Found %d HDF5 files in %s", len(h5_paths), dir_path)
        for h5_path in h5_paths:
            try:
                with h5py.File(h5_path, "r") as f:
                    for key, dataset_path in dataset_mappings.items():
                        data[key].append(f[dataset_path][:])
                    # Special case.
                    # Needs to be the same throughout the datasets.
                    karhu_psin_axis = f["karhu/psin_axis"][:]
                    karhu_theta_axis = f["karhu/theta_axis"][:]

            except (OSError, KeyError, ValueError) as e:
                logger.warning("Skipping %s: %s", h5_path, e)
    # Concatenate along the sample dimension
    data = {
        key: np.concatenate(arrays, axis=0)
        for key, arrays in data.items()
        if arrays
    }

    logger.info(
        f"Amount of unstable equilibria in dataset: "
        f"{np.sum(data['gamma']>0.03)}/{len(data['gamma'])} "
        f"= {np.sum(data['gamma']>0.03)/len(data['gamma'])*100:.2f}% ")

    # --- Build tensors ---
    features = {}
    for key, value in data.items():
        features[key] = torch.tensor(
            np.expand_dims(value.astype(np.float32), axis=1))
    logger.info(f"Features: {features.keys()}")

    # --- Detect NaNs per feature ---
    n_samples = next(iter(features.values())).shape[0]
    valid_mask = torch.ones(n_samples, dtype=torch.bool)

    for name, data in features.items():
        # Collapse all non-batch dimensions
        nan_mask = torch.isnan(data).view(n_samples, -1).any(dim=1)

        if nan_mask.any():
            bad_indices = torch.where(nan_mask)[0].tolist()
            logger.warning(
                f"[NaN detected] Feature '{name}' has NaNs at "
                + f"indices: {bad_indices[:10]} "
                + f"{'...' if len(bad_indices) > 10 else ''}")

        valid_mask &= ~nan_mask

    n_removed = (~valid_mask).sum().item()
    if n_removed > 0:
        logger.warning(
            f"[Dataset cleanup] Removing {n_removed} / {n_samples} "
            + "samples due to NaNs")

    # --- Filter all features consistently ---
    features = {
        name: data[valid_mask]
        for name, data in features.items()
    }

    # --- Safe scaling params (no NaNs now) ---
    scaling_params = {
        name: (torch.min(data).item(), torch.max(data).item())
        for name, data in features.items()
    }

    data_config = {
        "scaling_params": scaling_params,
        "karhu_psin_axis": karhu_psin_axis.tolist(),
        "karhu_theta_axis": karhu_theta_axis.tolist(),
        "n_removed_nan_samples": n_removed,
    }

    dataset = DatasetEquilibriumGmax(features)
    dataset.set_scaling_params(scaling_params)
    if log_transformation_c is not None:
        logger.debug(f"Log transforming gamma. C={log_transformation_c}.")
        data_config["log_transformation_c"] = log_transformation_c
        dataset.transform_labels(c=log_transformation_c)
    dataset.scale_data()
    return dataset, data_config


def setup_dataset_from_samples(dir_paths: list[str]):
    """Creates a DatasetEquilibriumGmax from all .h5 files in
    the given directories.

    Each .h5 file is expected to contain one sample under the
    ``equilibrium/profiles`` group.

    Detects and removes samples containing NaNs before scaling.
    """
    logger.info("Setting up dataset from samples...")

    h5_files = sorted(
        h5_file
        for dir_path in dir_paths
        for h5_file in Path(dir_path).glob("*.h5")
    )

    if not h5_files:
        raise FileNotFoundError(
            f"No .h5 files found in directories: {dir_paths}")

    # --- Collect samples from all H5 files ---
    qs_list = []
    p0_list = []
    rbphi_list = []
    boundary_list = []
    gamma_list = []
    rmag_list = []
    bmag_list = []

    n_profile_points = 64
    # Profiles initially given in S=sqrt(psi)
    karhu_psin_axis = np.linspace(1e-5, 1, n_profile_points) ** (1 / 4)
    karhu_theta_axis = np.linspace(1e-5, 2*np.pi, n_profile_points*2)

    for h5_path in h5_files:
        try:
            with h5py.File(h5_path, "r") as f:
                profiles = f["equilibrium/profiles"]

                qs = profiles["QS"][:]
                p0 = profiles["P0"][:]
                rbphi = profiles["RBPHI"][:]
                cs = profiles["CS"][:]
                vx = profiles["VX"][:]
                vy = profiles["VY"][:]

                # Interpolate to KARHU grid
                rho_0, theta_0 = get_polar_from_rz(vx, vy, symmetric=True)
                rho_1 = interpolate_psi_profile(
                    theta_0, rho_0, karhu_theta_axis)
                qs_interp = interpolate_psi_profile(cs, qs, karhu_psin_axis)
                p0_interp = interpolate_psi_profile(cs, p0, karhu_psin_axis)
                rbphi_interp = interpolate_psi_profile(
                    cs, rbphi, karhu_psin_axis)

                # Get max gamma
                sample_growthrates = [0.0]
                for ntor in list(f['mishka'].keys()):
                    sample_growthrates.append(
                        f['mishka'][ntor]['output']['ew'][()])

                # Calculate rmag and bmag
                rvac = f["equilibrium/input/phys/rvac"][()]
                bvac = f["equilibrium/input/phys/bvac"][()]
                epsilon = f["equilibrium/input/phys/eps"][()]
                radius = f["equilibrium/profiles/RADIUS"][()]
                b0 = f["equilibrium/scalars/b0"][()]
                rmag = (epsilon / radius) * rvac
                bmag = bvac / b0

                # Append to lists
                boundary_list.append(rho_1)
                qs_list.append(qs_interp)
                p0_list.append(p0_interp)
                rbphi_list.append(rbphi_interp)
                gamma_list.append(np.max(sample_growthrates)**(0.5))
                rmag_list.append(rmag)
                bmag_list.append(bmag)
        except Exception as e:
            logger.warning(f"{h5_path}: {e}. Skipping this file.")

    # --- Combine all samples ---
    qs = np.array(qs_list)
    p0 = np.array(p0_list)
    rbphi = np.array(rbphi_list)
    boundary = np.array(boundary_list)
    gamma = np.array(gamma_list)
    rmag = np.array(rmag_list)
    bmag = np.array(bmag_list)

    logger.debug(
        f"Dataset shapes: {qs.shape, p0.shape, rbphi.shape, boundary.shape}, "
        f"{gamma.shape, rmag.shape, bmag.shape}")

    # --- Build tensors ---
    features = {
        "p": torch.tensor(
            np.expand_dims(p0.astype(np.float32), axis=1)),
        "qs": torch.tensor(
            np.expand_dims(qs.astype(np.float32), axis=1)),
        "rbphi": torch.tensor(
            np.expand_dims(rbphi.astype(np.float32), axis=1)),
        "shape": torch.tensor(
            np.expand_dims(boundary.astype(np.float32), axis=1)
        ),
        "growthrate": torch.tensor(
            np.expand_dims(gamma.astype(np.float32), axis=1)
        ),
        "r_mag": torch.tensor(
            np.expand_dims(rmag.astype(np.float32), axis=1)
        ),
        "b_mag": torch.tensor(
            np.expand_dims(bmag.astype(np.float32), axis=1)
        ),
    }

    # --- Detect NaNs per feature ---
    n_samples = next(iter(features.values())).shape[0]
    valid_mask = torch.ones(n_samples, dtype=torch.bool)

    for name, data in features.items():
        # Collapse all non-batch dimensions
        nan_mask = torch.isnan(data).view(n_samples, -1).any(dim=1)

        if nan_mask.any():
            bad_indices = torch.where(nan_mask)[0].tolist()
            logger.info(
                f"[NaN detected] Feature '{name}' has NaNs at indices: "
                f"{bad_indices[:10]} "
                f"{'...' if len(bad_indices) > 10 else ''}"
            )

        valid_mask &= ~nan_mask

    n_removed = (~valid_mask).sum().item()

    if n_removed > 0:
        logger.info(
            f"[Dataset cleanup] Removing {n_removed} / "
            f"{n_samples} samples due to NaNs"
        )

    # --- Filter all features consistently ---
    features = {
        name: data[valid_mask]
        for name, data in features.items()
    }

    # --- Safe scaling params ---
    scaling_params = {
        name: (torch.min(data).item(), torch.max(data).item())
        for name, data in features.items()
    }

    data_config = {
        "scaling_params": scaling_params,
        "karhu_psin_axis": karhu_psin_axis.tolist(),
        "karhu_theta_axis": karhu_theta_axis.tolist(),
        "n_removed_nan_samples": n_removed,
    }

    dataset = DatasetEquilibriumGmax(features)
    dataset.set_scaling_params(scaling_params)
    dataset.scale_data()
    dataset.set_zeros_to_negative()

    return dataset, data_config


def get_polar_from_rz(r_vals, z_vals, symmetric=False):
    """
    Convert (R, Z) boundary coordinates to polar coordinates (rho, theta)
    relative to the boundary center (r0, z0).

    Handles both symmetric and asymmetric boundaries:

    - If symmetric (self.symmetric=True): input contains only the top half,
        and the function mirrors it to produce a full 0-2π contour.

    - If asymmetric: uses the full input directly.

    Args:
        r_vals (array_like): R (major radius) coordinates of the boundary.
        z_vals (array_like): Z (vertical) coordinates of the boundary.
        symmetric (bool): Whether the boundary is symmetric.

    Returns:
        tuple: (rho, theta) where rho is normalized radius and theta is
        poloidal angle.
    """
    r_vals = np.asarray(r_vals)
    z_vals = np.asarray(z_vals)
    r0 = (max(r_vals) + min(r_vals)) / 2
    ind = r_vals.argmax()
    z0 = 0 if symmetric else z_vals[ind]

    # Minor radius:
    amin = (max(r_vals) - min(r_vals)) / 2

    # Mirror the top half if the plasma is symmetric
    if symmetric:
        # Mirror across the midplane (z0)
        r_mirror = np.copy(r_vals[::-1])
        z_mirror = 2 * z0 - z_vals[::-1]

        # Combine top (input) and mirrored bottom
        # To avoid duplicate at midplane
        r_vals = np.concatenate([r_vals, r_mirror[1:]])
        z_vals = np.concatenate([z_vals, z_mirror[1:]])

    # Compute normalized radius and poloidal angle
    rho = np.sqrt((r_vals - r0)**2 + (z_vals - z0)**2) / amin
    theta = np.arctan2(z_vals - z0, r_vals - r0)

    # Convert θ range from (-π, π] → [0, 2π)
    theta = np.mod(theta, 2 * np.pi)

    # Sort points by increasing θ to ensure continuous boundary
    # order = np.argsort(theta)
    # return rho[order], theta[order]
    return rho, theta


def interpolate_psi_profile(x_0, y_0, x_1):
    """
    y_0 is the values at positions x_0
    x_0 is the corresponding locations between 0 and 1
    x_1 is the new locations where you want to interpolate y_0
    """

    interpolation_function = interp1d(
        x_0, y_0, kind="linear", fill_value='extrapolate')

    # Use the interpolation function to find y_1 at new x_1 locations
    y_1 = interpolation_function(x_1)

    # y_1 now contains the scaled down values corresponding to
    # the new locations x_1
    return y_1
