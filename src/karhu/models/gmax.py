import os
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d

# from torch.utils.data import Dataset
import h5py

from karhu.training.train import DatasetEquilibriumGmax
# from karhu.training.utils_plotting import get_regression_scores


logger = logging.getLogger(__name__)


def load_model(model_dir: str) -> tuple[torch.nn.Module, dict[str, np.ndarray]]:
    """ The model directory should contain
    - model.pt containing the weights
    - model_config.json containing the scaling parameters and interpolation
      axes for the inputs and outputs
    """

    # Load model config
    with open(
        os.path.join(model_dir, "model_config.json",),
        "r", encoding="utf-8",
    ) as f:
        model_config = json.load(f)

    # Load model
    model = GMaxPredictor(model_config)
    model.load_state_dict(
        torch.load(os.path.join(model_dir, "model.pt"), weights_only=True,)
    )
    model.eval()
    return model, model_config


def load_ensemble_model(ensemble_dir: str) -> tuple[list[torch.nn.Module], dict[str, np.ndarray]]:
    """ The model directory should contain
    - model.pt containing the weights
    - model_config.json containing the scaling parameters and interpolation
      axes for the inputs and outputs
    """
    # Load model config
    with open(
        os.path.join(ensemble_dir, "model_config.json",),
        "r", encoding="utf-8",
    ) as f:
        model_config = json.load(f)

    models = []
    model_dirs = [
        d for d in os.listdir(ensemble_dir)
        if os.path.isdir(os.path.join(ensemble_dir, d))
    ]

    for model_dir in model_dirs:

        # Load model
        model = GMaxPredictor(model_config)
        model.load_state_dict(
            torch.load(
                os.path.join(ensemble_dir, model_dir, "model.pt"),
                weights_only=True,)
        )
        model.eval()
        models.append(model)

    return models, model_config


def get_ensemble_prediction(models: list, model_inputs):
    """
    Given a list of models and a set of inputs, get the mean prediction
    and the standard deviation of the preditions. Note that the output
    is normalized and needs to be scaled back.
    """
    ensemble_preds = []

    for model in models:
        with torch.no_grad():
            y_pred_norm = model(*model_inputs)
        ensemble_preds.append(y_pred_norm)

    ensemble_preds = np.stack(ensemble_preds, axis=0)
    y_pred_mean = ensemble_preds.mean(axis=0)
    y_pred_std = ensemble_preds.std(axis=0)
    return y_pred_mean, y_pred_std


class GMaxPredictor(nn.Module):
    """
    GMaxPredictor model for predicting gmax from input sequences.
    The model consists of several convolutional layers followed by fully connected layers.
    The input consists of 4 sequences and 2 additional features (b_mag and r_mag).
    """

    def __init__(self, model_config: dict,):
        super().__init__()

        self.conv_input_sizes = model_config["conv_input_sizes"]
        self.scalar_inputs = model_config["scalar_inputs"]
        self.out_channels = model_config["out_channels"]
        self.conv_kernel_sizes = model_config["conv_kernel_sizes"]
        self.pool_kernel_size = model_config["pool_kernel_size"]
        self.fc_hidden_dims = model_config["fc_hidden_dims"]
        self.pool_kernel_size = model_config["pool_kernel_size"]
        self.pool = nn.MaxPool1d(kernel_size=self.pool_kernel_size, stride=2, padding=0)

        # --- Dropout layers ---
        self.conv_dropout = nn.Dropout1d(p=model_config["conv_dropout"])
        self.fc_dropout = nn.Dropout(p=model_config["fc_dropout"])

        # First 4 input conv layers (1 channel in)
        self.input_convs = nn.ModuleList(
            [
                nn.Conv1d(
                    1,
                    self.out_channels,
                    kernel_size=self.conv_kernel_sizes[0],
                    stride=1,
                    padding=self.conv_kernel_sizes[0] // 2,
                )
                for _ in range(4)
            ]
        )

        # Next 2 conv layers repeated 2 times each
        self.conv1s = nn.ModuleList(
            [
                nn.Conv1d(
                    self.out_channels,
                    self.out_channels,
                    kernel_size=self.conv_kernel_sizes[1],
                    stride=1,
                    padding=self.conv_kernel_sizes[1] // 2,
                )
                for _ in range(4)
            ]
        )

        self.conv2s = nn.ModuleList(
            [
                nn.Conv1d(
                    self.out_channels,
                    self.out_channels,
                    kernel_size=self.conv_kernel_sizes[2],
                    stride=1,
                    padding=self.conv_kernel_sizes[2] // 2,
                )
                for _ in range(4)
            ]
        )

        # Compute output size after 3 poolings
        self.flatten_dims = []
        for size in self.conv_input_sizes:
            conv_output = size // (2**3)
            self.flatten_dims.append(self.out_channels * conv_output)

        self.total_flatten_dim = sum(self.flatten_dims)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.total_flatten_dim + self.scalar_inputs, self.fc_hidden_dims[0])
        self.fc2 = nn.Linear(self.fc_hidden_dims[0], self.fc_hidden_dims[1])
        self.fc3 = nn.Linear(self.fc_hidden_dims[1], 1)

        self.loss_fn = nn.MSELoss()

    def forward(self, input_p, input_qs, input_rbphi, input_shape, b_mag, r_mag):
        inputs = [input_p, input_qs, input_rbphi, input_shape]
        features = []
        scalars = torch.stack([b_mag, r_mag], dim=1) 

        for i in range(4):
            x = F.leaky_relu(self.input_convs[i](inputs[i]))
            x = self.pool(x)
            x = self.conv_dropout(x)

            x = F.leaky_relu(self.conv1s[i](x))
            x = self.pool(x)
            x = self.conv_dropout(x)

            x = F.leaky_relu(self.conv2s[i](x))
            x = self.pool(x)
            x = self.conv_dropout(x)

            features.append(torch.flatten(x, start_dim=1))

        x = torch.cat(features, dim=1)

        # Concatenate scalars
        scalars = scalars.view(scalars.size(0), -1)
        x = torch.cat([x, scalars], dim=1)

        x = F.leaky_relu(self.fc1(x))
        x = self.fc_dropout(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.fc_dropout(x)
        x = self.fc3(x)

        return x

    def get_input_from_batch_data(self, data, device):
        """
        data:
            Batch from DataLoader
        decive:
            cpu or gpu
        """
        input_p, input_qs, input_rbphi, input_shape, b_mag, r_mag, labels = (
            data["p"].to(device),
            data["qs"].to(device),
            data["rbphi"].to(device),
            data["shape"].to(device),
            data["b_mag"].to(device),
            data["r_mag"].to(device),
            data["growthrate"].to(device),
        )
        return input_p, input_qs, input_rbphi, input_shape, b_mag, r_mag, labels


def setup_dataset(dir_path: str):
    """Creates a DatasetEquilibriumGmax with the files from the given path.
    Detects and removes samples containing NaNs before scaling.
    """
    h5_path_interp = dir_path
    with h5py.File(h5_path_interp, "r") as f:
        qs = f["profiles/qs"][:]
        p0 = f["profiles/p0"][:]
        rbphi = f["profiles/rbphi"][:]
        boundary = f["profiles/boundary_polar"][:]
        gamma = f["scalars/max_gr_mishka"][:]
        rmag = f["scalars/rmag"][:]
        bmag = f["scalars/bmag"][:]
        karhu_psin_axis = f["karhu/psin_axis"][:]
        karhu_theta_axis = f["karhu/theta_axis"][:]

    # --- Build tensors ---
    features = {
        "p": torch.tensor(np.expand_dims(p0.astype(np.float32), axis=1)),
        "qs": torch.tensor(np.expand_dims(qs.astype(np.float32), axis=1)),
        "rbphi": torch.tensor(np.expand_dims(rbphi.astype(np.float32), axis=1)),
        "shape": torch.tensor(np.expand_dims(boundary.astype(np.float32)[:, 0, :], axis=1)),
        "growthrate": torch.tensor(np.expand_dims(gamma.astype(np.float32), axis=1)),
        "r_mag": torch.tensor(np.expand_dims(rmag.astype(np.float32), axis=1)),
        "b_mag": torch.tensor(np.expand_dims(bmag.astype(np.float32), axis=1)),
    }

    # --- Detect NaNs per feature ---
    n_samples = next(iter(features.values())).shape[0]
    valid_mask = torch.ones(n_samples, dtype=torch.bool)

    for name, data in features.items():
        # Collapse all non-batch dimensions
        nan_mask = torch.isnan(data).view(n_samples, -1).any(dim=1)

        if nan_mask.any():
            bad_indices = torch.where(nan_mask)[0].tolist()
            logger.info(f"[NaN detected] Feature '{name}' has NaNs at indices: {bad_indices[:10]} "
                  f"{'...' if len(bad_indices) > 10 else ''}")

        valid_mask &= ~nan_mask

    n_removed = (~valid_mask).sum().item()
    if n_removed > 0:
        logger.info(f"[Dataset cleanup] Removing {n_removed} / {n_samples} samples due to NaNs")

    # --- Filter all features consistently ---
    features = {
        name: data[valid_mask]
        for name, data in features.items()
    }

    # --- Safe scaling params (no NaNs now) ---
    # scaling_params = {
    #     name: (torch.min(data).item(), torch.max(data).item())
    #     for name, data in features.items()
    # }
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


def setup_multiple_datasets(data_paths: list):
    datasets = []
    for data_path in data_paths:
        dataset, data_config = setup_dataset(data_path)
        datasets.append((dataset, data_config))
    return datasets


def setup_dataset_from_samples(dir_path: str):
    """Creates a DatasetEquilibriumGmax from all .h5 files in a directory.

    Each .h5 file is expected to contain one sample under the
    ``equilibrium/profiles`` group.

    Detects and removes samples containing NaNs before scaling.
    """
    logger.info("Setting up dataset from samples...")
    h5_files = sorted(Path(dir_path).glob("*.h5"))

    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in directory: {dir_path}")

    # --- Collect samples from all H5 files ---
    qs_list = []
    p0_list = []
    rbphi_list = []
    boundary_list = []
    gamma_list = []
    rmag_list = []
    bmag_list = []

    n_profile_points = 64
    karhu_psin_axis = np.linspace(1e-5, 1, n_profile_points) ** (1 / 4)  # S=sqrt(psi)
    karhu_theta_axis = np.linspace(1e-5, 2*np.pi, n_profile_points*2)

    for h5_path in h5_files:
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
            rho_1 = interpolate_psi_profile(theta_0, rho_0, karhu_theta_axis)

            qs_interp = interpolate_psi_profile(cs, qs, karhu_psin_axis)
            p0_interp = interpolate_psi_profile(cs, p0, karhu_psin_axis)
            rbphi_interp = interpolate_psi_profile(cs, rbphi, karhu_psin_axis)

            boundary_list.append(rho_1)
            qs_list.append(qs_interp)
            p0_list.append(p0_interp)
            rbphi_list.append(rbphi_interp)

            # Get max gamma
            sample_growthrates = [0.0]
            for ntor in list(f['mishka'].keys()):
                sample_growthrates.append(
                    f['mishka'][ntor]['output']['ew'][()])
            gamma_list.append(np.max(sample_growthrates)**(0.5))

            # Calculate rmag and bmag
            rvac = f["equilibrium/input/phys/rvac"][()]
            bvac = f["equilibrium/input/phys/bvac"][()]
            epsilon = f["equilibrium/input/phys/eps"][()]
            radius = f["equilibrium/profiles/RADIUS"][()]
            b0 = f["equilibrium/scalars/b0"][()]
            rmag = (epsilon / radius) * rvac
            bmag = bvac / b0
            rmag_list.append(rmag)
            bmag_list.append(bmag)

    # --- Combine all samples ---
    qs = np.array(qs_list)
    p0 = np.array(p0_list)
    rbphi = np.array(rbphi_list)
    boundary = np.array(boundary_list)
    gamma = np.array(gamma_list)
    rmag = np.array(rmag_list)
    bmag = np.array(bmag_list)

    logger.debug(f"Dataset: {qs.shape, p0.shape, rbphi.shape, boundary.shape, gamma.shape, rmag.shape, bmag.shape}")

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
        r_vals = np.concatenate([r_vals, r_mirror[1:]])  # avoid duplicate at midplane
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
