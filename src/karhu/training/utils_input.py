"""
Helper functions for creating model input from HELENA fort.10, fort.12, and fort.20 files.
"""

from scipy.interpolate import interp1d
import numpy as np
import os
import math
import torch
import f90nml

from torch.utils.data import random_split, ConcatDataset, DataLoader


def interpolate_psi_profile(x_0, y_0, x_1):
    """
    y_0 is the values at positions x_0
    x_0 is the corresponding locations between 0 and 1
    x_1 is the new locations where you want to interpolate y_0
    """

    interpolation_function = interp1d(x_0, y_0, kind="linear")

    # Use the interpolation function to find y_1 at new x_1 locations
    y_1 = interpolation_function(x_1)

    # y_1 now contains the scaled down values corresponding to the new locations x_1
    return y_1


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

def read_lines2(lines, start, end):
    """    Read lines from a list of strings and convert them to a numpy array of floats.
    Args:
        lines (list[str]): List of strings representing lines from a file.
        start (int): Starting index of the lines to read.
        end (int): Ending index of the lines to read.
    Returns:
        np.ndarray: Numpy array of floats containing the values from the specified lines.
    """
    return np.array([float(x) for line in lines[start:end] for x in line.split()], dtype=np.float32)


def minmax(data, scaler_min, scaler_max):
    """
    Scale data to the range [0, 1] using min-max scaling.
    Args:
        data (np.ndarray): Data to be scaled.
        scaler_min (float): Minimum value of the scaler.
        scaler_max (float): Maximum value of the scaler.
    Returns:
        np.ndarray: Scaled data in the range [0, 1].
    """
    return (data - scaler_min) / (scaler_max - scaler_min)


def descale_minmax(scaled_data, scaler_min, scaler_max):
    """
    Reverse min-max scaling to get the original data.
    Args:
        scaled_data (np.ndarray): Scaled data in the range [0, 1].
        scaler_min (float): Minimum value of the scaler.
        scaler_max (float): Maximum value of the scaler.
    Returns:
        np.ndarray: Original data before scaling.
    """
    return scaled_data * (scaler_max - scaler_min) + scaler_min


def scale_model_input(x, scaling_params):
    """
    Scale the model input using min-max scaling.
    Args:
        x (list): List of input features [p, qs, rbphi, vy, B_mag, R_mag].
        scaling_params (dict): Dictionary containing scaling parameters.
    Returns:
        list: Scaled input features.
    """
    p, qs, rbphi, vy, B_mag, R_mag = x[0], x[1], x[2], x[3], x[4], x[5]

    # Scale input
    p = minmax(p, scaling_params["p"][0], scaling_params["p"][1])
    qs = minmax(qs, scaling_params["qs"][0], scaling_params["qs"][1])
    rbphi = minmax(rbphi, scaling_params["rbphi"][0], scaling_params["rbphi"][1])
    vy = minmax(vy, scaling_params["shape"][0], scaling_params["shape"][1])
    B_mag = minmax(B_mag, scaling_params["b_mag"][0], scaling_params["b_mag"][1])
    R_mag = minmax(R_mag, scaling_params["r_mag"][0], scaling_params["r_mag"][1])
    return [p, qs, rbphi, vy, B_mag, R_mag]


def scale_model_output(y, scaling_params):
    try:
        y = y.item()
    except Exception:
        pass
    growthrate = descale_minmax(y, scaling_params["growthrate"][0], scaling_params["growthrate"][1])
    return growthrate
