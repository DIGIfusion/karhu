"""
Helper functions for creating model input from HELENA fort.10, fort.12, and fort.20 files.
"""
from scipy.interpolate import interp1d
import numpy as np
import os
import math
import torch
import f90nml

MU_0 = 4E-7 * np.pi 


def interpolate_profile(x_0, y_0, x_1):
    """
    y_0 is the values at positions x_0
    x_0 is the corresponding locations between 0 and 1
    x_1 is the new locations where you want to interpolate y_0
    """
    # FIXME this seems dangerous
    interpolation_function = interp1d(x_0, y_0, kind="linear", fill_value='extrapolate')

    # Use the interpolation function to find y_1 at new x_1 locations
    y_1 = interpolation_function(x_1)

    # y_1 now contains the scaled down values corresponding to the new locations x_1
    return y_1


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


def scale_parameter(value, scaling_params):
    """Apply min-max scaling unless min and max are equal."""
    min_val, max_val = scaling_params

    if min_val == max_val:
        return value

    return minmax(value, min_val, max_val)


def scale_model_input(x, scaling_params):
    """
    Scale the model input using min-max scaling.

    Args:
        x (list): Input features [p, qs, rbphi, vy, B_mag, R_mag].
        scaling_params (dict): Dictionary containing scaling parameters.

    Returns:
        list: Scaled input features.
    """
    p, qs, rbphi, vy, B_mag, R_mag = x

    p = scale_parameter(p, scaling_params["p"])
    qs = scale_parameter(qs, scaling_params["qs"])
    rbphi = scale_parameter(rbphi, scaling_params["rbphi"])
    vy = scale_parameter(vy, scaling_params["shape"])
    B_mag = scale_parameter(B_mag, scaling_params["b_mag"])
    R_mag = scale_parameter(R_mag, scaling_params["r_mag"])

    return [p, qs, rbphi, vy, B_mag, R_mag]


def scale_model_input_eq(x, scaling_params):
    """
    Scale the equilibrium model input using min-max scaling.

    Args:
        x (list): Input features [ne, Te, Ti, vy, B_mag, R_mag, ip, beta_n, zeff].
        scaling_params (dict): Dictionary containing scaling parameters.

    Returns:
        list: Scaled input features.
    """
    ne, Te, Ti, vy, B_mag, R_mag, ip, beta_n, zeff = x

    ne = scale_parameter(ne, scaling_params["ne"])
    Te = scale_parameter(Te, scaling_params["Te"])
    Ti = scale_parameter(Ti, scaling_params["Ti"])
    vy = scale_parameter(vy, scaling_params["shape"])
    B_mag = scale_parameter(B_mag, scaling_params["b_mag"])
    R_mag = scale_parameter(R_mag, scaling_params["r_mag"])
    ip = scale_parameter(ip, scaling_params["ip"])
    beta_n = scale_parameter(beta_n, scaling_params["beta_n"])
    zeff = scale_parameter(zeff, scaling_params["zeff"])

    return [ne, Te, Ti, vy, B_mag, R_mag, ip, beta_n, zeff]


def scale_model_output(y, scaling_params):
    try:
        y = y.item()
    except Exception:
        pass
    growthrate = descale_minmax(y, scaling_params["growthrate"][0], scaling_params["growthrate"][1])
    return growthrate
