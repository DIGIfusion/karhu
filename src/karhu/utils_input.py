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


def convert_profiles_si_to_dimensionless(pressure, rbphi, rbdry, zbdry,
                                         radius: float, R_mag: float, eps: float, B_mag: float):
    """
    KARHU/MISHKA dimensionless normalisation from SI units
    KARHU_PRESSURE = PRESSURE_SI / (B_magaxis**2 / mu_0)
    KARHU_RBPHI    = RBPHI_SI    / (R_magaxis * B_magaxis)
    KARHU_RBDRY    = RBDRY_SI    / (radius * R_magaxis) - 1.0 / eps
    KARHU_ZBDRY    = ZBDRY_SI    / (radius * R_magaxis)
    KARHU_Q        = Q

    where
        PRESSURE_SI [N / m^2] plasma pressure          (defined on PSIN)
        RBPHI_SI    [Tm]      poloidal flux function   (defined on PSIN)
        Q           []        q-profile, safety factor (defined on PSIN)
        RBDRY_SI    [m]       R/X coordinates of plasma boundary
        ZBDRY_SI    [m]       Z/Y coordinates of plasma boundary
        B_magaxis   [T]       magnetic field strength at the magnetic axis
        R_magaxis   [m]       major radius at the magnetic axis
        eps         [-]       inverse aspect ratio of the plasma boundary
        radius      [-]       dimensionless value, used to scale R/X boundary between -1 and 1
                                is equal to eps * R_geom / R_magaxis,  
                                where R_geom is the geometric axis
    """
    pressure_karhu = pressure / (B_mag**2 / MU_0)
    rbphi_karhu    = rbphi     / (R_mag * B_mag)
    rbndry_karhu   = rbdry    / (radius * R_mag) - 1.0 / eps
    zbndry_karhu   = zbdry    / (radius * R_mag)
    return pressure_karhu, rbphi_karhu, rbndry_karhu, zbndry_karhu
