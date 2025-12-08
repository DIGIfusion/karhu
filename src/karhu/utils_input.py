"""
Helper functions for creating model input from HELENA fort.10, fort.12, and fort.20 files.
"""
from __future__ import annotations 

from scipy.interpolate import interp1d
import numpy as np
import os
import math
import torch
import f90nml


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


def get_f12_data(filename, variables):
    """
    Read selected variables from HELENA fort.12 file.

    Args:
        filename (str): Path to fort.12 file.
        variables (list[str]): List of variable names to read.

    Supported variables:
        JS0, CS, QS, DQS_1, DQEC, DQS, CURJ, DJ0, DJE,
        NCHI, CHI, GEM11, GEM12, CPSURF, RADIUS, GEM33, RAXIS,
        P0, DP0, DPE, RBPHI, DRBPHI0, DRBPHIE, VX, VY, EPS, XOUT, YOUT
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Cannot find {filename}")

    lines = open(filename, "r").readlines()

    def read_n_lines(n, size):
        return math.ceil(size / n)

    data = {}

    JS0 = int(lines[0].split()[0])
    JS0_lines = read_n_lines(4, JS0)
    JS0_1_lines = read_n_lines(4, JS0 + 1)

    i = 1  # line counter

    def maybe_read(name, size):
        nonlocal i
        if name in variables:
            result = read_lines2(lines, i, i + size)
            i += size
            return result
        else:
            i += size
            return None

    if 'JS0' in variables:
        data['JS0'] = np.array(JS0, dtype=np.int32)

    if 'CS' in variables:
        data['CS'] = maybe_read('CS', JS0_1_lines)
    else:
        i += JS0_1_lines

    if 'QS' in variables:
        data['QS'] = maybe_read('QS', JS0_1_lines)
    else:
        i += JS0_1_lines

    if {'DQS_1', 'DQEC'} & set(variables):
        DQS_line = lines[i].split()
        if 'DQS_1' in variables:
            data['DQS_1'] = np.array(float(DQS_line[0]), dtype=np.float32)
        if 'DQEC' in variables:
            data['DQEC'] = np.array(float(DQS_line[1]), dtype=np.float32)
    i += 1

    if 'DQS' in variables:
        data['DQS'] = maybe_read('DQS', JS0_lines)
    else:
        i += JS0_lines

    if 'CURJ' in variables:
        data['CURJ'] = maybe_read('CURJ', JS0_1_lines)
    else:
        i += JS0_1_lines

    if {'DJ0', 'DJE'} & set(variables):
        DJ_line = lines[i].split()
        if 'DJ0' in variables:
            data['DJ0'] = np.array(float(DJ_line[0]), dtype=np.float32)
        if 'DJE' in variables:
            data['DJE'] = np.array(float(DJ_line[1]), dtype=np.float32)
    i += 1

    NCHI = int(lines[i].split()[0])
    NCHI_1_lines = read_n_lines(4, NCHI)
    NCHI_JS0_lines = read_n_lines(4, NCHI * (JS0 + 1) - (NCHI + 1))
    i += 1

    if 'NCHI' in variables:
        data['NCHI'] = np.array(NCHI, dtype=np.int32)

    for var, lines_needed in [
        ('CHI', NCHI_1_lines), ('GEM11', NCHI_JS0_lines), ('GEM12', NCHI_JS0_lines)
    ]:
        if var in variables:
            data[var] = maybe_read(var, lines_needed)
        else:
            i += lines_needed

    if {'CPSURF', 'RADIUS'} & set(variables):
        line = lines[i].split()
        if 'CPSURF' in variables:
            data['CPSURF'] = np.array(float(line[0]), dtype=np.float32)
        if 'RADIUS' in variables:
            data['RADIUS'] = np.array(float(line[1]), dtype=np.float32)
    i += 1

    if 'GEM33' in variables:
        data['GEM33'] = maybe_read('GEM33', NCHI_JS0_lines)
    else:
        i += NCHI_JS0_lines

    if 'RAXIS' in variables:
        data['RAXIS'] = np.array(float(lines[i].split()[0]), dtype=np.float32)
    i += 1

    for var in ['P0', 'RBPHI']:
        if var in variables:
            data[var] = maybe_read(var, JS0_1_lines)
        else:
            i += JS0_1_lines

        if var == 'P0' and {'DP0', 'DPE'} & set(variables):
            line = lines[i].split()
            if 'DP0' in variables:
                data['DP0'] = np.array(float(line[0]), dtype=np.float32)
            if 'DPE' in variables:
                data['DPE'] = np.array(float(line[1]), dtype=np.float32)
        if var == 'RBPHI' and {'DRBPHI0', 'DRBPHIE'} & set(variables):
            line = lines[i].split()
            if 'DRBPHI0' in variables:
                data['DRBPHI0'] = np.array(float(line[0]), dtype=np.float32)
            if 'DRBPHIE' in variables:
                data['DRBPHIE'] = np.array(float(line[1]), dtype=np.float32)
        i += 1

    for var in ['VX', 'VY']:
        if var in variables:
            data[var] = maybe_read(var, NCHI_1_lines)
        else:
            i += NCHI_1_lines

    if 'EPS' in variables:
        data['EPS'] = np.array(float(lines[i].split()[0]), dtype=np.float32)
    i += 1

    for var in ['XOUT', 'YOUT']:
        if var in variables:
            data[var] = maybe_read(var, NCHI_JS0_lines)
        else:
            i += NCHI_JS0_lines

    return data


def read_fort20_beta_section(filename: str):
    """
    ***************************************
    MAGNETIC AXIS :   0.01908  0.00000
    POLOIDAL BETA :   0.1198E+00
    TOROIDAL BETA :   0.3802E-02
    BETA STAR     :   0.4250E-02
    NORM. BETA    :   0.00335
    TOTAL CURRENT :   0.1428E+01
    TOTAL AREA    :   0.5115E+01
    TOTAL VOLUME  :   0.3110E+02
    INT. INDUCTANCE :  0.685990E+00
    POL. FLUX     :   0.2130E+01
    A,B,C         :   0.4176E+01  0.1522E-01  0.1000E+01
    ***************************************
    """
    (
        betan,
        betap,
        total_current,
        total_area,
        total_volume,
        beta_tor,
        beta_star,
        helenaBetap,
        b_last_round,
        radius,
        B0,
    ) = (None, None, None, None, None, None, None, None, None, None, None)

    file = open(filename, "r")
    lines = file.readlines()
    for line in lines:
        # line = file.readline()
        if line.find("NORM. BETA") > -1:
            spl = line.split(":")
            betan = float(spl[1]) * 100
        if line.find("POLOIDAL BETA") > -1:
            spl = line.split(":")
            betap = float(spl[1])
        if line.find("TOTAL CURRENT") > -1:
            spl = line.split(":")
            total_current = float(spl[1])
        if line.find("TOTAL AREA") > -1:
            spl = line.split(":")
            total_area = float(spl[1])
        if line.find("TOTAL VOLUME") > -1:
            spl = line.split(":")
            total_volume = float(spl[1])
        if line.find("TOROIDAL BETA") > -1:
            spl = line.split(":")
            beta_tor = float(spl[1])
        if line.find("BETA STAR") > -1:
            spl = line.split(":")
            beta_star = float(spl[1])
        if line.find("PED. BETAPOL") > -1:
            spl = line.split(":")
            helenaBetap = float(spl[1])
        if line.find("A,B,C") > -1:
            spl = line.split(":")
            sp2 = spl[1].split()
            b_last_round = float(sp2[1])
        if line.find("RADIUS") > -1:
            spl = line.split(":")
            radius = float(spl[1])
        if line.find("B0") > -1:
            spl = line.split(":")
            B0 = float(spl[1])
            break
    file.close()
    return (
        betan,
        betap,
        total_current,
        total_area,
        total_volume,
        beta_tor,
        beta_star,
        helenaBetap,
        b_last_round,
        radius,
        B0,
    )


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

def get_eps_from_f20(filename_f20):
    with open(filename_f20, 'r') as file:
        lines = file.readlines()

    for line in lines:
        if "EPS" in line:
            break
    """ The line we are looking for
        $PHYS     EPS  =  0.319, ALFA =  2.356, B =  0.005, C =  1.000,
    """
    line = line.lstrip()[4:]
    line = line.strip()
    eps, *_ = line.split(",")
    eps     = float(eps.split("="))
    return eps

def get_model_input(filename_f10, filename_f12, filename_f20, x_1, n_profile_points=64, ):
    """
    Get the model input from the HELENA fort.10, fort.12, and fort.20 files.
    Args:
        filename_f10 (str): Path to the fort.10 file.
        filename_f12 (str): Path to the fort.12 file.
        filename_f20 (str): Path to the fort.20 file.
        x_1 (np.ndarray): New locations where you want to interpolate y_0.
        n_profile_points (int): Number of points in the new profile.
    Returns:
        list: List of tensors containing the model input.
    """
    helena_input = f90nml.read(filename_f10)
    out = get_f12_data(
        filename_f12, variables=["CS", "QS", "RADIUS", "P0", "RBPHI", "VX", "VY"])

    CS, P, RBPHI, QS = out["CS"], out["P0"], out["RBPHI"], out["QS"]
    VX, VY = out["VX"], out["VY"]
    RADIUS = out["RADIUS"]

    (_, _, _, _, _, _, _, _, _, _, B0,) = read_fort20_beta_section(filename_f20)

    # Get scaling parameters
    epsilon = helena_input["phys"]["eps"]  # inverse aspect ratio
    R_vac = helena_input["phys"]["rvac"]
    # minor_radius = epsilon * R_vac
    B_vac = helena_input["phys"]["bvac"]
    R_mag = (epsilon / RADIUS) * R_vac
    B_mag = B_vac / B0

    # New grid
    vx_1 = np.linspace(-0.999, 0.999, n_profile_points)
    x_0 = CS
    p_1 = interpolate_psi_profile(x_0, P, x_1)
    qs_1 = interpolate_psi_profile(x_0, QS, x_1)
    rbphi_1 = interpolate_psi_profile(x_0, RBPHI, x_1)
    vy_1 = interpolate_psi_profile(x_0=VX, y_0=VY, x_1=vx_1)

    return [
        torch.tensor(p_1, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
        torch.tensor(qs_1, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
        torch.tensor(rbphi_1, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
        torch.tensor(vy_1, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
        torch.tensor(B_mag, dtype=torch.float32).unsqueeze(0),
        torch.tensor(R_mag, dtype=torch.float32).unsqueeze(0),
    ]



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
