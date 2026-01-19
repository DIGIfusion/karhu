import os 
import math 

import numpy as np 
import f90nml 
import torch

from karhu.utils_input import interpolate_profile
from karhu.common import get_polar_from_rz


def read_fort20_beta_section(filename: str) -> tuple[float]:
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
    eps = float(eps.split("="))
    return eps


def load_from_helena(helena_directory: str, karhu_psin_axis, karhu_theta_axis,) -> list[np.ndarray]:
    """
    Prepares the inputs to KARHU. 
    - Reads relevant data from HELENA
    - Interpolates onto the common axis (psi/rbndry) used for KARHU.
    - Converts data to tensors
    
    Profiles come from mapping file (fort.12) --> input into MISHKA
    They are normalised such that p(psi=0.0) = 1,

    Parameters
    ----------
    helena_directory : str
        Path to the directory where fort.10/fort.12/fort.20 should be

    psin_axis : sequence of float
        List-like object defining the normalized poloidal flux grid.

    theta_axis : sequence of float
        List-like object defining the poloidal angle grid in radians.

    Returns
    -------
    x : list of torch.Tensor
        List of tensors formatted for KARHU model input.
    """
    fname_f10: str = os.path.join(helena_directory, "fort.10")
    fname_f12: str = os.path.join(helena_directory, "fort.12")
    fname_f20: str = os.path.join(helena_directory, "fort.20")

    f12data = get_f12_data(fname_f12, variables=["CS", "QS", "RADIUS", "P0", "RBPHI", "VX", "VY"])
    f10data = f90nml.read(fname_f10)

    RADIUS: float = f12data["RADIUS"]                             # geometric axis minor radius?? 
    # rest are numpy arrays
    P, RBPHI, QS = f12data["P0"], f12data["RBPHI"], f12data["QS"] # normalised profiles
    CS = f12data["CS"]**2                                         # Helena uses sqrt(psi) grid, i.e., CS**2 = PSI, CS**4 = PSI**2,
    rbdry, zbdry = f12data["VX"], f12data["VY"]                 # Boundary in R, Z

    # geometric axis magnetic field strength
    *_, B_0H = read_fort20_beta_section(fname_f20)

    epsilon = f10data["phys"]["eps"]  # inv. aspect ratio
    R_vac   = f10data["phys"]["rvac"] # major radius in vaccuum
    B_vac   = f10data["phys"]["bvac"]
    R_mag   = (epsilon / RADIUS) * R_vac
    B_mag   = B_vac / B_0H

    pressure_karhu = P
    rbphi_karhu    = RBPHI
    q_karhu = abs(QS)

    # Interpolate the profiles to the axes used in model
    pressure_karhu = interpolate_profile(CS, pressure_karhu, karhu_psin_axis)
    rbphi_karhu = interpolate_profile(CS, rbphi_karhu, karhu_psin_axis)
    q_karhu = interpolate_profile(CS, q_karhu, karhu_psin_axis)

    # Construct boundary in polar coordinates
    symmetric = f10data["shape"]["ias"] ==  0
    rhobndry, thetabndry = get_polar_from_rz(r_vals=rbdry, z_vals=zbdry, symmetric=symmetric)
    rhobndry_karhu = interpolate_profile(x_0=thetabndry, y_0=rhobndry, x_1=karhu_theta_axis)

    x = [torch.tensor(pressure_karhu, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
         torch.tensor(q_karhu, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
         torch.tensor(rbphi_karhu, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
         torch.tensor(rhobndry_karhu, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
         torch.tensor(B_mag, dtype=torch.float32).unsqueeze(0),
         torch.tensor(R_mag, dtype=torch.float32).unsqueeze(0),
    ]
    return x
