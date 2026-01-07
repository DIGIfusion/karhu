import numpy as np
from freeqdsk import geqdsk
from karhu.common import convert_profiles_si_to_dimensionless, get_polar_from_rz
from karhu.utils_input import interpolate_profile
import torch


def load_from_eqdsk(eqdskpath: str, karhu_psin_axis, karhu_theta_axis,):
    """
    Load an EQDSK equilibrium file and convert profiles into KARHU model inputs.

    Parameters
    ----------
    eqdskpath : str
        Path to the EQDSK equilibrium file.

    psin_axis : sequence of float
        List-like object defining the normalized poloidal flux grid.

    theta_axis : sequence of float
        List-like object defining the poloidal angle grid in radians.

    Returns
    -------
    x : list of torch.Tensor
        List of tensors formatted for KARHU model input.
    """
    with open(eqdskpath, "r") as f:
        eqdsk = geqdsk.read(f)
    psin1d = np.linspace(0, 1.0, eqdsk.nx)
    R_mag  = eqdsk.rmagx
    B_mag  = eqdsk.fpol[0] / eqdsk.rmagx

    R_geom = (eqdsk.rbdry.max() + eqdsk.rbdry.min()) / 2.0
    a_geom = (eqdsk.rbdry.max() - eqdsk.rbdry.min()) / 2.0
    eps    = a_geom / R_geom

    radius = eps * R_geom / R_mag
    pressure_karhu, rbphi_karhu, rbndry_karhu, zbndry_karhu = convert_profiles_si_to_dimensionless(eqdsk.pressure, eqdsk.fpol, eqdsk.rbdry, eqdsk.zbdry,
                                                                                                   radius, R_mag, eps, B_mag, )
    q_karhu = eqdsk.qpsi

    # Interpolate the profiles to the axes used in model
    pressure_karhu = interpolate_profile(psin1d, pressure_karhu, karhu_psin_axis)
    rbphi_karhu = interpolate_profile(psin1d, rbphi_karhu, karhu_psin_axis)
    q_karhu = interpolate_profile(psin1d, q_karhu, karhu_psin_axis)
    q_karhu = abs(q_karhu)  # TODO/FIXME: Are the q-s normalised?

    # Construct boundary in polar coordinates
    rhobndry, thetabndry = get_polar_from_rz(r_vals=rbndry_karhu, z_vals=zbndry_karhu, symmetric=False)
    rhobndry_karhu = interpolate_profile(x_0=thetabndry, y_0=rhobndry, x_1=karhu_theta_axis)

    x = [
        torch.tensor(pressure_karhu, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
        torch.tensor(q_karhu, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
        torch.tensor(rbphi_karhu, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
        torch.tensor(rhobndry_karhu, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
        torch.tensor(B_mag, dtype=torch.float32).unsqueeze(0),
        torch.tensor(R_mag, dtype=torch.float32).unsqueeze(0),
    ]
    return x
