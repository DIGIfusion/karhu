""" 
KARHU/MISHKA dimensionless normalisation from SI units

KARHU_PRESSURE = PRESSURE_SI / (B_magaxis**2 / mu_0)
KARHU_RBPHI    = RBPHI_SI    / (R_magaxis * B_magaxis)
KARHU_RBDRY    = RBDRY_SI    / (radius * R_magaxis) - 1.0 / eps
KARHU_ZBDRY    = ZBDRY_SI    / (radius * R_magaxis) 
KARHU_Q        = Q

where PRESSURE_SI [N / m^2] plasma pressure          (defined on PSIN)
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
from __future__ import annotations 

import sys
import os
import glob 

import pytest
import numpy as np 
from freeqdsk import geqdsk 
import torch 

from karhu.models import load_model
from karhu.utils_input import interpolate_psi_profile, scale_model_input, descale_minmax

mu_0 = 4E-7 * np.pi  # TODO: move somewhere else
TESTDIR = os.path.dirname(__file__)
TESTDATADIR = os.path.join(TESTDIR, "data")

eqdsk_testfiles = glob.glob(os.path.join(TESTDATADIR, "eqdsk", "*"))
models_directory  = os.path.join(TESTDIR, "..", "model", "jet_2H") # TODO: add more models

""" 

"""
def load_eqdsk(eqfpath: str): 
    with open(eqfpath, "r") as f: 
        eqdsk = geqdsk.read(f)
    return eqdsk 

# TODO: move to utils
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
    pressure_karhu = pressure / (B_mag**2 / mu_0)
    rbphi_karhu    = rbphi     / (R_mag * B_mag)
    rbndry_karhu   = rbdry    / (radius * R_mag) - 1.0 / eps
    zbndry_karhu   = zbdry    / (radius * R_mag)
    return pressure_karhu, rbphi_karhu, rbndry_karhu, zbndry_karhu


def calculate_area(x, z):
    # Gauss-shoelace formula (https://en.wikipedia.org/wiki/Shoelace_formula)
    n = len(x)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n  # roll over at n
        area += x[i] * z[j]
        area -= z[i] * x[j]
    area = abs(area) / 2.0
    return area 

@pytest.mark.skipif(sys.version_info <= (3, 8), reason="freeqdsk has attributes only in versions available for python 3.9 or higher")
@pytest.mark.parametrize("eqdskpath", eqdsk_testfiles)
def test_normalisation(eqdskpath):
    eqdsk = load_eqdsk(eqdskpath)

    R_mag  = eqdsk.rmagx 
    B_mag  = eqdsk.fpol[0] / eqdsk.rmagx 

    R_geom = (eqdsk.rbdry.max() + eqdsk.rbdry.min()) / 2.0
    a_geom = (eqdsk.rbdry.max() - eqdsk.rbdry.min()) / 2.0
    eps    = a_geom / R_geom 

    radius = eps * R_geom / R_mag

    pressure_karhu, rbphi_karhu, rbndry_karhu, zbndry_karhu = convert_profiles_si_to_dimensionless(eqdsk.pressure, eqdsk.fpol, eqdsk.rbdry, eqdsk.zbdry, 
                                                                                                   radius, R_mag, eps, B_mag, )
    area_si         = calculate_area(eqdsk.rbdry, eqdsk.zbdry)
    area_normalised = calculate_area(rbndry_karhu, zbndry_karhu)
    
    assert np.isclose(rbphi_karhu[0], 1.0)
    assert np.all(rbndry_karhu >= -1.0 - 1E-8)
    assert np.all(rbndry_karhu <= 1. + 1E-8), f"Bad vals: {rbndry_karhu[~(rbndry_karhu <= 1.0)]}"
    assert np.isclose(area_normalised * (radius * R_mag) ** 2, area_si)
    

@pytest.mark.skipif(sys.version_info <= (3, 8), reason="freeqdsk has attributes only in versions available for python 3.9 or higher")
@pytest.mark.parametrize("eqdskpath", eqdsk_testfiles)
def test_inference_from_eqdsk(eqdskpath): 
    
    eqdsk = load_eqdsk(eqdskpath)
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
    
    ninterp = 64
    KARHU_PSIN_AXIS = np.linspace(1e-5, 1.0, ninterp) ** 0.5
    KARHU_VX_AXIS   = np.linspace(-0.999, 0.999, ninterp)   #

    pressure_karhu = interpolate_psi_profile(psin1d, pressure_karhu, KARHU_PSIN_AXIS)
    rbphi_karhu = interpolate_psi_profile(psin1d, rbphi_karhu, KARHU_PSIN_AXIS)
    q_karhu = interpolate_psi_profile(psin1d, q_karhu, KARHU_PSIN_AXIS)

    # FIXME: version 1.0 of the model only takes top half of the boundary
    reduced_bndry = zbndry_karhu > 0.0
    rbndry_top, zbndry_top = rbndry_karhu[reduced_bndry], zbndry_karhu[reduced_bndry]
    sorted_idx = np.argsort(rbndry_top)
    rbndry_karhu, zbndry_karhu = rbndry_top[sorted_idx], zbndry_top[sorted_idx]
    zbndry_karhu = interpolate_psi_profile(rbndry_karhu, zbndry_karhu, KARHU_VX_AXIS)

    # FIXME: can just cast everything as double later
    x = [torch.tensor(pressure_karhu, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
         torch.tensor(q_karhu, dtype=torch.float32).unsqueeze(0).unsqueeze(0), 
         torch.tensor(rbphi_karhu, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
         torch.tensor(zbndry_karhu, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
         torch.tensor(B_mag, dtype=torch.float32).unsqueeze(0),
         torch.tensor(R_mag, dtype=torch.float32).unsqueeze(0),
    ]

    model, scaling_params = load_model(models_directory)
    x = scale_model_input(x, scaling_params)
    with torch.no_grad(): 
        y_pred = model(*x)

    y_pred = descale_minmax(y_pred.item(), *scaling_params["growthrate"])
    assert y_pred >= 0.0  # TODO: have some benchmark cases? 

