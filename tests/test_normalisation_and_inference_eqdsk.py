"""
KARHU/MISHKA dimensionless normalisation from SI units

KARHU_PRESSURE = PRESSURE_SI / (B_magaxis**2 / MU_0)
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

import sys
import os
import glob

import f90nml
import pytest
import numpy as np
from freeqdsk import geqdsk
import torch

from karhu import load_model
from karhu.common import convert_profiles_si_to_dimensionless
from karhu.utils_input import (
    interpolate_profile,
    scale_model_input,
    descale_minmax)
from karhu.utils_helena import get_f12_data, read_fort20_beta_section, load_from_helena
from karhu.utils_eqdsk import load_from_eqdsk

TESTDIR = os.path.dirname(__file__)
TESTDATADIR = os.path.join(TESTDIR, "data")

eqdsk_testfiles = glob.glob(os.path.join(TESTDATADIR, "eqdsk", "*"))
models_directory = os.path.join(TESTDIR, "..", "model", "jet_2H")  # TODO: add more models
diiid_models_directory = os.path.join(TESTDIR, "..", "model", "diii-d")  # TODO: add more models


def load_eqdsk(eqfpath: str):
    """
    Load an EQDSK file using freeqdsk
    """
    with open(eqfpath, "r") as f:
        eqdsk = geqdsk.read(f)
    return eqdsk


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


@pytest.mark.skipif(sys.version_info < (3, 9), reason="freeqdsk has attributes only in versions available for python 3.9 or higher")
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


@pytest.mark.skipif(sys.version_info < (3, 9), reason="freeqdsk has attributes only in versions available for python 3.9 or higher")
@pytest.mark.parametrize("eqdskpath", eqdsk_testfiles)
def test_inference_from_eqdsk(eqdskpath):

    if "DIIID" in eqdskpath:
        model, model_config = load_model(diiid_models_directory)
    else:
        model, model_config = load_model(models_directory)
    scaling_params = model_config["scaling_params"]
    x = load_from_eqdsk(eqdskpath,
        karhu_psin_axis=model_config["karhu_psin_axis"],
        karhu_theta_axis=model_config["karhu_theta_axis"])
    x = scale_model_input(x, scaling_params)
    with torch.no_grad():
        y_pred = model(*x)

    y_pred = descale_minmax(y_pred.item(), *scaling_params["growthrate"])
    assert y_pred >= 0.0  # TODO: have some benchmark cases?
    print(f"Predicted growth rate: {y_pred:.4f}")


@pytest.mark.skipif(sys.version_info < (3, 9), reason="freeqdsk has attributes only in versions available for python 3.9 or higher")
@pytest.mark.parametrize("eqdskpath", eqdsk_testfiles)
def test_compare_inference_eqdsk_helena(eqdskpath):
    name = os.path.basename(eqdskpath).split('.eqdsk')[0]
    corresponding_helena = [fname for fname in glob.glob(os.path.join(TESTDATADIR, "helena", "*")) if name in fname]
    if len(corresponding_helena) == 0:
        pytest.skip("No corresponding HELENA for this EQDSK")
    corresponding_helena = corresponding_helena[0]
    print(corresponding_helena)
    """
    Inference with
    """
    if "DIIID" in eqdskpath:
        model, model_config = load_model(diiid_models_directory)
    else:
        model, model_config = load_model(models_directory)
    
    x_eqdsk = load_from_eqdsk(
        eqdskpath,
        karhu_psin_axis=model_config["karhu_psin_axis"],
        karhu_theta_axis=model_config["karhu_theta_axis"])
    x_helena = load_from_helena(
        corresponding_helena,
        karhu_psin_axis=model_config["karhu_psin_axis"],
        karhu_theta_axis=model_config["karhu_theta_axis"])

    scaling_params = model_config["scaling_params"]

    for _x_eq, _x_he in zip(x_eqdsk, x_helena):
        assert torch.allclose(abs(_x_eq), abs(_x_he), rtol=0.25, atol=1.0)

    predictions = []
    for x in [x_eqdsk, x_helena]:
        x = scale_model_input(x, scaling_params)
        with torch.no_grad():
            y_pred = model(*x)
        y_pred = descale_minmax(y_pred.item(), *scaling_params["growthrate"])
        y_pred = np.max((0.0, y_pred))
        # assert y_pred >= 0.0  # TODO: have some benchmark cases?
        predictions.append(y_pred)
    y_pred_eqdsk, y_pred_helena = predictions
    print(f"Predicted growth rate EQDSK:  {y_pred_eqdsk:.4f}")
    print(f"Predicted growth rate HELENA: {y_pred_helena:.4f}")
    assert np.isclose(y_pred_eqdsk, y_pred_helena, rtol=0.30, atol=0.1), f"Predictions are off between EQDSK and HELENA ran EQDSK\n EQDSK : {y_pred_eqdsk:.4} \nHELENA: {y_pred_helena:.4}"
