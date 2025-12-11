"""
Inference from HELENA
"""
from __future__ import annotations

import os
import pytest

import f90nml
import numpy as np
import torch

from karhu.utils_input import (
    get_f12_data,
    read_fort20_beta_section,
    interpolate_profile, minmax,
    scale_model_output)
from karhu.models import load_model

TESTDATADIR = os.path.dirname(__file__)

helena_directory = os.path.join(TESTDATADIR, "data", "helena_jet")
models_directory = os.path.join(TESTDATADIR, "..", "model", "jet_2H")


def load_data_from_helena_directory(helena_directory: str) -> dict[str, np.ndarray]:
    """
    Read the q(psi), pressure(psi), rbphi(psi), boundary
    And interpolate onto the axis that the model was trained on.
    Stack profiles for [P, Q, RBPHI, BNDRY, B0, R0], where B0 and R0 are scalars.

    """

    fname_f10: str = os.path.join(helena_directory, "fort.10")
    fname_f12: str = os.path.join(helena_directory, "fort.12")
    fname_f20: str = os.path.join(helena_directory, "fort.20")

    f12data = get_f12_data(fname_f12, variables=["CS", "QS", "RADIUS", "P0", "RBPHI", "VX", "VY"])
    f10data = f90nml.read(fname_f10)
    """
    Profiles come from mapping file (fort.12) --> input into MISHKA
    They are normalised such that p(psi=0.0) = 1,
    """
    RADIUS: float = f12data["RADIUS"]                             # geometric axis minor radius??
    # rest are numpy arrays
    P, RBPHI, QS = f12data["P0"], f12data["RBPHI"], f12data["QS"] # normalised profiles
    CS = f12data["CS"]                                            # Helena uses sqrt(psi) grid, i.e., CS**2 = PSI, CS**4 = PSI**2,
    rbndry, zbndry = f12data["VX"], f12data["VY"]                 # Boundary in R, Z

    # geometric axis magnetic field strength
    *_, B_0H = read_fort20_beta_section(fname_f20)

    epislon = f10data["phys"]["eps"]   # inv. aspect ratio
    R_vac   = f10data["phys"]["rvac"]  # major radius in vaccuum
    B_vac   = f10data["phys"]["bvac"]

    """ Normalisation parameters for HELENA/MISHKA """
    R0   = (epislon / RADIUS) * R_vac
    B0   = B_vac / B_0H                 # TODO: explain

    """
    Interpolation onto karhu grid
    """
    NPOINTS = 64
    new_grid = np.linspace(1e-5, 1, NPOINTS) ** (1 / 4)  # this uniform grid in PSI**2, which means to get back to CS points we need to do power 1/4
    old_grid = CS
    [P_interp, QS_interp, RBPHI_interp] = interpolate_profiles_onto_grid(new_grid, old_grid, profiles=[P, QS, RBPHI])

    # Boundary grid:
    new_grid_vx = np.linspace(-0.999, 0.999, NPOINTS)
    [VY_interp] = interpolate_profiles_onto_grid(newgrid=new_grid_vx, oldgrid=rbndry, profiles=[zbndry])

    # model_input = np.stack([P_interp, QS_interp, RBPHI_interp, VY_interp, B0, R0])
    model_input = dict(p=P_interp, qs=QS_interp, rbphi=RBPHI_interp, shape=VY_interp, b_mag=B0, r_mag=R0)
    return model_input


def interpolate_profiles_onto_grid(newgrid, oldgrid, profiles: list):
    new_profiles = []
    for _prof in profiles:
        new_prof = interpolate_profile(oldgrid, _prof, newgrid)
        new_profiles.append(new_prof)
    return new_profiles


def scale_inputs(inputs: dict[str, np.ndarray | float], scaling_dict: dict[str, tuple[float, float]]) -> dict[str, np.ndarray | float]:
    inputs_scaled = {}
    for key, value in inputs.items():
        inputs_scaled[key] = minmax(value, *scaling_dict[key])
    return inputs_scaled


def to_tensor(inputs: dict[str, np.ndarray | float]) -> dict[str, torch.Tensor]:
    inputs_tensor = {}
    for key, value in inputs.items():
        value_tensor = torch.tensor(value, dtype=torch.float32)
        if isinstance(value, np.ndarray):
            value_tensor = value_tensor.unsqueeze(0).unsqueeze(0)
        elif isinstance(value, float):
            value_tensor = value_tensor.unsqueeze(0)
        inputs_tensor[key] = value_tensor
    return inputs_tensor


@pytest.mark.parametrize("heldir,modeldir", [(helena_directory, models_directory)])
def test_inference(heldir, modeldir):
    model_inputs = load_data_from_helena_directory(heldir)
    model, scaling_params = load_model(model_dir=modeldir)
    model_inputs = scale_inputs(model_inputs, scaling_params)
    model_inputs = to_tensor(model_inputs)
    y_pred_norm = model(
        input_p=model_inputs["p"],
        input_qs=model_inputs["qs"],
        input_rbphi=model_inputs["rbphi"],
        input_shape=model_inputs["shape"],
        b_mag=model_inputs["b_mag"],
        r_mag=model_inputs["r_mag"])
    y_pred = scale_model_output(y_pred_norm, scaling_params)
    assert y_pred > 0.0
