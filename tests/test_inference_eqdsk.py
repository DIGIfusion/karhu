"""
Inference from EQDSK 

TODO: actually normalise these things as per HELENA. 
      - @awlishar need to reference the HELENA normalisation here and document the missing vars...
"""
from __future__ import annotations 

import os
import pytest

import f90nml 
from freeqdsk import geqdsk
import numpy as np
import torch 
import glob 

from karhu.utils_input import get_f12_data, read_fort20_beta_section, interpolate_psi_profile, minmax, scale_model_output
from karhu.models import load_model

TESTDIR = os.path.dirname(__file__)

TESTDATADIR = os.path.join(TESTDIR, "data")
eqdsk_testfiles = glob.glob(os.path.join(TESTDATADIR, "eqdsk", "*"))
models_directory  = os.path.join(TESTDIR, "..", "model", "jet_2H")
   

def load_from_eqdsk(eqdsk_path: str): 

    with open(eqdsk_path, "r") as file: 
        eqdsk = geqdsk.read(file)

    CS = np.linspace(0, 1, eqdsk.nx)
    P  = eqdsk.pres
    RBPHI = eqdsk.fpol 
    QS    = eqdsk.qpsi 
    rbndry = eqdsk.rbdry
    zbndry = eqdsk.zbdry
    B0     = eqdsk.bcentr
    R0     = eqdsk.rcentr

    # TODO: this is not correct, R0, B0, etc.,..

    """ 
    Interpolation onto karhu grid 
    """
    NPOINTS = 64
    new_grid = np.linspace(1e-5, 1, NPOINTS) ** (1 / 2) # this uniform grid in PSI**2, which means to get back to CS points we need to do power 1/4
    old_grid = CS 
    [P_interp, QS_interp, RBPHI_interp] = interpolate_profiles_onto_grid(new_grid, old_grid, profiles=[P, QS, RBPHI])

    # Boundary grid: 
    new_grid_vx = np.linspace(rbndry.min(), rbndry.max(), NPOINTS)
    [VY_interp]   = interpolate_profiles_onto_grid(newgrid=new_grid_vx, oldgrid=rbndry, profiles=[zbndry])

    # model_input = np.stack([P_interp, QS_interp, RBPHI_interp, VY_interp, B0, R0])
    model_input = dict(p=P_interp, qs=QS_interp, rbphi=RBPHI_interp, shape=VY_interp, b_mag=B0, r_mag=R0)
    return model_input


def interpolate_profiles_onto_grid(newgrid, oldgrid, profiles: list): 
    new_profiles = []
    for _prof in profiles: 
        new_prof = interpolate_psi_profile(oldgrid, _prof, newgrid)
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

@pytest.mark.parametrize("eqdsk", eqdsk_testfiles)
@pytest.mark.parametrize("modeldir", [models_directory])
def test_inference(eqdsk, modeldir):
    model_inputs = load_from_eqdsk(eqdsk)
    model, scaling_params = load_model(model_dir=modeldir)
    model_inputs = scale_inputs(model_inputs, scaling_params)
    model_inputs = to_tensor(model_inputs) 
    y_pred_norm  = model(input_p=model_inputs["p"], input_qs=model_inputs["qs"], input_rbphi=model_inputs["rbphi"], input_shape=model_inputs["shape"], b_mag=model_inputs["b_mag"], r_mag=model_inputs["r_mag"])
    y_pred       = scale_model_output(y_pred_norm, scaling_params)

    print(y_pred)
    assert y_pred > 0.0