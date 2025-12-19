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
    scale_model_input,
    interpolate_profile,
    minmax,
    scale_model_output)
from karhu.utils_helena import (
    load_from_helena, get_f12_data, read_fort20_beta_section)
from karhu.models import load_model

TESTDATADIR = os.path.dirname(__file__)

helena_directory = os.path.join(TESTDATADIR, "data", "helena_jet")
models_directory = os.path.join(TESTDATADIR, "..", "model", "jet_2H")

@pytest.mark.parametrize("heldir,modeldir", [(helena_directory, models_directory)])
def test_inference(heldir, modeldir):
    model, scaling_params = load_model(model_dir=modeldir)
    model_inputs = load_from_helena(heldir)
    model_inputs = scale_model_input(model_inputs, scaling_params)
    with torch.no_grad(): 
        y_pred_norm = model(*model_inputs)
    y_pred = scale_model_output(y_pred_norm, scaling_params)
    print(f"Predicted growth rate: {y_pred:.4f}")
    assert y_pred > 0.0
