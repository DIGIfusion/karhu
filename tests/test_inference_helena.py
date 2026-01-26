"""
Inference from HELENA
"""
from __future__ import annotations

import os
import pytest

import torch

from karhu import (
    scale_model_input,
    scale_model_output,
    load_from_helena)
from karhu.models import (
    load_model, load_ensemble_model, get_ensemble_prediction)

TESTDATADIR = os.path.dirname(__file__)

helena_directory = os.path.join(TESTDATADIR, "data", "helena_jet")
models_directory = os.path.join(TESTDATADIR, "..", "model", "jet_2H")
ensembles_directory = os.path.join(TESTDATADIR, "..", "model_ensemble", "jet_diii-d")


@pytest.mark.parametrize("heldir,modeldir", [(helena_directory, models_directory)])
def test_inference(heldir, modeldir):
    model, model_config = load_model(model_dir=modeldir)
    scaling_params = model_config["scaling_params"]
    model_inputs = load_from_helena(heldir,
        karhu_psin_axis=model_config["karhu_psin_axis"],
        karhu_theta_axis=model_config["karhu_theta_axis"])
    model_inputs = scale_model_input(model_inputs, scaling_params)
    with torch.no_grad():
        y_pred_norm = model(*model_inputs)
    y_pred = scale_model_output(y_pred_norm, scaling_params)
    print(f"Predicted growth rate: {y_pred:.4f}")
    assert y_pred > 0.0


@pytest.mark.parametrize("heldir,modeldir", [(helena_directory, ensembles_directory)])
def test_inference_ensemble(heldir, modeldir):
    models, model_config = load_ensemble_model(ensemble_dir=modeldir)
    scaling_params = model_config["scaling_params"]
    model_inputs = load_from_helena(heldir,
        karhu_psin_axis=model_config["karhu_psin_axis"],
        karhu_theta_axis=model_config["karhu_theta_axis"])
    model_inputs = scale_model_input(model_inputs, scaling_params)
    y_pred_mean, y_pred_std = get_ensemble_prediction(models, model_inputs)
    y_pred_mean = scale_model_output(y_pred_mean, scaling_params)
    y_pred_std  = scale_model_output(y_pred_std, scaling_params)
    print(f"Predicted mean growth rate: {y_pred_mean:.4f} with std: {y_pred_std:.4f}")

    assert y_pred_mean > 0.0
