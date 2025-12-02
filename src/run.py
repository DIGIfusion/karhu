# run.py
import os
import json
import gzip
import shutil
import logging
import argparse

import torch
import numpy as np

from karhu.utils_input import get_model_input, scale_model_input, descale_minmax
from karhu.models import CNN_gmax

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger = logging.getLogger(__name__)


def main(args: argparse.Namespace):
    """
    Main function for running the surrogate model.

    Args:
        args (argparse.Namespace): Namespace containing the configuration parameters.
    """

    n_profile_points = 64
    x_1 = np.linspace(1e-5, 1, n_profile_points) ** (1 / 4)

    run_dir = args.helena_directory
    model, scaling_params = load_model(model_dir=args.model_directory)


    with gzip.open(os.path.join(run_dir, 'fort.12.gz'), 'rb') as f_in:
        with open(os.path.join(run_dir, 'fort.12'), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    x = get_model_input(
        filename_f10=os.path.join(run_dir, 'fort.10'),
        filename_f12=os.path.join(run_dir, 'fort.12'),
        filename_f20=os.path.join(run_dir, 'fort.20'),
        x_1=x_1,
        n_profile_points=n_profile_points,)

    x = scale_model_input(x, scaling_params)
    y_pred = model(*x)
    y_pred = descale_minmax(
        y_pred.item(),
        scaling_params["growthrate"][0],
        scaling_params["growthrate"][1],
    )
    y_pred = 0.0 if y_pred <= 0.0 else y_pred
    print(f"KARHU predicted growthrate: {y_pred}")
    return y_pred

def load_model(model_dir: str):
    """ The model directory should contain
    - model.pt containing the weights
    - scaling_params.json containing the scaling parameters for the inputs and outputs
    """

    # Load scaling parameters
    with open(os.path.join(model_dir, "scaling_params.json",), "r", encoding="utf-8",) as f:
        scaling_params = json.load(f)

    # Load model
    model = CNN_gmax()
    model.load_state_dict(
        torch.load(os.path.join(model_dir, "model.pt"), weights_only=True,)
    )
    model.eval()
    return model, scaling_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Machine learning surrogate model for ideal MHD stability code MISHKA.")
    parser.add_argument(
        "-hd",
        "--helena_directory",
        type=str,
        default="../example/data/jet_2H",
        help="Path to a directory containing HELENA equilibrium files fort.12, fort.20, fort.10.",
    )
    parser.add_argument(
        "-m",
        "--model_directory",
        type=str,
        default="../model/jet_2H",
        help="Path to a directory containing the trained model.",
    )
    args = parser.parse_args()
    main(args)
