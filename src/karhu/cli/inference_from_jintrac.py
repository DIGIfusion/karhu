"""
An example python script for running KARHU from a HELENA directory and writing the result to a file. 
"""
import argparse
import sys
import numpy as np
from enum import Enum 

sys.path.append("/home/mn2596/JETPEDESTAL_ANALYSIS/karhu/src/karhu")
from karhu.utils_helena import load_from_helena
from karhu.models import load_model
from karhu.utils_input import scale_model_input, scale_model_output
from karhu.common import convert_profiles_si_to_dimensionless, get_polar_from_rz
from karhu.utils_input import interpolate_profile

import torch


"""
Where to source from 
"""
class RUNMODE(Enum): 
    HELENA   = 0
    DATFILES = 1

WP     = torch.float32  # TODO: will this change in future? 
DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"


def do_inference(x, scaling_params, model): 
    x = scale_model_input(x, scaling_params)
    x = [_x.to(DEVICE) for _x in x]
    with torch.no_grad():
        y = model(*x)
    y = scale_model_output(y, scaling_params)
    return y


def read_jettoin(dirname):
    # TODO: need the rmag and bmag for the normalisation
    rbnd = np.loadtxt(dirname + "/JETTO_RBNDin.dat")
    zbnd = np.loadtxt(dirname + "/JETTO_ZBNDin.dat")
    psig = np.loadtxt(dirname + "/JETTO_PSIin.dat")
    pr   = np.loadtxt(dirname + "/JETTO_PRin.dat")
    rbphi   = np.loadtxt(dirname + "/JETTO_Fin.dat")
    q    = np.loadtxt(dirname + "/JETTO_QSFin.dat")
    BMAG = 1.0  # TODO/FIXME: need to get the actual Bmag and Rmag from the datfiles, for now we just set them to 1.0 since the model should be able to handle this normalisation as well, but this is not ideal
    RMAG = 1.0  # TODO/FIXME: need to get the actual Bmag and Rmag from the datfiles, for now we just set them to 1.0 since the model should be able to handle this normalisation as well, but this is not ideal
    return rbnd, zbnd, pr, q, rbphi, psig, BMAG, RMAG


def get_from_datfiles(datfiles_dir: str, model_config: dict):

    rbnd, zbnd, pr, q, rbphi, psig, B_mag, R_mag = read_jettoin(datfiles_dir)

    RGEO = (rbnd.max() + rbnd.min()) / 2.0
    AGEO = (rbnd.max() - rbnd.min()) / 2.0
    EPS = AGEO / RGEO
    radius = EPS * RGEO / R_mag

    pr, rbphi, rbnd, zbnd = convert_profiles_si_to_dimensionless(
        pr, rbphi, rbnd, zbnd, radius, R_mag, EPS, B_mag)
    pr_karhu = interpolate_profile(psig, pr, model_config["karhu_psin_axis"])
    q_karhu = interpolate_profile(psig, q, model_config["karhu_psin_axis"])
    rbphi_karhu = interpolate_profile(psig, rbphi, model_config["karhu_psin_axis"])
    symmetric = False 
    rhobndry, thetabdry = get_polar_from_rz(
        r_vals=rbnd, z_vals=zbnd,  symmetric=symmetric)

    rhobndry_karhu = interpolate_profile(
        x_0=thetabdry, y_0=rhobndry, x_1=model_config["karhu_theta_axis"])

    x = [torch.tensor(pr_karhu, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
         torch.tensor(q_karhu, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
         torch.tensor(rbphi_karhu, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
         torch.tensor(rhobndry_karhu, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
         torch.tensor(B_mag, dtype=torch.float32).unsqueeze(0),
         torch.tensor(R_mag, dtype=torch.float32).unsqueeze(0),
    ]
    return x 


if __name__ == "__main__": 
    parser = argparse.ArgumentParser("Run KARHU on a HELENA directory")
    parser.add_argument("-dd", "--data_directory", type=str, required=True, help="Path to data directory")
    parser.add_argument("-m",  "--model_directory", type=str, required=True, help="Path to KARHU model directory")
    parser.add_argument('-w', "--write_filename", type=str, default=None, help="Write the prediction to file with name given here, if not passed, no file will be written")
    parser.add_argument("-rm", "--runmode", type=int, choices=[mode.value for mode in RUNMODE], default=RUNMODE.HELENA.value, help="Where to source the input data from, 0 for HELENA, 1 for datfiles")
    args = parser.parse_args()

    model, model_config = load_model(args.model_directory)
    scaling_params = model_config["scaling_params"]
    model = model.to(device=DEVICE, dtype=WP)

    if args.runmode == RUNMODE.HELENA.value:
        x = load_from_helena(args.data_directory,
                             karhu_psin_axis=model_config["karhu_psin_axis"], 
                             karhu_theta_axis=model_config["karhu_theta_axis"])

    elif args.runmode == RUNMODE.DATFILES.value:
        x = get_from_datfiles(
            args.model_directory, args.data_directory, model_config)
    else:
        raise NotImplementedError(
            "Choose (0, or 1) for runmode, got {}".format(args.runmode))

    prediction = do_inference(x, scaling_params, model)
    print("Prediction: {:.4}".format(prediction))
    prediction = 0.0 if prediction < 0.0 else prediction

    print("Prediction: {:.4}".format(prediction))
    if args.write_filename is not None:
        with open(args.write_filename, 'w') as file:
            file.write(f"{prediction}")
