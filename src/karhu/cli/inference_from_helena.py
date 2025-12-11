"""
An example python script for running KARHU from a HELENA directory and writing the result to a file. 
"""
import argparse 
from karhu.utils_helena import get_model_input
from karhu.models import load_model
from karhu.utils_input import scale_model_input, scale_model_output

import torch

WP     = torch.float32 #TODO: will this change in future? 
DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"
    
def main(model_dir:str, helena_dir: str):
    model, scaling_params = load_model(model_dir)
    model = model.to(device=DEVICE, dtype=WP)
    p, q, rbphi, vy, bmag, rmag = get_model_input(helena_dir)

    x = [torch.tensor(p, dtype=WP).unsqueeze(0).unsqueeze(0), 
        torch.tensor(q, dtype=WP).unsqueeze(0).unsqueeze(0), 
        torch.tensor(rbphi, dtype=WP).unsqueeze(0).unsqueeze(0), 
        torch.tensor(vy, dtype=WP).unsqueeze(0).unsqueeze(0), 
        torch.tensor(bmag, dtype=WP).unsqueeze(0), 
        torch.tensor(rmag, dtype=WP).unsqueeze(0), 
        ]
    print(x)
    x = scale_model_input(x, scaling_params)
    with torch.no_grad(): 
        y = model(*x)
    y = scale_model_output(y, scaling_params)
    return y 

if __name__ == "__main__": 
    parser = argparse.ArgumentParser("Run KARHU on a HELENA directory")
    parser.add_argument("-hd", "--helena_directory", type=str, required=True, help="Path to HELENA directory")
    parser.add_argument("-m",  "--model_directory", type=str, required=True, help="Path to KARHU model directory")
    parser.add_argument('-w', "--write_filename", type=str, default=None, help="Write the prediction to file with name given here, if not passed, no file will be written")
    args = parser.parse_args()

    prediction = main(args.model_directory, args.helena_directory)
    print("Prediction: {:.4}".format(prediction))
    # TODO: better output writitng? 
    if args.write_filename is not None: 
        with open(args.write_filename, 'w') as file:
            file.write(f"{prediction}")
