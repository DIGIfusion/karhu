"""
Train a model. Set environment variables for MLFlow first.
"""

import argparse
import os
import sys
import json
import logging
from datetime import datetime
import numpy as np
import torch
# from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split
# import mlflow
from dotenv import load_dotenv


from karhu import setup_logger
from karhu.models import GMaxPredictor, setup_dataset
from karhu.training import train_model, test_model
from karhu.training import plot_losses,get_regression_scores


# Load environment variables from .env file
load_dotenv(dotenv_path="/scratch/project_2009007/mishka-nn/.env")
os.environ["MLFLOW_TRACKING_URI"]       = os.getenv("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_TRACKING_USERNAME"]  = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"]  = os.getenv("MLFLOW_TRACKING_PASSWORD")
os.environ["AWS_ACCESS_KEY_ID"]         = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"]     = os.getenv("AWS_SECRET_ACCESS_KEY")

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
# SAVE_NAME   = f"model_{datetime.now():%Y%m%d_%H%M%S}"
# SAVE_DIR    = os.path.join(SCRIPT_DIR, "../models", SAVE_NAME)


def main():
    """ Main function to train a model."""
    start_time = datetime.now()

    # Create argument parser
    parser = argparse.ArgumentParser(description="Train a CNN model")
    parser.add_argument("-e", "--epochs", type=int, help="Number of epochs for training", required=True)
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the training dataset directory",)

    # Optional arguments for second dataset mix
    parser.add_argument("--seed", type=int, default=42, help="Random seed.",)
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.",)
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1.",)
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2.",)
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay.",)
    parser.add_argument("--batch_size", type=int, default=256, help="Size of batch (training and validation sets).",)
    parser.add_argument("--model_name", type=str, default="KARHU-ensemble", help="Name of the model to MLFLOW.",)
    parser.add_argument("--ensemble_id", type=int, required=True)
    parser.add_argument("--training_split", type=float, default=0.7, help="Fraction of the data used in training vs validation.")
    
    args = parser.parse_args()
    
    slurm_id = os.environ.get("SLURM_JOB_ID", "noslurm")
    SAVE_NAME = f"model_{slurm_id}_ens{args.ensemble_id}"
    SAVE_DIR = os.path.join(SCRIPT_DIR, "../models", SAVE_NAME)
    os.makedirs(SAVE_DIR, exist_ok=False)

    setup_logger(os.path.join(SAVE_DIR, "training.log"))
    logger = logging.getLogger(__name__)
    logger.info("Starting.")
    logger.info("Device: %s", DEVICE)
    logger.info("System Version: %s", sys.version)
    logger.info("PyTorch Version: %s", torch.__version__)
    logger.info("NumPy Version: %s", np.__version__)
    logger.info(get_args_description(args))

    
    epochs = args.epochs

    # Setup model configuration
    model_config = {
        "conv_input_sizes": (64, 64, 64, 128),
        "scalar_inputs": 2,
        "conv_kernel_sizes": (7, 5, 3),
        "out_channels": 16,
        "pool_kernel_size": 2,
        "fc_hidden_dims": (128, 64),
        "conv_dropout": 0.1,
        "fc_dropout": 0.2,
    }

    # Load data
    dataset, data_config = setup_dataset(dir_path=args.dataset_dir)
    config = data_config | model_config
    save_config(SAVE_DIR, config)

    # Ratios for splitting data set
    test_ratio = 0.10
    train_ratio = args.training_split

    n_total = len(dataset)
    n_test = int(test_ratio * n_total)
    n_trainval = n_total - n_test
    n_train = int(train_ratio * n_trainval)
    n_val   = n_trainval - n_train

    logger.info(
        f"Total data samples: {n_total}, "
        f"test: {n_test}, train: {n_train}, val: {n_val}"
    )


    g_test = torch.Generator().manual_seed(args.seed)

    trainval_data, test_data = random_split(dataset, [n_trainval, n_test], generator=g_test)

    ens_id = args.ensemble_id
    ens_seed = args.seed + ens_id
    
    if True:
    # mlflow.set_experiment(args.model_name)
    # with mlflow.start_run(run_name=f"ensemble_{ens_id}"):
    #     mlflow.log_params({
    #         **args.__dict__,
    #         "ensemble_id": ens_id,
    #         "seed": ens_seed
    #     })
        logger.info(f"==== Training ensemble member {ens_id} ====")

        # Different seed per ensemble member
        torch.manual_seed(ens_seed)
        np.random.seed(ens_seed)

        # Create model
        model = GMaxPredictor(model_config).to(DEVICE)
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay
        )

        # --- Train / val split (varies per ensemble member) ---
        g_tv = torch.Generator().manual_seed(ens_seed)
        train_data, val_data = random_split(
            trainval_data,
            [n_train, n_val],
            generator=g_tv
        )

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        val_loader   = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

        # ======================== #
        # Train model
        # ======================== #

        train_losses, val_losses = train_model(
            model,
            optimizer,
            model.loss_fn,
            train_loader,
            val_loader,
            epochs=epochs,
            early_stopping=True,
            early_stopping_min_delta=1e-5,
            early_stopping_patience=70
        )

        # Save member
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "model.pt"))
        plot_losses(
            train_losses,
            val_losses,
            filename=os.path.join(SAVE_DIR, "training.png")
        )

        # ======================== #
        # Test set evaluation
        # ======================== #
        test_loader = DataLoader(
            test_data,
            batch_size=len(test_data),
            shuffle=False
        )

        y_test, y_pred = test_model(
            model,
            test_loader,
            dataset=dataset,
            device=DEVICE
        )
        np.save(os.path.join(SAVE_DIR, "y_test.npy"), y_test)
        np.save(os.path.join(SAVE_DIR, "y_pred.npy"), y_pred)

        metrics = get_regression_scores(y_test, y_pred)
        # mlflow.log_metric("final_train_loss", train_losses[-1])
        # mlflow.log_metric("final_val_loss", val_losses[-1])

        # for k, v in metrics.items():
        #     mlflow.log_metric(f"{k}_member", v)

        logger.info(
            f"Member {ens_id} metrics: "
            + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        )
        logger.info("Total runtime: %s", datetime.now() - start_time)


def get_args_description(args):
    """Get a description of the input arguments."""
    desc = []
    desc.append("\n=== Input arguments ===")
    for key, value in vars(args).items():
        desc.append(f"  {key}: {value}")
    desc.append("=======================")
    return "\n".join(desc)


def save_config(save_dir, config):
    with open(
        os.path.join(save_dir, "model_config.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(config, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
