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
from torch.utils.data import DataLoader

from dotenv import load_dotenv

from karhu import setup_logger
from karhu.models import GMaxPredictor, setup_dataset
from karhu.training import train_model, test_model, split_dataset
from karhu.training import plot_losses, plot_pred_vs_true, get_regression_scores

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
SAVE_NAME   = f"model_{datetime.now():%Y%m%d_%H%M%S}"
SAVE_DIR    = os.path.join(SCRIPT_DIR, "../../models", SAVE_NAME)


def main():
    """ Main function to train a model."""
    logger = logging.getLogger(__name__)
    start_time = datetime.now()
    logger.info("Starting.")
    logger.info("Device: %s", DEVICE)
    logger.info("System Version: %s", sys.version)
    logger.info("PyTorch Version: %s", torch.__version__)
    logger.info("NumPy Version: %s", np.__version__)

    # Create argument parser
    parser = argparse.ArgumentParser(description="Train a CNN model")
    parser.add_argument("-e", "--epochs", type=int, help="Number of epochs for training")
    parser.add_argument("-m", "--model", type=str, help="ML model")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the training dataset directory",)

    # Optional arguments for second dataset mix
    parser.add_argument("--seed", type=int, default=42, help="Random seed.",)
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.",)
    parser.add_argument("--batch_size", type=int, default=256, help="Size of batch (training and validation sets).",)
    parser.add_argument("--model_name", type=str, default="KARHU", help="Name of the model to MLFLOW.",)
    parser.add_argument("--castor", action="store_true", help="Use CASTOR growth rate.")

    args = parser.parse_args()
    logger.info(get_args_description(args))

    # New setup
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
    model = GMaxPredictor(model_config)
    epochs = args.epochs
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ======================== #
    # Setup data
    # ======================== #

    # Load dataset(s)
    dataset, data_config = setup_dataset(dir_path=args.dataset_dir)
    config = data_config | model_config
    save_config(SAVE_DIR, config)

    # Ratios for splitting data set
    test_ratio = 0.10
    train_ratio = 0.70
    val_ratio = 0.20
    train_data, val_data, test_data = split_dataset(
        dataset, train_ratio, val_ratio, test_ratio, base_seed=args.seed)

    logger.info("Training samples (%d%%): %d", int(train_ratio * 100), len(train_data),)
    logger.info("Validation samples (%d%%): %d", int(val_ratio * 100), len(val_data),)
    logger.info("Test samples (%d%%): %d", int(test_ratio * 100), len(test_data),)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)


    # Train model
    train_losses, val_losses = train_model(
        model, optimizer, model.loss_fn, train_loader, val_loader, epochs=epochs,
        early_stopping=False, early_stopping_min_delta=0.00001, early_stopping_patience=20
    )
    with open(os.path.join(SAVE_DIR, "losses.npy"), 'wb') as f:
        np.save(f, train_losses)
        np.save(f, val_losses)

    # Save model and scaling params
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "model.pt"))
    dataset.save_scaling_params(SAVE_DIR)
    logger.info("Saved to: %s", SAVE_DIR)

    # Plot losses and save fig
    plot_losses(train_losses, val_losses, filename=os.path.join(SAVE_DIR, "training.png"))

    # ======================== #
    # Test data set
    # ======================== #
    logger.info("Evaluating TEST SET...")

    y_test, y_pred = test_model(model, test_loader, dataset=dataset, device=DEVICE)

    plot_pred_vs_true(y_test, y_pred, filename=os.path.join(SAVE_DIR, "pred_vs_true_testset.png"),)
    metrics = get_regression_scores(y_test, y_pred)
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.5f}")

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
    os.makedirs(SAVE_DIR, exist_ok=False)
    setup_logger(os.path.join(SAVE_DIR, "training.log"))
    main()
