"""
Evaluate an ensemble of trained models.

Assumes directory structure:
models/
  model_<slurm>_ens0/
    ensemble_0/model.pt
  model_<slurm>_ens1/
    ensemble_1/model.pt
  ...

All ensemble members must share:
- identical model config
- identical dataset

Use:

source /scratch/project_2009007/.venv/bin/activate
python scripts/test_ensemble.py \
    --models_root /scratch/project_2009007/ambrunc/dev_karhu/karhu-training/models/ensemble_jet2h_20260129 \
    --dataset_path /scratch/project_2009007/ambrunc/dev_karhu/karhu-training/db/JET_2H/interpolated64.h5 \
    --output_dir /scratch/project_2009007/ambrunc/dev_karhu/karhu-training/models/ensemble_jet2h_20260129

"""

import os
import json
import argparse
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from karhu_training.models import setup_dataset
from karhu_training.train import test_ensemble
from karhu_training.logger_config import setup_logger
from karhu_training.utils_plotting import (
    plot_pred_vs_true,
    plot_pred_vs_true_colored,
    plot_uncertainty_vs_error,
    plot_coverage_curve,
    plot_uncertainty_histogram,
    get_regression_scores,
)
from karhu_training.models import load_ensemble_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Test an ensemble")
    parser.add_argument("--models_root", type=str, required=True,
                        help="Path containing model_*_ensX directories")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Dataset directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to store ensemble results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logger(os.path.join(args.output_dir, "ensemble_test.log"))
    logger = logging.getLogger(__name__)

    logger.info("Starting ensemble evaluation")
    logger.info("Device: %s", DEVICE)

    # ---------------------------
    # Load dataset
    # ---------------------------
    dataset, _ = setup_dataset(dir_path=args.dataset_path)
    test_ratio = 0.10
    train_ratio = 0.70

    n_total = len(dataset)
    n_test = int(test_ratio * n_total)
    n_trainval = n_total - n_test
    n_train = int(train_ratio * n_trainval)
    n_val   = n_trainval - n_train

    logger.info(
        f"Total data samples: {n_total}, "
        f"test: {n_test}, train: {n_train}, val: {n_val}"
    )
    g_test = torch.Generator().manual_seed(42)

    trainval_data, test_data = random_split(
        dataset, [n_trainval, n_test], generator=g_test)

    test_loader = DataLoader(
        test_data,
        batch_size=len(test_data),
        shuffle=False,
    )

    # ---------------------------
    # Discover ensemble members
    # ---------------------------
    models, model_config = load_ensemble_model(ensemble_dir=args.models_root)
    scaling_params = model_config["scaling_params"]

    logger.info("Found %d ensemble members", len(models))

    y_test_ref, y_pred_mean, y_pred_std = test_ensemble(
        models, test_loader, dataset)

    logger.info("Mean predictive std: %.5f", y_pred_std.mean())
    logger.info("Median predictive std: %.5f", np.median(y_pred_std))

    # ---------------------------
    # Metrics
    # ---------------------------
    scores = get_regression_scores(y_test_ref, y_pred_mean)
    for k, v in scores.items():
        logger.info("ENSEMBLE %s: %.5f", k, v)

    # ---------------------------
    # Plots
    # ---------------------------
    plot_pred_vs_true(
        y_test_ref,
        y_pred_mean,
        y_pred_std=y_pred_std,
        filename=os.path.join(args.output_dir, "pred_vs_true.png"),
    )

    plot_pred_vs_true_colored(
        y_test_ref,
        y_pred_mean,
        y_pred_std,
        os.path.join(args.output_dir, "pred_vs_true_colored.png"),
    )

    plot_uncertainty_vs_error(
        y_pred_std,
        np.abs(y_pred_mean - y_test_ref),
        os.path.join(args.output_dir, "uncertainty_vs_error.png"),
    )

    plot_coverage_curve(
        y_test_ref,
        y_pred_mean,
        y_pred_std,
        os.path.join(args.output_dir, "coverage_curve.png"),
    )

    plot_uncertainty_histogram(
        y_pred_std,
        os.path.join(args.output_dir, "uncertainty_hist.png"),
    )

    # ---------------------------
    # Save arrays
    # ---------------------------
    np.save(os.path.join(args.output_dir, "y_test.npy"), y_test_ref)
    np.save(os.path.join(args.output_dir, "y_pred_mean.npy"), y_pred_mean)
    np.save(os.path.join(args.output_dir, "y_pred_std.npy"), y_pred_std)

    logger.info("Ensemble evaluation complete")


if __name__ == "__main__":
    main()
