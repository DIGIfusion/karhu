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
import matplotlib.pyplot as plt

from karhu import setup_dataset, setup_logger
from karhu.gmax import test_ensemble, load_ensemble_model
from karhu import (
    plot_pred_vs_true,
    plot_pred_vs_true_colored,
    plot_uncertainty_vs_error,
    plot_coverage_curve,
    plot_uncertainty_histogram,
    get_regression_scores,
    get_classification_scores,
    plot_confusion_matrix
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Test an ensemble")
    parser.add_argument("--models_root", type=str, required=True, help="Path containing model_*_ensX directories")
    parser.add_argument("--db_dirs", type=str, nargs="+", required=True, help="Paths to the database directories",)
    parser.add_argument("--output_dir", type=str, required=True, help="Where to store ensemble results")
    parser.add_argument("--c_logp1", type=float, default=None, help="Constant logp1_transformation_c")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logger(os.path.join(args.output_dir, "ensemble_test.log"))
    logger = logging.getLogger(__name__)

    logger.info("Starting ensemble evaluation")
    logger.info("Device: %s", DEVICE)

    models, model_config = load_ensemble_model(ensemble_dir=args.models_root)
    scaling_params = model_config["scaling_params"]

    # ---------------------------
    # Load dataset
    # ---------------------------
    dataset_mappings = {
        "qs": "profiles/qs",
        "p": "profiles/p0",
        "rbphi": "profiles/rbphi",
        "shape": "profiles/boundary_polar",
        "gamma": "scalars/max_gr_mishka",
        "beta_n": "scalars/betan",
        # "r_mag": "scalars/rmag",
        # "b_mag": "scalars/bmag",
        # "eps": "scalars/radius"
        "eps": "scalars/eps"
    }
    dataset, _ = setup_dataset(
        dir_paths=args.db_dirs,
        dataset_mappings=dataset_mappings,
        scaling_params=scaling_params,
        log_transformation_c=args.c_logp1)
    test_ratio = 0.10
    train_ratio = 0.70

    n_total = len(dataset)
    n_test = int(test_ratio * n_total)
    n_trainval = n_total - n_test
    n_train = int(train_ratio * n_trainval)
    n_val = n_trainval - n_train

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
    classification_metrics = get_classification_scores(
        y_true=y_test_ref, y_pred=y_pred_mean, threshold=0.03)
    for k, v in classification_metrics.items():
        logger.info("ENSEMBLE %s: %.5f", k, v)

    # ---------------------------
    # Plots
    # ---------------------------
    filename = os.path.join(args.output_dir, "pred_vs_true.png")
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    plot_pred_vs_true(
        y_test_ref,
        y_pred_mean,
        y_pred_std=y_pred_std,
        ax=axs[0]
    )

    plot_pred_vs_true_colored(
        y_test_ref,
        y_pred_mean,
        y_pred_std,
        ax=axs[1],
    )

    plot_confusion_matrix(
        ax=axs[2],
        y_true=y_test_ref,
        y_pred=y_pred_mean, threshold=0.03)

    fig.tight_layout()
    plt.savefig(filename)
    logger.info(f"Figure saved: {filename}")

    filename = os.path.join(args.output_dir, "uncertainty.png")
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    plot_uncertainty_vs_error(
        y_pred_std,
        np.abs(y_pred_mean - y_test_ref),
        ax=axs[0]
    )
    plot_coverage_curve(
        y_test_ref,
        y_pred_mean,
        y_pred_std,
        ax=axs[1],
    )
    plot_uncertainty_histogram(
        y_pred_std,
        ax=axs[2],
    )
    fig.tight_layout()
    plt.savefig(filename)
    logger.info(f"Figure saved: {filename}")

    # ---------------------------
    # Save arrays
    # ---------------------------
    np.save(os.path.join(args.output_dir, "y_test.npy"), y_test_ref)
    np.save(os.path.join(args.output_dir, "y_pred_mean.npy"), y_pred_mean)
    np.save(os.path.join(args.output_dir, "y_pred_std.npy"), y_pred_std)

    logger.info("Ensemble evaluation complete")


if __name__ == "__main__":
    main()
