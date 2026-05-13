import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde

# import sklearn
from sklearn.metrics import (
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

plt.rcParams.update({"font.size": 12})
plt.rcParams.update({"axes.labelsize": 14})

BLUE_VERY_LIGHT = "#aec9fe"
BLUE_LIGHTER = "#95b8fe"
BLUE_LIGHT = "#3f7efd"
BLUE = "#0000FF"
BLUE_EUROFUSION = "#053991"
BLUE_DARK = "#000075"
BLACK = "#000000"


def get_regression_scores(y_true, y_pred):
    """
    Calculate prediction scores: MSE, MAE, MAPE, and R2.
    Args:
        y_true (array-like): True values.
        y_pred (array-like): Predicted values.
    Returns:
        mse (float): Mean Squared Error.
        mae (float): Mean Absolute Error.
        mape (float): Mean Absolute Percentage Error.
        r2 (float): R-squared score.
    """
    if y_pred.shape != y_true.shape:
        print(f"SHAPE ISSUE: y_true {y_true.shape}, y_pred {y_pred.shape}")

    y_residuals = y_true - y_pred
    mse = np.mean(y_residuals**2)
    mae = np.mean(np.abs(y_residuals))
    # mape = np.mean(np.abs((y_true - y_pred)/y_true)) * 100
    mape = 0.0
    r2 = r2_score(y_true, y_pred)
    return {"mse": mse, "mae": mae, "mape": mape, "r2": r2}

def get_classification_scores(y_true, y_pred_proba, threshold=0.5):
    """
    Calculate classification metrics for a binary 0/1 model.
    
    Args:
        y_true (array-like): True labels (0 or 1).
        y_pred_proba (array-like): Predicted probabilities or logits.
        threshold (float): Threshold to convert probabilities to class labels.
    
    Returns:
        metrics (dict): Dictionary containing accuracy, precision, recall, 
                        F1 score, ROC AUC, confusion matrix.
    """
    # If logits were passed   (shape [N, 1])
    if y_pred_proba.ndim > 1:
        y_pred_proba = y_pred_proba.squeeze()

    # Convert probabilities to 0/1
    y_pred = (y_pred_proba >= threshold).astype(int)
    y_true = y_true.astype(int).flatten()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_pred_proba)
    }

    return metrics

def plot_losses(train_losses, val_losses, filename: str = None):
    """
    Plot training and validation losses over epochs.
    Args:
        train_losses (list): List of training losses.
        val_losses (list): List of validation losses.
        filename (str): Filename to save the plot. If None, the plot is not saved.
    """
    fig, axs = plt.subplots(1, 1, figsize=(15, 10))
    axs.plot(train_losses[1:], label="Training loss")
    axs.plot(val_losses[1:], label="Validation loss")
    axs.legend()
    axs.set_ylabel("Loss (log)")
    axs.set_xlabel("Epochs")
    axs.set_title("Loss over epochs")
    axs.set_yscale("log")
    axs.grid(True)
    plt.tight_layout()

    if filename is not None:
        fig.savefig(filename)
    return fig


def plot_datasets(
    _x_train, _y_train, _x_val, _y_val, _x_test, _y_test, stability_threshold=0.03
):
    """
    _x_train = dataset.features["dped"][train_data.indices]
    _y_train = dataset.features["growthrate"][train_data.indices]

    _x_val = dataset.features["dped"][val_data.indices]
    _y_val = dataset.features["growthrate"][val_data.indices]

    _x_test = dataset_84541.dped
    _y_test = dataset_84541.growthrate.squeeze()
    # _x_test = dataset.features["dped"][test_data.indices]
    # _y_test = dataset.features["growthrate"][test_data.indices]
    """

    fig, axs = plt.subplots(1, 3, figsize=(12, 3))
    colors = [BLUE_VERY_LIGHT, BLUE_LIGHT, BLUE, BLUE_DARK, BLACK]
    cmap = mcolors.LinearSegmentedColormap.from_list("mycmap", colors, 5)

    xy_train = np.vstack([_x_train.T, _y_train.T])
    z_train = gaussian_kde(xy_train)(xy_train)
    xy_val = np.vstack([_x_val.T, _y_val.T])
    z_val = gaussian_kde(xy_val)(xy_val)
    xy_test = np.vstack([_x_test.T, _y_test.T])
    print(xy_test.shape)
    z_test = gaussian_kde(xy_test)(xy_test)

    sc0 = axs[0].scatter(_x_train, _y_train, c=z_train, cmap=cmap, vmin=0, vmax=400)
    stable_fraction = (
        np.where(_y_train < stability_threshold)[0].shape[0] / _y_train.shape[0]
    )
    axs[0].set_title(
        f"Training\n({_y_train.shape[0]} samples, {stable_fraction*100:.2f}% stable)"
    )
    axs[0].set_xlabel(r"$\Delta_{\text{ped}}$")
    axs[0].set_ylabel(r"$\gamma_{\text{max}}$")

    stable_fraction = (
        np.where(_y_val < stability_threshold)[0].shape[0] / _y_val.shape[0]
    )
    axs[1].set_title(
        f"Validation\n({_y_val.shape[0]} samples, {stable_fraction*100:.2f}% stable)"
    )
    sc1 = axs[1].scatter(_x_val, _y_val, c=z_val, cmap=cmap, vmin=0, vmax=400)
    axs[1].set_xlabel(r"$\Delta_{\text{ped}}$")
    axs[1].set_ylabel(r"$\gamma_{\text{max}}$")

    stable_fraction = (
        np.where(_y_test < stability_threshold)[0].shape[0] / _y_test.shape[0]
    )
    axs[2].set_title(
        f"Test\n({_y_test.shape[0]} samples, {stable_fraction*100:.2f}% stable)"
    )
    sc2 = axs[2].scatter(_x_test, _y_test, c=z_test, cmap=cmap, vmin=0, vmax=400)
    axs[2].set_xlabel(r"$\Delta_{\text{ped}}$")
    axs[2].set_ylabel(r"$\gamma_{\text{max}}$")

    axs[0].set_xlim(0.01, 0.065)
    axs[0].set_ylim(-0.05, 1.05)
    axs[1].set_xlim(0.01, 0.065)
    axs[1].set_ylim(-0.05, 1.05)
    axs[2].set_xlim(0.01, 0.065)
    axs[2].set_ylim(-0.05, 1.05)
    plt.colorbar(sc0)
    plt.colorbar(sc1)
    plt.colorbar(sc2)
    return fig


def plot_dataset_dped_gamma(dataset, scaled=True):
    """
    Plot the dataset with color coding based on the growth rate.
    Args:
        dataset: The dataset object containing the data to plot.
        scaled: Boolean indicating whether to use scaled values or not.
    """
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    _x = dataset.features["dped"]
    if scaled:
        _y = dataset.features["growthrate"]
    else:
        _y = dataset.descale_minmax(
            dataset.features["growthrate"],
            dataset.growthrate_scaler_min,
            dataset.growthrate_scaler_max,
        )
    xy = np.vstack([_x.T, _y.T])
    z = gaussian_kde(xy)(xy)
    ax.scatter(_x, _y, c=z)
    ax.set_xlabel(r"$\Delta_{\text{ped}}$")
    ax.set_ylabel(r"$\gamma_{\text{max}}$")

    return fig


def get_prediction_scores(y_true, y_pred):
    """
    Calculate prediction scores: MSE, MAE, MAPE, and R2.
    Args:
        y_true (array-like): True values.
        y_pred (array-like): Predicted values.
    Returns:
        mse (float): Mean Squared Error.
        mae (float): Mean Absolute Error.
        mape (float): Mean Absolute Percentage Error.
        r2 (float): R-squared score.
    """
    y_residuals = y_true - y_pred
    mse = np.mean(y_residuals**2)
    mae = np.mean(np.abs(y_residuals))
    mape = np.mean(np.abs((y_true - y_pred))) * 100
    r2 = r2_score(y_true, y_pred)
    return mse, mae, mape, r2


def print_prediction_scores(y_true, y_pred):
    """
    Print prediction scores: MSE, MAE, MAPE, and R2.
    """
    mse, mae, mape, r2 = get_prediction_scores(y_true, y_pred)
    print(f"MSE: {mse:.5f}\nMAE: {mae:.5f}\nMAPE: {mape:.5f}\nR2: {r2:.5f}")
    return mse, mae, mape, r2


def plot_pred_vs_true(y_test, y_pred, y_pred_std=None, filename=None, gridsize=60):
    """
    Plot predicted vs true values using a density (hexbin) plot.

    This visualization is designed for large datasets where scatter plots
    suffer from overplotting. The density of samples is shown using a 2D
    hexagonal histogram with logarithmic color scaling.

    Args:
        y_test (array-like): True target values.
        y_pred (array-like): Predicted mean values.
        y_pred_std (array-like, optional): Predictive standard deviation.
            If provided, the mean uncertainty is shown in the title and a
            faint scatter of predictions is added on top of the density.
        filename (str, optional): Path to save the figure.
        gridsize (int, optional): Number of hexagons in the x-direction.
            Higher values give finer resolution.

    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    y_test = np.asarray(y_test).squeeze()
    y_pred = np.asarray(y_pred).squeeze()

    if y_pred_std is not None:
        y_pred_std = np.asarray(y_pred_std).squeeze()

    metrics = get_regression_scores(y_true=y_test, y_pred=y_pred)
    mae, r2 = metrics["mae"], metrics["r2"]

    fig, ax = plt.subplots(figsize=(8, 5))

    # --- Density plot ---
    hb = ax.hexbin(
        y_test,
        y_pred,
        gridsize=gridsize,
        bins="log",          # log density
        cmap="viridis",
        mincnt=1
    )

    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label("log10(N samples per bin)")

    # Optional faint scatter to show spread / outliers
    if y_pred_std is not None:
        ax.scatter(y_test, y_pred, s=5, alpha=0.05, color="black")
        mean_std = np.mean(y_pred_std)
        title = f"R² = {r2:.5f}   MAE = {mae:.5f}   ⟨σ⟩ = {mean_std:.5f}"
    else:
        title = f"R² = {r2:.5f}   MAE = {mae:.5f}"

    # Ideal line
    min_val = min(np.min(y_test), np.min(y_pred))
    max_val = max(np.max(y_test), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Ideal")

    # Formatting
    ax.set_title(title)
    ax.set_xlabel(r"True")
    ax.set_ylabel(r"Predicted")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if filename:
        fig.savefig(filename, dpi=300)
        print(f"Figure saved: {filename}")

    return fig


def plot_pred_vs_true_colored(y_test, y_pred_mean, y_pred_std, filename=None):
    fig, ax = plt.subplots(figsize=(5, 4))

    sc = ax.scatter(
        y_test,
        y_pred_mean,
        c=y_pred_std,
        cmap="viridis",
        s=12,
        alpha=0.8
    )

    max_val = max(y_test.max(), y_pred_mean.max())
    ax.plot([0, max_val], [0, max_val], "r")

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Predictive σ")

    ax.set_xlabel(r"true $\gamma_{\max}$")
    ax.set_ylabel(r"predicted $\gamma_{\max}$")
    ax.set_title("Prediction colored by uncertainty")

    ax.grid(True)
    plt.tight_layout()
    if filename:
        fig.savefig(filename)
        print(f"Figure saved: {filename}")

    return fig


def plot_uncertainty_vs_error(y_pred_std, abs_error, filename=None):
    fig, ax = plt.subplots(figsize=(5, 4))

    ax.scatter(y_pred_std, abs_error, s=10, alpha=0.6)

    corr = np.corrcoef(y_pred_std, abs_error)[0, 1]

    ax.set_xlabel("Predictive σ")
    ax.set_ylabel("|Prediction error|")
    ax.set_title(f"σ vs error (corr = {corr:.3f})")

    ax.grid(True)
    plt.tight_layout()
    if filename:
        fig.savefig(filename)
        print(f"Figure saved: {filename}")

    return fig


def plot_coverage_curve(y_test, y_pred_mean, y_pred_std, filename=None):
    z_scores = np.abs(y_test - y_pred_mean) / y_pred_std

    conf_levels = np.linspace(0.1, 0.99, 30)
    empirical = [(z_scores < c).mean() for c in conf_levels]

    fig, ax = plt.subplots(figsize=(5, 4))

    ax.plot(conf_levels, empirical, label="Empirical")
    ax.plot(conf_levels, conf_levels, "--", label="Ideal")

    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Uncertainty calibration")
    ax.legend()

    ax.grid(True)
    plt.tight_layout()
    if filename:
        fig.savefig(filename)
        print(f"Figure saved: {filename}")
    return fig


def plot_uncertainty_histogram(y_pred_std, filename=None):
    fig, ax = plt.subplots(figsize=(5, 4))

    ax.hist(y_pred_std, bins=40, alpha=0.8)
    ax.set_xlabel("Predictive σ")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of ensemble uncertainty")

    ax.grid(True)
    plt.tight_layout()
    if filename:
        fig.savefig(filename)
        print(f"Figure saved: {filename}")
    return fig
