import time
import os
import json
import logging

# import json
import numpy as np
import torch
from torch.utils.data import Dataset
import mlflow

from karhu_training.addon_stopping import EarlyStopping
from karhu_training.utils_plotting import get_regression_scores

logger = logging.getLogger(__name__)


class DatasetEquilibriumGmax(Dataset):
    """ Extending the Dataset class. """

    def __init__(self, features):
        """Initialize the dataset with features.
        Args:
            features (dict): Dictionary of features where keys are feature names and values
            are numpy arrays.
        """
        self.features = features
        self.scaling_params = {}
        self.scaled = False

    def set_scaling_params(self, scaling_params):
        """
        Set the scaling parameters from a dict.
        """
        self.scaling_params = scaling_params
        return

    def set_scaling_params_from_data(self):
        """
        Set the scaling parameters for each feature based on the data.
        This method calculates the min and max values for each feature and stores
        them in the scalers dictionary.
        """
        logger.info("Setting scaling params from data...")
        if self.scaled:
            logger.warning("WARNING: Profiles have already been scaled.")
        self.scaling_params = {
            name: (torch.min(data).item(), torch.max(data).item())
            for name, data in self.features.items()
        }
        logger.info(f"Scaling params: {self.scaling_params}")
        return

    def set_scaling_params_from_dataset(self, other):
        """Set the scaling parameters from another dataset.
        This method copies the scaling parameters from another dataset instance.
        Args:
            other (DatasetEquilibriumGmax): Another dataset instance from which to
            copy scaling parameters.
        """
        logger.info("Setting scaling params from another dataset...")
        self.scaling_params = other.scalers.copy()
        logger.info(f"Scaling params: {self.scaling_params}")

    def get_scaling_params(self):
        """Get the scaling parameters for each feature.
        Returns:
            dict: Dictionary of scaling parameters where keys are feature names and values
            are tuples of (min, max).
        """
        scaling_params = {}
        for name, scaler in self.scaling_params.items():
            scaling_params[f"{name}_min"] = scaler[0]
            scaling_params[f"{name}_max"] = scaler[1]
        return scaling_params

    def scale_data(self):
        """Perform min-max scaling on the data"""
        logger.info("Scaling data...")
        if not self.scaled:
            for name, data in self.features.items():
                min_, max_ = self.scaling_params[name]
                if max_ != min_:
                    self.features[name] = (data - min_) / (max_ - min_)
            self.scaled = True
        else:
            print("Data already scaled.")

    def descale_data(self):
        """Undo the minmax scaling on the data"""
        if self.scaled:
            for name, data in self.features.items():
                min_, max_ = self.scaling_params[name]
                if max_ != min_:
                    self.features[name] = data * (max_ - min_) + min_
            self.scaled = False
        else:
            print("Data is not scaled.")

    def set_zeros_to_negative(self, negative_value: float = -1.0):
        """Set all zeros in the labels of the dataset to a negative value"""
        self.features["growthrate"][self.features["growthrate"] == 0.0] = negative_value

    def minmax(self, data, scaler_min, scaler_max):
        """Perform min-max scaling on the data.
        Args:
            data (torch.Tensor): Data to be scaled.
            scaler_min (float): Minimum value for scaling.
            scaler_max (float): Maximum value for scaling.
        Returns:
            torch.Tensor: Scaled data.
        """
        return (data - scaler_min) / (scaler_max - scaler_min)

    def descale_minmax(self, scaled_data, scaler_min, scaler_max):
        """Undo the min-max scaling on the data.
        Args:
            scaled_data (torch.Tensor): Scaled data to be descaled.
            scaler_min (float): Minimum value used for scaling.
            scaler_max (float): Maximum value used for scaling.
        Returns:
            torch.Tensor: Descaled data.
        """
        return scaled_data * (scaler_max - scaler_min) + scaler_min

    def __getitem__(self, index):
        return {name: data[index] for name, data in self.features.items()}

    def __len__(self):
        return len(next(iter(self.features.values())))

    def get_features_description(self):
        """Get a description of the features in the dataset.
        Returns:
            str: Description of the features and their shapes.
        """
        desc = []
        desc.append("Features in dataset:")
        for key, value in self.features.items():
            desc.append(f"  {key}: {value.shape}")
        return "\n".join(desc)

    def save_scaling_params(self, save_path):
        """Save the scaling parameters to a JSON file.
        Args:
            save_path (str): Path to save the JSON file.
        """
        with open(
            os.path.join(save_path, "scaling_params.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(self.scaling_params, f, ensure_ascii=False, indent=4)
        return


def train_model(
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    epochs=200,
    device: str = "cpu",
    early_stopping: bool = False,
    early_stopping_patience: int = 3,
    early_stopping_min_delta: float = 1e-3,
):
    """Train the model with the given parameters.
    Args:
        model: The model to be trained.
        optimizer: The optimizer for the model.
        criterion: The loss function.
        train_loader: DataLoader for the training data.
        val_loader: DataLoader for the validation data.
        epochs (int): Number of epochs to train the model.
        device (str): Device to use for training ('cpu' or 'cuda').
        early_stopping (bool): Whether to use early stopping.
        early_stopping_patience (int): Number of epochs with no improvement after
            which training will be stopped.
        early_stopping_min_delta (float): Minimum change in the monitored quantity
            to qualify as an improvement.
    Returns:
        tuple: Training and validation losses for each epoch.
    """
    start_time = time.time()
    train_losses, val_losses = [], []
    if early_stopping:
        print(
            f"Early stopping constraints: patience={early_stopping_patience}, "
            + f"min_delta={early_stopping_min_delta}"
        )
        # Early stopping
        early_stopper = EarlyStopping(
            patience=early_stopping_patience, delta=early_stopping_min_delta)

    loss_dict_epoch = {}
    for epoch in range(epochs):
        start_time_epoch = time.time()
        # Training phase
        model.train()
        running_loss = 0.0
        for data in train_loader:
            # Move inputs and labels to the device
            (input_p, input_qs, input_rbphi, input_shape, b_mag, r_mag, labels) = (
                model.get_input_from_batch_data(data, device)
            )
            optimizer.zero_grad()
            outputs = model(input_p, input_qs, input_rbphi, input_shape, b_mag, r_mag)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        # mlflow.log_metric("train_loss", train_loss, step=epoch)
        loss_dict_epoch[str(epoch)] = running_loss

        # Validation phase
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                # Move inputs and labels to the device
                (input_p, input_qs, input_rbphi, input_shape, b_mag, r_mag, labels) = (
                    model.get_input_from_batch_data(data, device))
                optimizer.zero_grad()
                outputs =  model(input_p, input_qs, input_rbphi, input_shape, b_mag, r_mag)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)
        val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        # mlflow.log_metric("val_loss", val_loss, step=epoch)
        logger.info(
            f"Epoch {epoch+1}/{epochs} - "
            + f"Train loss: {train_loss:.6f}, "
            + f"Validation loss: {val_loss:.6f}, "
            + f"Seconds: {time.time()-start_time_epoch:.2f}"
        )

        if early_stopping:   # and epoch > 200:
            early_stopper(val_loss, model, epoch=epoch)
            if early_stopper.early_stop:
                print("Early stopping")
                break

    # Load the best model
    if early_stopping:
        early_stopper.load_best_model(model)

    logger.info(f"Total training time (s): {time.time()-start_time:.2f}")
    return train_losses, val_losses


def test_model(model, test_loader, dataset, device: str = "cpu",):
    """ Get model prediction given a dataloader. """
    for data in test_loader:
        (input_p, input_qs, input_rbphi, input_shape, b_mag, r_mag, labels) = (
            model.get_input_from_batch_data(data, device)
        )
        y_pred = model(input_p, input_qs, input_rbphi, input_shape, b_mag, r_mag)

        y_test = labels
        y_test = torch.clamp(y_test, min=0.0)
        y_pred = torch.clamp(y_pred, min=0.0)

        y_pred = dataset.descale_minmax(
            y_pred,
            dataset.scaling_params["growthrate"][0],
            dataset.scaling_params["growthrate"][1],
        )
        y_test = dataset.descale_minmax(
            y_test,
            dataset.scaling_params["growthrate"][0],
            dataset.scaling_params["growthrate"][1],
        )

    return y_test.detach().numpy(), y_pred.detach().numpy()


def test_ensemble(models, test_loader, dataset, device: str = "cpu",):
    """
    Get model prediction given a dataloader.
    The data loader is assumed to have been initialized with batch_size
    the same size as the give dataset.
    """
    ensemble_preds = []
    y_test_ref = []

    i = 0
    for model in models:
        print(f"model {i}")
        for data in test_loader:
            (input_p, input_qs, input_rbphi, input_shape, b_mag, r_mag, labels) = (
                model.get_input_from_batch_data(data, device)
            )
            y_pred = model(input_p, input_qs, input_rbphi, input_shape, b_mag, r_mag)
            y_test = labels
        print(y_test.shape, y_pred.shape)
        y_test_ref.append(y_test.numpy().flatten())
        y_pred = torch.clamp(y_pred, min=0.0)
        ensemble_preds.append(y_pred.detach().numpy().flatten())
        i+=1

    y_test_ref = y_test_ref[0]
    ensemble_preds = np.stack(ensemble_preds, axis=0)
    print(y_test_ref.shape, ensemble_preds.shape)

    y_pred_mean = ensemble_preds.mean(axis=0)
    
    y_test_ref = np.clip(y_test_ref, a_min=0.0, a_max=None)

    y_pred_std  = ensemble_preds.std(axis=0)  # epistemic uncertainty

    y_pred_mean = dataset.descale_minmax(
        y_pred_mean,
        dataset.scaling_params["growthrate"][0],
        dataset.scaling_params["growthrate"][1],
    )
    y_pred_std = dataset.descale_minmax(
        y_pred_std,
        dataset.scaling_params["growthrate"][0],
        dataset.scaling_params["growthrate"][1],
    )
    y_test_ref = dataset.descale_minmax(
        y_test_ref,
        dataset.scaling_params["growthrate"][0],
        dataset.scaling_params["growthrate"][1],
    )

    print(y_test_ref.shape, y_pred_mean.shape, y_pred_std.shape)
    return y_test_ref, y_pred_mean, y_pred_std


def ensemble_predict(ensemble, x):
    """
    Epistemic uncertainty (uncertainty due to lack of knowledge or data).

    ensemble:
        List of models
    x:
        Input
    """
    with torch.no_grad():
        preds = torch.stack([model(*x) for model in ensemble])
        mean_pred = preds.mean(dim=0)
        std_pred = preds.std(dim=0)
        mean_pred = torch.clamp(mean_pred, min=0.0)
    return mean_pred, std_pred
