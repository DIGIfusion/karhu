import time
import logging
import numpy as np
import torch

from karhu.addon_stopping import EarlyStopping


logger = logging.getLogger(__name__)


def train_model(
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    dataset_mappings: dict,
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
        early_stopping_patience (int): Number of epochs with no
            improvement after which training will be stopped.
        early_stopping_min_delta (float): Minimum change in the
            monitored quantity to qualify as an improvement.
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
            (input_p, input_qs, input_rbphi, input_shape, b_mag, labels) = (
                model.get_input_from_batch_data(data, device, dataset_mappings)
            )
            optimizer.zero_grad()
            outputs = model(input_p, input_qs, input_rbphi, input_shape, b_mag)
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
                (
                    input_p, input_qs, input_rbphi, input_shape,
                    b_mag, labels
                ) = (
                    model.get_input_from_batch_data(data, device))
                optimizer.zero_grad()
                outputs = model(
                    input_p, input_qs, input_rbphi, input_shape, b_mag)
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
        (input_p, input_qs, input_rbphi, input_shape, b_mag, labels) = (
            model.get_input_from_batch_data(data, device)
        )
        y_pred = model(
            input_p, input_qs, input_rbphi, input_shape, b_mag)

        y_test = labels

        y_test = y_test.detach().numpy()
        y_pred = y_pred.detach().numpy()
        y_test = np.maximum(0.0, y_test)
        y_pred = np.maximum(0.0, y_pred)

        y_pred = dataset.descale_minmax(
            y_pred,
            dataset.scaling_params["gamma"][0],
            dataset.scaling_params["gamma"][1],
        )
        y_test = dataset.descale_minmax(
            y_test,
            dataset.scaling_params["gamma"][0],
            dataset.scaling_params["gamma"][1],
        )

        if dataset.transformed:
            y_pred = dataset._detransform(y_pred, c=dataset.c)
            y_test = dataset._detransform(y_test, c=dataset.c)

    return y_test, y_pred


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
            (input_p, input_qs, input_rbphi, input_shape,
             b_mag, r_mag, labels) = (
                model.get_input_from_batch_data(data, device)
            )
            y_pred = model(
                input_p, input_qs, input_rbphi, input_shape, b_mag, r_mag)
            y_test = labels
        print(y_test.shape, y_pred.shape)
        y_pred = torch.clamp(y_pred, min=0.0)

        y_pred = y_pred.detach().numpy().flatten()
        y_test = y_test.detach().numpy().flatten()

        y_test_ref.append(y_test)
        ensemble_preds.append(y_pred)
        i += 1

    y_test_ref = y_test_ref[0]
    ensemble_preds = np.stack(ensemble_preds, axis=0)
    print(y_test_ref.shape, ensemble_preds.shape)

    y_pred_mean = ensemble_preds.mean(axis=0)

    y_test_ref = np.clip(y_test_ref, a_min=0.0, a_max=None)

    y_pred_std = ensemble_preds.std(axis=0)  # epistemic uncertainty

    y_pred_mean = dataset.descale_minmax(
        y_pred_mean,
        dataset.scaling_params["gamma"][0],
        dataset.scaling_params["gamma"][1],
    )
    y_pred_std = dataset.descale_minmax(
        y_pred_std,
        dataset.scaling_params["gamma"][0],
        dataset.scaling_params["gamma"][1],
    )
    y_test_ref = dataset.descale_minmax(
        y_test_ref,
        dataset.scaling_params["gamma"][0],
        dataset.scaling_params["gamma"][1],
    )

    if dataset.transformed:
        y_pred_mean = dataset._detransform(y_pred_mean, c=dataset.c)
        y_pred_std = dataset._detransform(y_pred_std, c=dataset.c)
        y_test_ref = dataset._detransform(y_test_ref, c=dataset.c)

    print(y_test_ref.shape, y_pred_mean.shape, y_pred_std.shape)
    return y_test_ref, y_pred_mean, y_pred_std


def train_model_eq(
    model,
    model_karhu,
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
        early_stopping_patience (int): Number of epochs with no
            improvement after which training will be stopped.
        early_stopping_min_delta (float): Minimum change in the
            monitored quantity to qualify as an improvement.
    Returns:
        tuple: Training and validation losses for each epoch.
    """
    train_losses, val_losses = [], []
    if early_stopping:
        print(
            f"Early stopping constraints: patience={early_stopping_patience}, "
            + f"min_delta={early_stopping_min_delta}"
        )
        # Early stopping
        early_stopper = EarlyStopping(
            patience=early_stopping_patience,
            delta=early_stopping_min_delta)

    loss_dict_epoch = {}
    for epoch in range(epochs):
        start_time = time.time()
        # Training phase
        model.train()
        running_loss = 0.0
        for data in train_loader:
            # Move inputs and labels to the device
            (input_ne, input_Te, input_Ti, input_shape,
             b_mag, r_mag, Ip, beta_n, zeff, labels, gamma2) = (
                model.get_input_from_batch_data(data, device)
            )
            optimizer.zero_grad()

            outputs = model(
                input_ne, input_Te, input_Ti, input_shape,
                b_mag, r_mag, Ip, beta_n, zeff)
            gamma_pred = model_karhu(
                outputs[:, 0].reshape((-1, 1, 64)),
                outputs[:, 1].reshape((-1, 1, 64)),
                outputs[:, 2].reshape((-1, 1, 64)),
                input_shape,
                b_mag,
                r_mag
                # beta_n
            )
            loss = criterion(gamma_pred, gamma2)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * gamma2.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        loss_dict_epoch[str(epoch)] = running_loss

        # Validation phase
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                # Move inputs and labels to the device
                (input_ne, input_Te, input_Ti, input_shape,
                 b_mag, r_mag, Ip, beta_n, zeff, labels, gamma2) = (
                    model.get_input_from_batch_data(data, device)
                )
                optimizer.zero_grad()

                outputs = model(
                    input_ne, input_Te, input_Ti, input_shape,
                    b_mag, r_mag, Ip, beta_n, zeff)
                gamma_pred = model_karhu(
                    outputs[:, 0].reshape((-1, 1, 64)),
                    outputs[:, 1].reshape((-1, 1, 64)),
                    outputs[:, 2].reshape((-1, 1, 64)),
                    input_shape,
                    b_mag,
                    r_mag
                    # beta_n
                )
                loss = criterion(gamma_pred, gamma2)
                # running_loss += loss.item() * labels.size(0)
                running_loss += loss.item() * gamma2.size(0)
        val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        logger.info(
            f"Epoch {epoch+1}/{epochs} - "
            + f"Train loss: {train_loss:.6f}, "
            + f"Validation loss: {val_loss:.6f}, "
            + f"Seconds: {time.time()-start_time:.2f}"
        )

        if early_stopping:   # and epoch > 200:
            early_stopper(val_loss, model, epoch=epoch)
            if early_stopper.early_stop:
                print("Early stopping")
                break

    # Load the best model
    if early_stopping:
        early_stopper.load_best_model(model)

    return train_losses, val_losses


def test_model_eq(
        model, model_karhu, test_loader, dataset, device: str = "cpu"):

    for data in test_loader:
        (input_ne, input_Te, input_Ti, input_shape,
         b_mag, r_mag, Ip, beta_n, zeff, labels, gamma2) = (
            model.get_input_from_batch_data(data, device)
        )

        y_pred = model(
            input_ne, input_Te, input_Ti, input_shape,
            b_mag, r_mag, Ip, beta_n, zeff)
        gamma_pred = model_karhu(
            y_pred[:, 0].reshape((-1, 1, 64)),
            y_pred[:, 1].reshape((-1, 1, 64)),
            y_pred[:, 2].reshape((-1, 1, 64)),
            input_shape,
            b_mag,
            r_mag
            # beta_n
        )
        y_test = labels
        y_pred[:, 0] = dataset.descale_minmax(
            y_pred[:, 0],
            dataset.scaling_params["p"][0],
            dataset.scaling_params["p"][1],
            )
        y_pred[:, 1] = dataset.descale_minmax(
            y_pred[:, 1],
            dataset.scaling_params["qs"][0],
            dataset.scaling_params["qs"][1],
            )
        y_pred[:, 2] = dataset.descale_minmax(
            y_pred[:, 2],
            dataset.scaling_params["rbphi"][0],
            dataset.scaling_params["rbphi"][1],
            )

        gamma2 = dataset.descale_minmax(
            gamma2,
            dataset.scaling_params["gamma"][0],
            dataset.scaling_params["gamma"][1],
            )
        gamma_pred = dataset.descale_minmax(
            gamma_pred,
            dataset.scaling_params["gamma"][0],
            dataset.scaling_params["gamma"][1],
            )

    return (
        y_test.detach().numpy(), y_pred.detach().numpy(),
        gamma2.detach().numpy(), gamma_pred.detach().numpy()
    )


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
