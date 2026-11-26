import time
import logging

# import json

import torch
from karhu_training.addon_stopping import EarlyStopping


logger = logging.getLogger(__name__)


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
        early_stopping_patience (int): Number of epochs with no improvement
        after which training will be stopped.
        early_stopping_min_delta (float): Minimum change in the monitored
         quantity to qualify as an improvement.
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
            patience=early_stopping_patience, delta=early_stopping_min_delta)

    loss_dict_epoch = {}
    for epoch in range(epochs):
        start_time = time.time()
        # Training phase
        model.train()
        running_loss = 0.0
        for data in train_loader:
            # Move inputs and labels to the device
            (input_p, input_qs, input_rbphi, input_shape,
             b_mag, r_mag, beta_n, labels) = (
                model.get_input_from_batch_data(data, device)
            )
            optimizer.zero_grad()

            (obs_rec, obs, post, post_loc, post_scal,
             aux_rec, aux, prior_state, prior_loc, prior_scal,
             gamma) = model(
                input_p, input_qs, input_rbphi, input_shape,
                b_mag, r_mag, beta_n)
            loss = criterion(
                obs, obs_rec, post, post_loc, post_scal,
                aux_rec, aux, prior_state, prior_loc, prior_scal,
                gamma, labels)
            loss = (
                loss['rec'] + loss['aux'] + loss['gamma']
                + loss['kl_prior'] + loss['kl_cond'])
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        loss_dict_epoch[str(epoch)] = running_loss

        # Validation phase
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                # Move inputs and labels to the device
                (input_p, input_qs, input_rbphi, input_shape,
                 b_mag, r_mag, beta_n, labels) = (
                    model.get_input_from_batch_data(data, device))
                optimizer.zero_grad()

                (obs_rec, obs, post, post_loc, post_scal,
                 aux_rec, aux, prior_state, prior_loc, prior_scal,
                 gamma) = model(
                    input_p, input_qs, input_rbphi, input_shape,
                    b_mag, r_mag, beta_n)
                loss = criterion(
                    obs, obs_rec, post, post_loc, post_scal,
                    aux_rec, aux, prior_state, prior_loc, prior_scal,
                    gamma, labels)
                loss = (
                    loss['rec'] + loss['aux'] + loss['gamma']
                    + loss['kl_prior'] + loss['kl_cond'])
                running_loss += loss.item() * labels.size(0)
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


def test_model(model, test_loader, dataset, device: str = "cpu",):
    for data in test_loader:
        (input_p, input_qs, input_rbphi, input_shape,
         b_mag, r_mag, beta_n, labels) = (
            model.get_input_from_batch_data(data, device)
        )
        (obs_rec, obs, post, post_loc, post_state_scal,
         aux_rec, aux, prior_state, prior_loc, prior_scal, gamma) = model(
            input_p, input_qs, input_rbphi, input_shape,
            b_mag, r_mag, beta_n)

        # y_test = labels
        # y_pred = dataset.descale_minmax(
        #    y_pred,
        #    dataset.scaling_params["growthrate"][0],
        #    dataset.scaling_params["growthrate"][1],
        # )
        # y_test = dataset.descale_minmax(
        #    y_test,
        #    dataset.scaling_params["growthrate"][0],
        #    dataset.scaling_params["growthrate"][1],
        # )

    return (
        obs.detach().numpy(), obs_rec.detach().numpy(),
        gamma.detach().numpy(), labels.detach().numpy()
    )
