import os
import torch
from torch.utils.data import DataLoader
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.air import RunConfig

from karhu.models import GMaxPredictor, setup_dataset
from torch.utils.data import random_split


# ---------------------------
# Training / Testing Loops
# ---------------------------
def train_epoch(model, optimizer, train_loader):
    model.train()
    criterion = model.loss_fn
    running_loss = 0.0
    for data in train_loader:
        # Move inputs and labels to the device
        (input_p, input_qs, input_rbphi, input_shape, b_mag, r_mag, labels) = (
            model.get_input_from_batch_data(data, "cpu")
        )
        optimizer.zero_grad()
        scalars = torch.stack([b_mag, r_mag], dim=1) 
        outputs = model(input_p, input_qs, input_rbphi, input_shape, scalars)
        # print(outputs, labels)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
    train_loss = running_loss / len(train_loader.dataset)
    # print(train_loss)


def test(model, test_loader):
    model.eval()
    criterion = model.loss_fn
    running_loss = 0.0
    with torch.no_grad():
        for data in test_loader:
            # Move inputs and labels to the device
            (input_p, input_qs, input_rbphi,
             input_shape, b_mag, r_mag, labels) = (
                model.get_input_from_batch_data(data, "cpu"))
            scalars = torch.stack([b_mag, r_mag], dim=1)
            outputs = model(
                input_p, input_qs, input_rbphi, input_shape, scalars)

            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)

    return running_loss / len(test_loader.dataset)


# ---------------------------
# Objective for Ray Tune
# ---------------------------
def objective(config):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = os.path.join(
        "../../..//karhu-training/models_raytune",
        '_'.join([f"{key}{value:.5f}" for key, value in config.items()])
    )
    os.makedirs(save_dir)
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
    dataset, _ = setup_dataset(
        dir_paths=[
            "../../../karhu-training/db/all/interpolated64.h5",
            ]
        )

    n_total = len(dataset)
    n_test = int(0.2 * n_total)
    n_train = int(n_total - n_test)
    g_test = torch.Generator().manual_seed(42)

    train_data, test_data = random_split(dataset, [n_train, n_test], generator=g_test)
    train_loader = DataLoader(train_data, batch_size=int(config["batch_size"]), shuffle=True)
    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        betas=(config["beta1"], config["beta2"]),
    )

    for epoch in range(50):
        train_epoch(model, optimizer, train_loader)
        loss = test(model, test_loader)
        print(f"[Trial {tune.get_trial_id()}] Epoch {epoch+1}: loss={loss:.4f}", flush=True)
        tune.report(mean_loss=loss)
        torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
        print(f"Saved to: {save_dir}", flush=True)
    return


# ---------------------------
# Define Search Space & Run Tuning
# ---------------------------
search_space = {
    "lr": tune.loguniform(1e-4, 1e-2),
    # "momentum": tune.uniform(0.1, 0.9),
    "weight_decay": tune.loguniform(1e-8, 1e-2),
    "beta1": tune.uniform(0.8, 0.99),
    "beta2": tune.uniform(0.95, 0.9999),
    "batch_size":  tune.choice([64, 128, 256, 512, 1024]),
}

algo = OptunaSearch()  # Use Bayesian optimization from Optuna

tuner = tune.Tuner(
    objective,
    run_config=RunConfig(
        stop={"training_iteration": 50},
        verbose=2,  # 0 = silent, 1 = normal, 2 = detailed
    ),
    tune_config=tune.TuneConfig(
        metric="mean_loss",
        mode="min",
        search_alg=algo,
        num_samples=30,  # number of hyperparameter trials
        max_concurrent_trials=1,
    ),
    param_space=search_space,
)

results = tuner.fit()
print("Best config is:", results.get_best_result().config)
print("Best mean loss:", results.get_best_result().metrics["mean_loss"])
