"""
Train a model.
"""

import torch
from torch import optim
from torch.utils.data import DataLoader

# Custom libraries
from karhu_training.models import GMaxPredictor, setup_dataset
from karhu_training.train import train_model, test_model
from karhu_training.utils_input import split_dataset
from karhu_training.utils_plotting import get_regression_scores

device = "cuda" if torch.cuda.is_available() else "cpu"
test_model.__test__ = False


def test_training_simple():
    epochs = 5
    dataset_dir = "./tests/data/interpolated64.h5"
    seed = 42
    lr = 0.01
    batch_size = 32

    model_config={
        "conv_input_sizes": [64, 64, 64, 128], # 128
        "scalar_inputs": 2,
        "conv_kernel_sizes": [7, 5, 3],
        "out_channels": 16,
        "pool_kernel_size": 2,
        "fc_hidden_dims": [128, 64],
        "conv_dropout": 0.1,
        "fc_dropout": 0.2,
    }
    model = GMaxPredictor(model_config)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Load dataset(s)
    dataset, data_config = setup_dataset(dir_path=dataset_dir)
    torch.manual_seed(seed)  # This changes for each ensemble member

    test_ratio = 0.10
    train_ratio = 0.70
    val_ratio = 0.20
    train_data, val_data, test_data = split_dataset(
        dataset, train_ratio, val_ratio, test_ratio, base_seed=seed)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)


    # Train model
    train_losses, val_losses = train_model(
        model, optimizer, model.loss_fn, train_loader, val_loader, epochs=epochs,
        early_stopping=False, early_stopping_min_delta=0.0001, early_stopping_patience=5
    )
    print(train_losses, val_losses)

    y_test, y_pred = test_model(model, test_loader, dataset=dataset, device=device)
    metrics = get_regression_scores(y_test, y_pred)

    print(metrics)
    assert metrics["mse"] > 0
    assert metrics["mae"] > 0
    assert metrics["r2"] > 0
