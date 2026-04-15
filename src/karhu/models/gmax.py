import os
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


def load_model(model_dir: str) -> tuple[torch.nn.Module, dict[str, np.ndarray]]:
    """ The model directory should contain
    - model.pt containing the weights
    - model_config.json containing the scaling parameters and interpolation axes for the inputs and outputs
    """

    # Load model config
    with open(os.path.join(model_dir, "model_config.json",), "r", encoding="utf-8",) as f:
        model_config = json.load(f)

    # Load model
    model = GMaxPredictor(model_config)
    model.load_state_dict(
        torch.load(os.path.join(model_dir, "model.pt"), weights_only=True,)
    )
    model.eval()
    return model, model_config


def load_ensemble_model(ensemble_dir: str) -> tuple[list[torch.nn.Module], dict[str, np.ndarray]]:
    """ The model directory should contain
    - model.pt containing the weights
    - model_config.json containing the scaling parameters and interpolation axes for the inputs and outputs
    """
    # Load model config
    with open(os.path.join(ensemble_dir, "model_config.json",), "r", encoding="utf-8",) as f:
        model_config = json.load(f)

    models = []
    
    model_dirs = [
        d for d in os.listdir(ensemble_dir)
        if os.path.isdir(os.path.join(ensemble_dir, d))
    ]

    for model_dir in model_dirs:

        # Load model
        model = GMaxPredictor(model_config)
        model.load_state_dict(
            torch.load(os.path.join(ensemble_dir, model_dir, "model.pt"), weights_only=True,)
        )
        model.eval()
        models.append(model)

    return models, model_config


def get_ensemble_prediction(models: list, model_inputs):
    """
    Given a list of models and a set of inputs, get the mean prediction
    and the standard deviation of the preditions. Note that the output
    is normalized and needs to be scaled back.
    """
    ensemble_preds = []

    for model in models:
        with torch.no_grad():
            y_pred_norm = model(*model_inputs)
        ensemble_preds.append(y_pred_norm)

    ensemble_preds = np.stack(ensemble_preds, axis=0)
    y_pred_mean = ensemble_preds.mean(axis=0)
    y_pred_std = ensemble_preds.std(axis=0)
    return y_pred_mean, y_pred_std


class GMaxPredictor(nn.Module):
    """
    GMaxPredictor model for predicting gmax from input sequences.
    The model consists of several convolutional layers followed by fully connected layers.
    The input consists of 4 sequences and 2 additional features (b_mag and r_mag).
    """

    def __init__(self, model_config: dict,):
        super().__init__()

        self.conv_input_sizes = model_config["conv_input_sizes"]
        self.scalar_inputs = model_config["scalar_inputs"]
        self.out_channels = model_config["out_channels"]
        self.conv_kernel_sizes = model_config["conv_kernel_sizes"]
        self.pool_kernel_size = model_config["pool_kernel_size"]
        self.fc_hidden_dims = model_config["fc_hidden_dims"]
        self.pool_kernel_size = model_config["pool_kernel_size"]
        self.pool = nn.MaxPool1d(kernel_size=self.pool_kernel_size, stride=2, padding=0)

        # --- Dropout layers ---
        self.conv_dropout = nn.Dropout1d(p=model_config["conv_dropout"])
        self.fc_dropout = nn.Dropout(p=model_config["fc_dropout"])

        # First 4 input conv layers (1 channel in)
        self.input_convs = nn.ModuleList(
            [
                nn.Conv1d(
                    1,
                    self.out_channels,
                    kernel_size=self.conv_kernel_sizes[0],
                    stride=1,
                    padding=self.conv_kernel_sizes[0] // 2,
                )
                for _ in range(4)
            ]
        )

        # Next 2 conv layers repeated 2 times each
        self.conv1s = nn.ModuleList(
            [
                nn.Conv1d(
                    self.out_channels,
                    self.out_channels,
                    kernel_size=self.conv_kernel_sizes[1],
                    stride=1,
                    padding=self.conv_kernel_sizes[1] // 2,
                )
                for _ in range(4)
            ]
        )

        self.conv2s = nn.ModuleList(
            [
                nn.Conv1d(
                    self.out_channels,
                    self.out_channels,
                    kernel_size=self.conv_kernel_sizes[2],
                    stride=1,
                    padding=self.conv_kernel_sizes[2] // 2,
                )
                for _ in range(4)
            ]
        )

        # Compute output size after 3 poolings
        self.flatten_dims = []
        for size in self.conv_input_sizes:
            conv_output = size // (2**3)
            self.flatten_dims.append(self.out_channels * conv_output)

        self.total_flatten_dim = sum(self.flatten_dims)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.total_flatten_dim + self.scalar_inputs, self.fc_hidden_dims[0])
        self.fc2 = nn.Linear(self.fc_hidden_dims[0], self.fc_hidden_dims[1])
        self.fc3 = nn.Linear(self.fc_hidden_dims[1], 1)

        self.loss_fn = nn.MSELoss()

    def forward(self, input_p, input_qs, input_rbphi, input_shape, b_mag, r_mag):
        inputs = [input_p, input_qs, input_rbphi, input_shape]
        features = []
        scalars = torch.stack([b_mag, r_mag], dim=1) 

        for i in range(4):
            x = F.leaky_relu(self.input_convs[i](inputs[i]))
            x = self.pool(x)
            x = self.conv_dropout(x)

            x = F.leaky_relu(self.conv1s[i](x))
            x = self.pool(x)
            x = self.conv_dropout(x)

            x = F.leaky_relu(self.conv2s[i](x))
            x = self.pool(x)
            x = self.conv_dropout(x)

            features.append(torch.flatten(x, start_dim=1))

        x = torch.cat(features, dim=1)

        # Concatenate scalars
        scalars = scalars.view(scalars.size(0), -1)
        x = torch.cat([x, scalars], dim=1)

        x = F.leaky_relu(self.fc1(x))
        x = self.fc_dropout(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.fc_dropout(x)
        x = self.fc3(x)

        return x
