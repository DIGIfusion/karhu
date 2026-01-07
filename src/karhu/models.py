import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import logging

logger = logging.getLogger(__name__)


def load_model(model_dir: str) -> tuple[torch.nn.Module, dict[str, np.ndarray]]:
    """ The model directory should contain
    - model.pt containing the weights
    - model_config.json containing the scaling parameters and interpolation axes for the inputs and outputs
    """

    # Load model config
    with open(os.path.join(model_dir, "model_config.json",), "r", encoding="utf-8",) as f:
        model_config = json.load(f)

    scaling_params = model_config["scaling_params"]

    # Load model
    model = CNN_gmax(
        conv_input_sizes=[64, 64, 64, 128],
        scalar_inputs=2,
        conv_kernel_sizes=[7, 5, 3],
        out_channels=16,
        pool_kernel_size=2,
        fc_hidden_dims=[128, 64],
        classifier=False
    )
    model.load_state_dict(
        torch.load(os.path.join(model_dir, "model.pt"), weights_only=True,)
    )
    model.eval()
    return model, scaling_params


class CNN_gmax(nn.Module):
    """
    CNN_gmax model for predicting gmax from input sequences.
    The model consists of several convolutional layers followed by fully connected layers.
    The input consists of 4 sequences and 2 additional features (b_mag and r_mag).
    """

    def __init__(
        self,
        conv_input_sizes=[64, 64, 64, 64],
        scalar_inputs=2,
        conv_kernel_sizes=[7, 5, 3],
        out_channels=16,
        pool_kernel_size=2,
        fc_hidden_dims=[128, 64],
        classifier=False
    ):
        super().__init__()

        self.conv_input_sizes = conv_input_sizes
        self.scalar_inputs = scalar_inputs
        self.out_channels = out_channels
        self.conv_kernel_sizes = conv_kernel_sizes
        self.pool_kernel_size = pool_kernel_size
        self.fc_hidden_dims = fc_hidden_dims
        self.classifier = classifier
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel_size, stride=2, padding=0)

        # First 4 input conv layers (1 channel in)
        self.input_convs = nn.ModuleList(
            [
                nn.Conv1d(
                    1,
                    out_channels,
                    kernel_size=conv_kernel_sizes[0],
                    stride=1,
                    padding=conv_kernel_sizes[0] // 2,
                )
                for _ in range(len(self.conv_input_sizes))
            ]
        )

        # Next 2 conv layers repeated 2 times each
        self.conv1s = nn.ModuleList(
            [
                nn.Conv1d(
                    out_channels,
                    out_channels,
                    kernel_size=conv_kernel_sizes[1],
                    stride=1,
                    padding=conv_kernel_sizes[1] // 2,
                )
                for _ in range(len(self.conv_input_sizes))
            ]
        )

        self.conv2s = nn.ModuleList(
            [
                nn.Conv1d(
                    out_channels,
                    out_channels,
                    kernel_size=conv_kernel_sizes[2],
                    stride=1,
                    padding=conv_kernel_sizes[2] // 2,
                )
                for _ in range(len(self.conv_input_sizes))
            ]
        )

        # Compute output size after 3 poolings
        self.flatten_dims = []
        for size in conv_input_sizes:
            conv_output = size // (2**3)
            self.flatten_dims.append(out_channels * conv_output)

        self.total_flatten_dim = sum(self.flatten_dims)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.total_flatten_dim + self.scalar_inputs, fc_hidden_dims[0])
        self.fc2 = nn.Linear(fc_hidden_dims[0], fc_hidden_dims[1])
        self.fc3 = nn.Linear(fc_hidden_dims[1], 1)

        # Set regressor vs classifier specific attributes
        if self.classifier:
            self.loss_fn = nn.BCELoss()
        else:
            self.loss_fn = nn.MSELoss()

    def forward(self, input_p, input_qs, input_rbphi, input_shape, b_mag, r_mag, **kwargs):
        profiles = [input_p, input_qs, input_rbphi, input_shape]
        features = []

        for i in range(len(self.conv_input_sizes)):
            x = F.leaky_relu(self.input_convs[i](profiles[i]))
            x = self.pool(x)
            x = F.leaky_relu(self.conv1s[i](x))
            x = self.pool(x)
            x = F.leaky_relu(self.conv2s[i](x))
            x = self.pool(x)
            features.append(torch.flatten(x, start_dim=1))

        x = torch.cat(features, dim=1)

        # Concatenate b_mag and r_mag (reshaped to batch size x 1)
        b_mag = b_mag.view(-1, 1)
        r_mag = r_mag.view(-1, 1)
        if self.scalar_inputs == 3:
            beta_n = kwargs.get('beta_n')
            beta_n = beta_n.view(-1, 1)
        if self.scalar_inputs == 3:
            x = torch.cat([x, b_mag, r_mag, beta_n], dim=1)
        else:
            x = torch.cat([x, b_mag, r_mag], dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.classifier:
            x = self.fc3(x)
            x = F.sigmoid(x)
        else:
            x = self.fc3(x)

        return x

    def get_model_description(self):
        desc = []
        desc.append("\n=== CNN_gmax Model Description ===")
        desc.append(f"Input size (per sequence): {self.conv_input_sizes}")
        desc.append(f"Input size (per sequence): {self.scalar_inputs}")
        desc.append(f"Convolution kernel sizes: {self.conv_kernel_sizes}")
        desc.append(f"Output channels per conv layer: {self.out_channels}")
        desc.append(f"Pooling kernel size: {self.pool_kernel_size}")
        desc.append("")
        desc.append("Input Convolutions:")
        for i, conv in enumerate(self.input_convs):
            desc.append(
                f"  InputConv[{i}]: in_channels={conv.in_channels}, "
                f"out_channels={conv.out_channels}, "
                f"kernel_size={conv.kernel_size}, padding={conv.padding}"
            )
        desc.append("\nConv1 Layers (repeated per input):")
        for i, conv in enumerate(self.conv1s):
            desc.append(
                f"  Conv1[{i}]: in_channels={conv.in_channels}, out_channels={conv.out_channels}, "
                f"kernel_size={conv.kernel_size}, padding={conv.padding}"
            )
        desc.append("\nConv2 Layers (repeated per input):")
        for i, conv in enumerate(self.conv2s):
            desc.append(
                f"  Conv2[{i}]: in_channels={conv.in_channels}, out_channels={conv.out_channels}, "
                f"kernel_size={conv.kernel_size}, padding={conv.padding}"
            )
        desc.append(
            f"\nFlattened output per input after 3x pooling: {self.flatten_dims}"
        )
        desc.append(
            f"Total flattened feature size (4 inputs + b_mag + r_mag): {self.total_flatten_dim + self.scalar_inputs}"
        )
        desc.append("")
        desc.append("Fully Connected Layers:")
        desc.append(
            f"  fc1: input={self.total_flatten_dim + self.scalar_inputs}, output={self.fc1.out_features}"
        )
        desc.append(
            f"  fc2: input={self.fc1.out_features}, output={self.fc2.out_features}"
        )
        desc.append(
            f"  fc3: input={self.fc2.out_features}, output={self.fc3.out_features}"
        )
        desc.append("==================================")
        return "\n".join(desc)
