import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os 
import json 

logger = logging.getLogger(__name__)

def load_model(model_dir: str):
    """ The model directory should contain
    - model.pt containing the weights
    - scaling_params.json containing the scaling parameters for the inputs and outputs
    """

    # Load scaling parameters
    with open(os.path.join(model_dir, "scaling_params.json",), "r", encoding="utf-8",) as f:
        scaling_params = json.load(f)

    # Load model
    model = CNN_gmax()
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
        input_size=64,  # e.g. 64
        conv_kernel_sizes=[7, 5, 3],
        out_channels=16,
        pool_kernel_size=2,
        fc_hidden_dims=[128, 64],
    ):
        super().__init__()

        self.input_size = input_size
        self.out_channels = out_channels
        self.conv_kernel_sizes = conv_kernel_sizes
        self.pool_kernel_size = pool_kernel_size
        self.fc_hidden_dims = fc_hidden_dims
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
                for _ in range(4)
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
                for _ in range(4)
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
                for _ in range(4)
            ]
        )

        # Compute output size after 3 poolings
        conv_output_size = input_size // (2**3)
        self.flatten_dim = out_channels * conv_output_size

        # Fully connected layers
        # Now +2 because of b_mag and r_mag
        self.fc1 = nn.Linear(self.flatten_dim * 4 + 2, fc_hidden_dims[0])
        self.fc2 = nn.Linear(fc_hidden_dims[0], fc_hidden_dims[1])
        self.fc3 = nn.Linear(fc_hidden_dims[1], 1)

    def forward(self, input_p, input_qs, input_rbphi, input_shape, b_mag, r_mag):
        inputs = [input_p, input_qs, input_rbphi, input_shape]
        features = []

        for i in range(4):
            x = F.leaky_relu(self.input_convs[i](inputs[i]))
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
        x = torch.cat([x, b_mag, r_mag], dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def get_model_description(self):
        desc = []
        desc.append("\n=== CNN_gmax Model Description ===")
        desc.append(f"Input size (per sequence): {self.input_size}")
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
            f"\nFlattened output per input after 3x pooling: {self.flatten_dim}"
        )
        desc.append(
            f"Total flattened feature size (4 inputs + b_mag + r_mag): {self.flatten_dim * 4 + 2}"
        )
        desc.append("")

        desc.append("Fully Connected Layers:")
        desc.append(
            f"  fc1: input={self.flatten_dim * 4 + 2}, output={self.fc1.out_features}"
        )
        desc.append(
            f"  fc2: input={self.fc1.out_features}, output={self.fc2.out_features}"
        )
        desc.append(
            f"  fc3: input={self.fc2.out_features}, output={self.fc3.out_features}"
        )
        desc.append("==================================")

        return "\n".join(desc)

