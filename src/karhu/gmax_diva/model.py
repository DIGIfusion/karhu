import os
import json
import logging
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import h5py

from karhu_training.train import DatasetEquilibriumGmax

logger = logging.getLogger(__name__)


class GMaxDIVA(nn.Module):
    """
    TODO.
    """

    def __init__(self):  # , model_config: dict,):
        super().__init__()

        # Numerical stability
        self.min_std_dev = 1e-10

        # Loss multipliers
        self.rec_loss_mul = 1.0e-4
        self.prior_loss_mul = 1.0e-7
        self.prior_rec_loss_mul = 1.0e-4
        self.cond_prior_loss_mul = 1.0e-5
        self.diff_loss_mul = 1.0e-5
        self.gamma_loss = 1.0

        # Dimensions
        self.state_size = 20
        self.in_channels = 3
        self.out_spatial_dim = 64
        self.embedding_size = 168

        # Conv block
        self.conv_block = nn.ModuleList()
        self.conv_block.append(
            nn.Conv1d(self.in_channels, 6, kernel_size=4, stride=2))
        self.conv_block.append(nn.Conv1d(6, 12, kernel_size=4, stride=2))
        # self.conv_block.append(nn.Conv1d(12, 24, kernel_size=4, stride=2))
        self.hidden_output_size = 168

        # Trans conv block
        self.trans_conv_block = nn.ModuleList()
        self.trans_conv_block.append(
            nn.ConvTranspose1d(self.state_size, 16, kernel_size=5, stride=3))
        self.trans_conv_block.append(
            nn.ConvTranspose1d(16, 12, kernel_size=5, stride=3))
        self.trans_conv_block.append(
            nn.ConvTranspose1d(12, 6, kernel_size=5, stride=3))
        self.trans_conv_block.append(
            nn.ConvTranspose1d(6, 3, kernel_size=5, stride=3))

        # Shape conv block
        self.sconv_block = nn.ModuleList()
        self.sconv_block.append(nn.Conv1d(1, 6, kernel_size=4, stride=2))
        self.sconv_block.append(nn.Conv1d(6, 12, kernel_size=4, stride=2))
        self.shidden_output_size = 360

        # Shape trans conv block
        self.strans_conv_block = nn.ModuleList()
        self.strans_conv_block.append(
            nn.ConvTranspose1d(
                self.embedding_size, 16, kernel_size=5, stride=3))
        self.strans_conv_block.append(
            nn.ConvTranspose1d(16, 12, kernel_size=5, stride=3))
        self.strans_conv_block.append(
            nn.ConvTranspose1d(12, 6, kernel_size=5, stride=3))
        self.strans_conv_block.append(
            nn.ConvTranspose1d(6, 1, kernel_size=5, stride=3))

        # Latent space layers
        self.locs = nn.Linear(self.hidden_output_size, self.state_size)
        self.scales = nn.Linear(self.hidden_output_size, self.state_size)
        self.slocs = nn.Linear(self.shidden_output_size, self.state_size)
        self.sscales = nn.Linear(self.shidden_output_size, self.state_size)

        # Gamma predictor
        self.gamma_fc = nn.Linear(self.state_size, 20)
        self.gamma_fc1 = nn.Linear(20, 20)
        self.gamma_fc2 = nn.Linear(20, 1)

        # Output fc and embedding
        self.fc_embedding = nn.Linear(self.state_size, self.embedding_size)
        self.fc_out = nn.Linear(161, 64)  # self.out_spatial_dim)
        self.fcs_out = nn.Linear(161, 2*self.out_spatial_dim + 3)

        self.loss_fn = nn.MSELoss()

    def encoder_1d(self, observation: torch.Tensor):
        for lay in self.conv_block:
            observation = F.relu(lay(observation))
        observation = observation.view(-1, self.hidden_output_size)
        return self.locs(observation), self.scales(observation)

    def decoder_1d(self, state: torch.Tensor):
        # state = self.fc_embedding(state)
        # state = state.view(-1, self.embedding_size, 1)
        state = state.squeeze()
        state = state.unsqueeze(2)
        for lay in self.trans_conv_block:
            state = F.relu(lay(state))
        state = self.fc_out(state)
        observation = state
        # observation = F.interpolate(state, size=64)
        return observation

    def prior_regression(
            self, shape: torch.Tensor,
            b_mag: torch.Tensor, r_mag: torch.Tensor,
            beta_n: torch.Tensor):
        for lay in self.sconv_block:
            shape = F.relu(lay(shape))
        shape = shape.view(-1, self.shidden_output_size)
        # features = torch.cat((shape, b_mag, r_mag, beta_n), axis=1)
        return self.slocs(shape), self.sscales(shape)

    def aux_regression(self, state: torch.Tensor):
        state = self.fc_embedding(state)
        state = state.view(-1, self.embedding_size, 1)
        # state = state.squeeze().T
        for lay in self.strans_conv_block:
            state = F.relu(lay(state))
        # shape = F.interpolate(state, size=130)
        shape = self.fcs_out(state)
        return shape

    def gamma_predictor(self, state: torch.Tensor):
        state = F.relu(self.gamma_fc(state))
        state = F.relu(self.gamma_fc1(state))
        return self.gamma_fc2(state)

    def forward(
            self, input_p, input_qs, input_rbphi, input_shape,
            b_mag, r_mag, beta_n):
        observation = torch.stack(
            (input_p, input_qs, input_rbphi), axis=1).squeeze()
        posterior_loc, posterior_scal = self.encoder_1d(observation)
        posterior_state_scales = F.softplus(posterior_scal) + self.min_std_dev
        posterior_state = (
            posterior_loc
            + posterior_state_scales*torch.randn_like(posterior_loc))
        observation_rec = self.decoder_1d(posterior_state)

        prior_loc, prior_scal = self.prior_regression(
            input_shape, b_mag, r_mag, beta_n)
        prior_scales = F.softplus(prior_scal) + self.min_std_dev
        prior_state = prior_loc + prior_scales*torch.randn_like(prior_loc)
        aux_reg = self.aux_regression(prior_state).squeeze()

        shape = input_shape.view(-1, 128)
        aux = torch.cat((shape, b_mag, r_mag, beta_n), axis=1)
        aux = aux.squeeze()
        gamma = self.gamma_predictor(posterior_state)

        return (
            observation_rec, observation, posterior_state,
            posterior_loc, posterior_state_scales,
            aux_reg, aux, prior_state, prior_loc, prior_scales, gamma)

    def loss_fn(
            self, obs, obs_rec, posterior_state, posterior_locs,
            posterior_state_scales, aux_reg, aux,
            prior_state, prior_loc, prior_scales, gamma_predict, gamma):

        # diffloss = (torch.sum(
        #     torch.abs(torch.diff(torch.diff(torch.diff(obs_rec)))))
        #     * self.diff_loss_mul)

        rec_loss = torch.sum(
            self.rec_loss_mul*F.mse_loss(obs, obs_rec, reduction='none'))
        kl_prior_loss = torch.sum(
            self.prior_loss_mul*dist.kl_divergence(
                dist.Normal(posterior_locs, posterior_state_scales),
                dist.Normal(0, 1)))
        aux_reg_loss = torch.sum(
            self.prior_rec_loss_mul*F.mse_loss(aux, aux_reg, reduction='none'))
        kl_prior_cond_loss = torch.sum(
            self.cond_prior_loss_mul*dist.kl_divergence(
                dist.Normal(posterior_locs, posterior_state_scales),
                dist.Normal(prior_loc, prior_scales)))
        gamma_loss = self.gamma_loss*torch.sum(
            F.mse_loss(gamma_predict, gamma, reduction='none'))

        loss_dict = {
            'rec': rec_loss,
            'aux': aux_reg_loss,
            'gamma': gamma_loss,
            'kl_prior': kl_prior_loss,
            'kl_cond': kl_prior_cond_loss
        }
        return loss_dict

    def get_input_from_batch_data(self, data, device):
        """
        data:
            Batch from DataLoader
        decive:
            cpu or gpu
        """
        (input_p, input_qs, input_rbphi, input_shape,
         b_mag, r_mag, beta_n, labels) = (
            data["p"].to(device),
            data["qs"].to(device),
            data["rbphi"].to(device),
            data["shape"].to(device),
            data["b_mag"].to(device),
            data["r_mag"].to(device),
            data["beta_n"].to(device),
            data["gamma"].to(device),
        )
        return (
            input_p, input_qs, input_rbphi, input_shape,
            b_mag, r_mag, beta_n, labels)


def load_model(
    model_dir: str
) -> tuple[torch.nn.Module, dict[str, np.ndarray]]:
    """ The model directory should contain
    - model.pt containing the weights
    - model_config.json containing the scaling parameters and interpolation
      axes for the inputs and outputs
    """

    # Load model config
    with open(
            os.path.join(model_dir, "model_config.json",),
            "r", encoding="utf-8",) as f:
        model_config = json.load(f)

    # Load model
    model = GMaxDIVA(model_config)
    model.load_state_dict(
        torch.load(os.path.join(model_dir, "model.pt"), weights_only=True,)
    )
    model.eval()
    return model, model_config


def load_ensemble_model(
    ensemble_dir: str
) -> tuple[list[torch.nn.Module], dict[str, np.ndarray]]:
    """ The model directory should contain
    - model.pt containing the weights
    - model_config.json containing the scaling parameters and interpolation
      axes for the inputs and outputs
    """
    # Load model config
    with open(
            os.path.join(ensemble_dir, "model_config.json",),
            "r", encoding="utf-8",) as f:
        model_config = json.load(f)

    models = []
    model_dirs = [
        d for d in os.listdir(ensemble_dir)
        if os.path.isdir(os.path.join(ensemble_dir, d))
    ]

    for model_dir in model_dirs:

        # Load model
        model = GMaxDIVA(model_config)
        model.load_state_dict(
            torch.load(
                os.path.join(ensemble_dir, model_dir, "model.pt"),
                weights_only=True,)
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
