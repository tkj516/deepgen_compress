# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

from sklearn.covariance import log_likelihood
sys.path.append('..')

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tvtF
from layers.general_divisive_normalization import GeneralizedDivisiveNormalization
from torch import Tensor
from torch.nn.parameter import Parameter
from dgcspn.dgcspn import DgcSpn
from torch.distributions import Normal


def _conv(
    cin: int,
    cout: int,
    kernel_size: int,
    stride: int = 1,
) -> nn.Conv2d:
    return nn.Conv2d(
        cin, cout, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2
    )


def _deconv(
    cin: int,
    cout: int,
    kernel_size: int,
    stride: int = 1,
) -> nn.ConvTranspose2d:
    return nn.ConvTranspose2d(
        cin,
        cout,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )

# TODO: 04/22/2022
# I am going to increase the number of output channels by two to 
# model the mean and variance of the bottlenech Gaussian

class Encoder(nn.Module):
    """
    Image analysis network for the scale hyperprior model.

    Args:
        network_channels, compression_channels: see ScaleHyperprior
    """

    def __init__(self, network_channels: int, compression_channels: int):
        super().__init__()
        self.model = nn.Sequential(
            _conv(1, network_channels, kernel_size=5, stride=2),
            GeneralizedDivisiveNormalization(network_channels),
            _conv(network_channels, network_channels, kernel_size=5, stride=2),
            GeneralizedDivisiveNormalization(network_channels),
            _conv(network_channels, network_channels, kernel_size=5, stride=2),
            GeneralizedDivisiveNormalization(network_channels),
            _conv(network_channels, network_channels, kernel_size=5, stride=2),
        )

        self.mean_conv = _conv(network_channels, compression_channels, kernel_size=1, stride=1)
        self.log_scale_conv = _conv(network_channels, compression_channels, kernel_size=1, stride=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)

        return self.mean_conv(x), torch.exp(self.log_scale_conv(x))


class Decoder(nn.Module):
    """
    Image synthesis network for the scale hyperprior model.

    Args:
        network_channels, compression_channels: see ScaleHyperprior
    """

    def __init__(self, network_channels: int, compression_channels: int, num_base_distributions: int):
        super().__init__()
        self.model = nn.Sequential(
            _deconv(compression_channels, network_channels, kernel_size=5, stride=2),
            GeneralizedDivisiveNormalization(network_channels, inverse=True),
            _deconv(network_channels, network_channels, kernel_size=5, stride=2),
            GeneralizedDivisiveNormalization(network_channels, inverse=True),
            _deconv(network_channels, network_channels, kernel_size=5, stride=2),
            GeneralizedDivisiveNormalization(network_channels, inverse=True),
            _deconv(network_channels, network_channels, kernel_size=5, stride=2),
        )

        self.mean_conv = _conv(network_channels, num_base_distributions, kernel_size=1, stride=1)
        self.log_scale_conv = _conv(network_channels, num_base_distributions, kernel_size=1, stride=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)

        return self.mean_conv(x), self.log_scale_conv(x)

class VAE(nn.Module):

    def __init__(
        self,
        network_channels: int = 128,
        compression_channels: int = 256,
        num_base_distributions: int = 32,
        in_size: Tuple[int] = (1, 32, 32),
        sum_channels: int = 32.
    ):
        super().__init__()

        self.encoder = Encoder(network_channels, compression_channels)
        self.decoder = Decoder(network_channels, compression_channels, num_base_distributions)

        self.spn = DgcSpn(
            in_size=in_size,
            n_batch=num_base_distributions,
            sum_channels=sum_channels,
            depthwise=True,
        )

    def forward(self, images: Tensor) -> Tuple[Tensor, Tensor, Tensor]:

        mean, scale = self.encoder(images)  # (bs, compression_channels, h, w), (bs, compression_channels, h, w)

        z = self.sample(mean, scale)  # (bs, compression_channels, h, w)

        obs_mean, obs_log_scale = self.decoder(z)  # (bs, num_base_distributions, h, w)

        # Unsqueeze to feed to SPN
        obs_mean = obs_mean.unsqueeze(2)
        obs_log_scale = obs_log_scale.unsqueeze(2)

        log_likelihood = self.spn(images, obs_mean, obs_log_scale)

        log_prior = self.log_prior_prob(z)

        log_posterior = self.log_post_prob(z, mean, scale)

        elbo = log_likelihood + log_prior - log_posterior

        return -elbo, -log_likelihood

    def log_prior_prob(self, z):

        return torch.sum(Normal(0, 1).log_prob(z), dim=[1, 2, 3]).unsqueeze(-1)

    def log_post_prob(self, z, mean, scale):

        return torch.sum(Normal(mean, scale).log_prob(z), dim=[1, 2, 3]).unsqueeze(-1)

    def sample(self, mean, scale):

        eps = torch.randn_like(scale)

        return eps.mul(scale).add_(mean)

    def decode(self, z):

        obs_mean, obs_log_scale = self.decoder(z)  # (bs, compression_channels, h, w)

        return obs_mean, obs_log_scale