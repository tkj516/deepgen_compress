import torch
import torch.nn.functional as F
import numpy as np


class SpatialBernoulliLayer(torch.nn.Module):
    """Spatial Bernoulli input layer."""
    def __init__(self, in_size, out_channels, optimize_scale, dropout=None, quantiles_loc=None, uniform_loc=None):
        """
        Initialize a Spatial Bernoulli input layer.

        :param in_size: The size of the input tensor.
        :param out_channels: The number of output channels.
        """
        super(SpatialBernoulliLayer, self).__init__()
        self.in_size = in_size
        self.out_channels = out_channels

        # Insantiate epsilon
        self.eps = 1e-5

        # Instantiate the Bernoulli parameters
        self.prob = torch.nn.Parameter(
            torch.rand(out_channels, *self.in_size),
            requires_grad = True
        )

        # Initialize some useful constants
        self.register_buffer('zero', torch.zeros(1))
        self.register_buffer('nan', torch.tensor(np.nan))

    @property
    def in_channels(self):
        return self.in_size[0]

    @property
    def in_height(self):
        return self.in_size[1]

    @property
    def in_width(self):
        return self.in_size[2]

    @property
    def out_size(self):
        return self.out_channels, self.in_height, self.in_width

    def log_prob(self, x):

        return x*torch.log(self.prob) + (1-x)*torch.log(1-self.prob)

    def forward(self, x):
        """
        Evaluate the layer given some inputs.

        :param x: The inputs.
        :return: The tensor result of the layer.
        """

        # First clamp the probability
        self.prob.clamp_(min=self.eps, max=1.0-self.eps)

        # Compute the log-likelihoods
        x = torch.unsqueeze(x, dim=1)
        x = self.log_prob(x)

        # Marginalize missing values (denoted with NaNs)
        x = torch.where(torch.isnan(x), self.zero, x)

        # This implementation assumes independence between channels of the same pixel random variables
        return torch.sum(x, dim=2)


class SpatialIndicatorLayer(torch.nn.Module):
    """Spatial Indicator input layer."""
    def __init__(self, in_size, out_channels):
        """
        Initialize a Spatial Indicator input layer.

        :param in_size: The size of the input tensor.
        :param out_channels: The number of output channels.
        """
        super(SpatialIndicatorLayer, self).__init__()
        self.in_size = in_size
        self.out_channels = out_channels

        # Initialize some useful constants
        self.register_buffer('zero', torch.zeros(1))
        self.register_buffer('nan', torch.tensor(np.nan))

    @property
    def in_channels(self):
        return self.in_size[0]

    @property
    def in_height(self):
        return self.in_size[1]

    @property
    def in_width(self):
        return self.in_size[2]

    @property
    def out_size(self):
        return self.out_channels, self.in_height, self.in_width

    def forward(self, x):
        """
        Evaluate the layer given some inputs.

        :param x: The inputs.
        :return: The tensor result of the layer.
        """

        # Compute the one-hot (indicator) distribution: 0 where 1 and -inf where 0
        indicators = torch.where(torch.isnan(x), self.zero, x)
        indicators = torch.log(F.one_hot(indicators.long(), num_classes=self.out_channels).float())
        indicators = indicators.squeeze(1).permute(0, 3, 1, 2) # b, 2, h, w

        # Replace nan values in x with 0 and others with 1 - marginalization
        indicators.masked_fill_(torch.isnan(x), 0.0)

        return indicators

if __name__ == "__main__":

    x = (torch.rand(2, 1, 3, 3) > 0.5).float()
    x[..., -1, :] = float('nan')

    indicator_leaf = SpatialIndicatorLayer(in_size=(1, 28, 28), out_channels=2)
    
    print(x)
    print(indicator_leaf(x))
    print(torch.sum(indicator_leaf(x), dim=1))