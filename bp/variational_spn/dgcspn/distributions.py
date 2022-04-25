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


def log_min_exp(a, b, epsilon=1e-8):
    """
    Computes the log of exp(a) - exp(b) in a (more) numerically stable fashion.
    Using:
     log(exp(a) - exp(b))
     c + log(exp(a-c) - exp(b-c))
     a + log(1 - exp(b-a))
    And note that we assume b < a always.
    """
    y = a + torch.log(1 - torch.exp(b - a) + epsilon)

    return y


class SpatialDiscreteLogisticLayer(torch.nn.Module):
    """Spatial Discrete Logistic input layer."""
    def __init__(self, in_size, out_channels, inverse_width=2**8):
        """
        Initialize a Spatial Bernoulli input layer.

        :param in_size: The size of the input tensor.
        :param out_channels: The number of output channels.
        """
        super(SpatialDiscreteLogisticLayer, self).__init__()
        self.in_size = in_size
        self.out_channels = out_channels
        self.inv_width = inverse_width

        # Insantiate epsilon
        self.eps = 1e-5

        # Instantiate the mean and scale parameters
        self.mean = torch.nn.Parameter(
            torch.rand(out_channels, *self.in_size),
            requires_grad = True,
        )
        self.log_scale = torch.nn.Parameter(
            torch.rand(out_channels, *self.in_size),
            requires_grad = True,
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

    @property
    def logits(self):

        grid = torch.arange(0, self.inv_width).reshape(1, -1, 1, 1)
        grid = grid.repeat(self.out_channels, 1, self.in_height, self.in_width)
        return self.log_prob(grid)    

    def log_prob(self, x):

        scale = torch.exp(self.log_scale).unsqueeze(0)
        mean = self.mean.unsqueeze(0)
        if mean.isnan().any() == True or scale.isnan().any() == True:
            print("Nan detected!!!")
            exit(0)
        prob = log_min_exp(F.logsigmoid((x + 0.5 / self.inv_width - mean) / scale), 
                F.logsigmoid((x - 0.5 / self.inv_width - mean) / scale))

        return prob

    def forward(self, x):
        """
        Evaluate the layer given some inputs.

        :param x: The inputs.
        :return: The tensor result of the layer.
        """

        # Compute the log-likelihoods
        x = torch.unsqueeze(x, dim=1) / self.inv_width
        x = self.log_prob(x)

        # Marginalize missing values (denoted with NaNs)
        x = torch.where(torch.isnan(x), self.zero, x)

        # This implementation assumes independence between channels of the same pixel random variables
        return torch.sum(x, dim=2)


class SpatialGaussianLayer(torch.nn.Module):
    """Spatial Gaussian input layer."""
    def __init__(self, in_size, out_channels):
        """
        Initialize a Spatial Gaussian input layer.

        :param in_size: The size of the input tensor.
        :param out_channels: The number of output channels.
        """
        super(SpatialGaussianLayer, self).__init__()
        self.in_size = in_size
        self.out_channels = out_channels

        # Insantiate epsilon
        self.eps = 1e-5

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

    def log_prob(self, x, mean, log_scale):

        if mean.isnan().any() == True or log_scale.isnan().any() == True:
            print("Nan detected!!!")
            exit(0)

        var = torch.exp(2 * log_scale)

        log_prob = -((x - mean) ** 2) / (2 * var) - log_scale - 0.5 * np.log(2 * np.pi)

        return log_prob

    def cdf(self, x):

        phi = lambda x: 0.5 * (1 + torch.erf(x / np.sqrt(2)))

        return phi(x)

    def forward(self, x, mean, log_scale):
        """
        Evaluate the layer given some inputs.

        :param x: The inputs should be in range [-1, 1]
        :return: The tensor result of the layer.
        """

        # Compute the log-likelihoods
        x = torch.unsqueeze(x, dim=1)
        x = self.log_prob(x, mean, log_scale)

        # Marginalize missing values (denoted with NaNs)
        x = torch.where(torch.isnan(x), self.zero, x)

        # This implementation assumes independence between channels of the same pixel random variables
        return torch.sum(x, dim=2)