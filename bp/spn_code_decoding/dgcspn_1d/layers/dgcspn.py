import torch
import numpy as np
import torch.nn.functional as F

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
    def in_length(self):
        return self.in_size[1]

    @property
    def out_size(self):
        return self.out_channels, self.in_length

    def forward(self, x):
        """
        Evaluate the layer given some inputs.

        :param x: The inputs.
        :return: The tensor result of the layer.
        """

        # Compute the one-hot (indicator) distribution: 0 where 1 and -inf where 0
        indicators = torch.where(torch.isnan(x), self.zero, x)
        indicators = torch.log(F.one_hot(indicators.long(), num_classes=self.out_channels).float())
        indicators = indicators.squeeze(1).permute(0, 2, 1) # b, num_classes, l

        # Replace nan values in x with 0 and others with 1 - marginalization
        indicators.masked_fill_(torch.isnan(x), 0.0)

        return indicators

class SpatialGaussianLayer(torch.nn.Module):
    """Spatial Gaussian input layer."""
    def __init__(self, in_size, out_channels, optimize_scale, dropout=None, quantiles_loc=None, uniform_loc=None):
        """
        Initialize a Spatial Gaussian input layer.

        :param in_size: The size of the input tensor.
        :param out_channels: The number of output channels.
        :param optimize_scale: Whether to optimize scale.
        :param dropout: The leaf nodes dropout rate. It can be None.
        :param quantiles_loc: The mean quantiles for location initialization. It can be None.
        :param uniform_loc: The uniform range for location initialization. It can be None.
        """
        super(SpatialGaussianLayer, self).__init__()
        self.in_size = in_size
        self.out_channels = out_channels
        self.optimize_scale = optimize_scale
        self.dropout = dropout
        self.quantiles_loc = quantiles_loc
        self.uniform_loc = uniform_loc

        # Instantiate the location parameter
        if self.quantiles_loc is not None:
            self.loc = torch.nn.Parameter(self.quantiles_loc, requires_grad=False)
        elif self.uniform_loc is not None:
            low, high = self.uniform_loc
            linspace = torch.linspace(low, high, steps=self.out_channels).view(-1, 1, 1)
            self.loc = torch.nn.Parameter(linspace.repeat(1, *self.in_size), requires_grad=True)
        else:
            self.loc = torch.nn.Parameter(torch.randn(self.out_channels, *self.in_size), requires_grad=True)

        # Instantiate the scale parameter
        if self.optimize_scale:
            self.scale = torch.nn.Parameter(
                0.5 + 0.1 * torch.tanh(torch.randn(self.out_channels, *self.in_size)),
                requires_grad=True
            )
        else:
            self.scale = torch.nn.Parameter(
                torch.ones(self.out_channels, *self.in_size),
                requires_grad=False
            )

        # Instantiate the multi-batch normal distribution
        self.distribution = torch.distributions.Normal(self.loc, self.scale)

        # Initialize some useful constants
        self.register_buffer('zero', torch.zeros(1))
        self.register_buffer('nan', torch.tensor(np.nan))

    @property
    def in_channels(self):
        return self.in_size[0]

    @property
    def in_length(self):
        return self.in_size[1]

    @property
    def out_size(self):
        return self.out_channels, self.in_height, self.in_length

    def forward(self, x):
        """
        Evaluate the layer given some inputs.

        :param x: The inputs.
        :return: The tensor result of the layer.
        """
        # Compute the log-likelihoods
        x = torch.unsqueeze(x, dim=1)
        x = self.distribution.log_prob(x).float()

        # Apply the input dropout, if specified
        if self.training and self.dropout is not None:
            x = torch.where(torch.gt(torch.rand_like(x), self.dropout), x, self.nan)

        # Marginalize missing values (denoted with NaNs)
        x = torch.where(torch.isnan(x), self.zero, x)

        # This implementation assumes independence between channels of the same pixel random variables
        return torch.sum(x, dim=2)

class SpatialProductLayer(torch.nn.Module):
    """Spatial Product layer class."""
    def __init__(self, in_size, depthwise, kernel_size, padding, stride, dilation, rand_state=None):
        """
        Initialize a Spatial Product layer.

        :param in_size: The input tensor size. It should be (in_channels, in_length).
        :param depthwise: Whether to use depthwise convolutions.
        :param kernel_size: The size of the kernels.
        :param stride: The strides to use.
        :param padding: The padding mode to use. It can be 'valid', 'full' or 'final'.
        :param dilation: The space between the kernel points.
        :param rand_state: The random state used to generate the weight mask. It can be None if depthwise is True.
        """
        super(SpatialProductLayer, self).__init__()
        self.in_size = in_size
        self.depthwise = depthwise
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.rand_state = rand_state
        self.groups = self.in_channels if self.depthwise else 1
        self.out_channels = self.in_channels if self.depthwise else self.in_channels * np.prod(self.kernel_size)

        # Compute the effective kernel size (due to dilation)
        ds = self.dilation
        ks = self.kernel_size
        kas = (ks - 1) * ds + 1
        self.effective_kernel_size = kas

        # Compute the padding to apply
        if self.padding == 'valid':
            self.pad = [0, 0]
        elif self.padding == 'full':
            self.pad = [kas - 1, kas - 1]
        elif self.padding == 'final':
            self.pad = [(kas - 1) * 2 - self.in_length, 0]
        else:
            raise NotImplementedError('Unknown padding mode named ' + self.padding)

        # Build the convolution kernels
        if self.depthwise:
            weight = self._build_depthwise_kernels()
        else:
            weight = self._build_sparse_kernels(rand_state)

        # Initialize the weight tensor
        self.weight = torch.nn.Parameter(torch.tensor(weight), requires_grad=False)

    @property
    def in_channels(self):
        return self.in_size[0]

    @property
    def in_length(self):
        return self.in_size[1]

    @property
    def out_size(self):
        kas = self.effective_kernel_size
        out_length = self.pad[0] + self.pad[1] + self.in_length - kas + 1
        out_length = int(np.ceil(out_length / self.stride).item())
        return self.out_channels, out_length

    def forward(self, x):
        """
        Evaluate the layer given some inputs.

        :param x: The inputs.
        :return: The tensor result of the layer.
        """
        # Pad the input and compute the log-likelihoods
        x = torch.nn.functional.pad(x, self.pad)
        return torch.nn.functional.conv1d(
            x, self.weight, stride=self.stride, dilation=self.dilation, groups=self.groups
        )

    def _build_sparse_kernels(self, rand_state):
        """
        Generate sparse (and non-depthwise) convolution kernels randomly.

        :param rand_state: The random state used to generate the weight mask.
        :return: The convolution kernels.
        """
        kernels_idx = []
        ks = self.kernel_size

        # Generate a sparse representation of weight
        for _ in range(ks):
            idx = np.arange(self.out_channels) % self.in_channels
            rand_state.shuffle(idx)
            kernels_idx.append(idx)
        kernels_idx = np.stack(kernels_idx, axis=1)
        kernels_idx = np.reshape(kernels_idx, (self.out_channels, 1, ks))

        # Generate a dense representation of weight, given the sparse representation
        weight = np.ones((self.out_channels, self.in_channels, ks))
        weight = weight * np.arange(self.in_channels).reshape((1, self.in_channels, 1))
        weight = np.equal(weight, kernels_idx)
        return weight.astype(np.float32)

    def _build_depthwise_kernels(self):
        """
        Generate depthwise convolution kernels.

        :return: The convolution kernels.
        """
        ks = self.kernel_size
        return np.ones((self.out_channels, 1, ks)).astype(np.float32)


class SpatialSumLayer(torch.nn.Module):
    """Spatial Sum layer class."""
    def __init__(self, in_size, out_channels, dropout=None):
        """
        Initialize a Spatial Sum layer.

        :param in_size: The input tensor size. It should be (in_channels, in_length).
        :param out_channels: The number of output channels.
        :param dropout: The input nodes dropout rate. It can be None.
        """
        super(SpatialSumLayer, self).__init__()
        self.in_size = in_size
        self.out_channels = out_channels
        self.dropout = dropout

        # Initialize the weight tensor
        self.weight = torch.nn.Parameter(
            torch.normal(0.0, 1e-1, [self.out_channels, *self.in_size]),
            requires_grad=True
        )

    @property
    def in_channels(self):
        return self.in_size[0]

    @property
    def in_length(self):
        return self.in_size[1]

    @property
    def out_size(self):
        return self.out_channels, self.in_length

    def forward(self, x):
        """
        Evaluate the layer given some inputs.

        :param x: The inputs.
        :return: The tensor result of the layer.
        """
        # Apply the dropout, if specified
        if self.training and self.dropout is not None:
            x = x + torch.log(torch.floor(1.0 - self.dropout + torch.rand_like(x)))

        # Normalize the weight using log softmax
        w = torch.log_softmax(self.weight, dim=1)

        # Apply the logsumexp trick
        x = torch.unsqueeze(x, dim=1)
        return torch.logsumexp(x + w, dim=2)


class SpatialRootLayer(torch.nn.Module):
    """Spatial Root layer class."""
    def __init__(self, in_size, out_channels):
        """
        Initialize a Spatial Root layer.

        :param in_size: The input tensor size. It should be (in_channels, in_length).
        :param out_channels: The number of output channels.
        """
        super(SpatialRootLayer, self).__init__()
        self.in_size = in_size
        self.out_channels = out_channels

        # Initialize the weight tensor
        in_flatten_size = np.prod(self.in_size).item()
        self.weight = torch.nn.Parameter(
            torch.normal(0.0, 1e-1, [self.out_channels, in_flatten_size]),
            requires_grad=True
        )

    @property
    def in_channels(self):
        return self.in_size[0]

    @property
    def in_length(self):
        return self.in_size[1]

    @property
    def out_size(self):
        return self.out_channels,

    def forward(self, x):
        """
        Evaluate the layer given some inputs.

        :param x: The inputs.
        :return: The tensor result of the layer.
        """
        # Calculate the log likelihood using the "logsumexp" trick
        x = torch.flatten(x, start_dim=1)
        x = torch.unsqueeze(x, dim=1)              # (-1, 1, in_flatten_size)
        w = torch.log_softmax(self.weight, dim=1)  # (out_channels, in_flatten_size)
        x = torch.logsumexp(x + w, dim=-1)         # (-1, out_channels)
        return x
