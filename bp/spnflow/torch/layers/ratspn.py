import abc
import torch
import numpy as np


class RegionGraphLayer(abc.ABC, torch.nn.Module):
    """Abstract class for input distributions layers."""
    def __init__(self, in_features, out_channels, regions, rg_depth, dropout):
        super(RegionGraphLayer, self).__init__()
        self.in_features = in_features
        self.in_regions = len(regions)
        self.out_channels = out_channels
        self.regions = regions
        self.rg_depth = rg_depth
        self.dropout = dropout
        self.distribution = None

        # Compute the padding and the number of features for each base distribution batch
        self.pad = -self.in_features % (2 ** self.rg_depth)
        self.dimension = (self.in_features + self.pad) // (2 ** self.rg_depth)

        # Append dummy variables to regions orderings and update the pad mask
        mask = self.regions.copy()
        if self.pad > 0:
            pad_mask = np.full([self.in_regions, 1, self.dimension], True)
            for i in range(self.in_regions):
                n_dummy = self.dimension - len(self.regions[i])
                if n_dummy > 0:
                    pad_mask[i, :, -n_dummy:] = False
                    mask[i] = tuple(mask[i]) + (mask[i][-1],) * n_dummy
            self.register_buffer('pad_mask', torch.tensor(pad_mask))
        self.register_buffer('mask', torch.tensor(mask))

        # Build the flatten inverse mask
        self.in_features_pad = self.in_features + self.pad
        inv_mask = torch.argsort(torch.reshape(self.mask, [-1, self.in_features_pad]))
        self.register_buffer('inv_mask', inv_mask)

        # Build the flatten inverted pad mask
        if self.pad > 0:
            inv_pad_mask = torch.reshape(self.pad_mask, [-1, self.in_features_pad])
            inv_pad_mask = torch.gather(inv_pad_mask, dim=1, index=self.inv_mask)
            self.register_buffer('inv_pad_mask', inv_pad_mask)

        # Initialize some useful constants
        self.register_buffer('zero', torch.zeros(1))
        self.register_buffer('nan', torch.tensor([np.nan]))

    def forward(self, x):
        """
        Execute the layer on some inputs.

        :param x: The inputs.
        :return: The log likelihood of each distribution leaf.
        """
        # Gather the inputs and compute the log-likelihoods
        x = torch.unsqueeze(x[:, self.mask], dim=2)
        x = self.distribution.log_prob(x)

        # Apply the input dropout, if specified
        if self.training and self.dropout is not None:
            x = torch.where(torch.gt(torch.rand_like(x), self.dropout), x, self.nan)

        # Marginalize missing values (denoted with NaNs)
        x = torch.where(torch.isnan(x), self.zero, x)

        # Pad to zeros
        if self.pad > 0:
            x = torch.where(self.pad_mask, x, self.zero)
        return torch.sum(x, dim=-1)

    @torch.no_grad()
    def mpe(self, x, idx_group, idx_offset):
        """
        Evaluate the layer given some inputs for maximum at posteriori estimation.

        :param x: The inputs. Random variables can be marginalized using NaN values.
        :param idx_group: The group indices.
        :param idx_offset: The offset indices.
        :return: The samples having maximum at posteriori estimates on marginalized random variables.
        """
        n_samples = idx_group.size(0)

        # Get the maximum at posteriori estimation of the base distribution
        # and filter the base samples by the region and offset indices
        samples = torch.flatten(self.distribution.mean[idx_group, idx_offset], start_dim=1)

        # Reorder the samples
        idx_repetitions = idx_group[:, 0] // (2 ** self.rg_depth)
        samples = torch.gather(samples, dim=1, index=self.inv_mask[idx_repetitions])

        # Remove the padding, if required
        if self.pad > 0:
            samples = samples[self.inv_pad_mask[idx_repetitions]].view(n_samples, self.in_features)

        # Assign the maximum at posteriori estimation to NaN random variables
        return torch.where(torch.isnan(x), samples, x)

    @torch.no_grad()
    def sample(self, idx_group, idx_offset):
        """
        Sample from a base distribution.

        :param idx_group: The group indices.
        :param idx_offset: The offset indices.
        :return: The computed samples.
        """
        n_samples = idx_group.size(0)

        # Sample from the base distribution and filter the samples
        samples = self.distribution.sample([n_samples])
        samples = samples[torch.unsqueeze(torch.arange(n_samples), dim=1), idx_group, idx_offset]
        samples = torch.flatten(samples, start_dim=1)

        # Reorder the samples
        idx_repetitions = idx_group[:, 0] // (2 ** self.rg_depth)
        samples = torch.gather(samples, dim=1, index=self.inv_mask[idx_repetitions])

        # Remove the padding, if required
        if self.pad > 0:
            samples = samples[self.inv_pad_mask[idx_repetitions]].view(n_samples, self.in_features)
        return samples


class GaussianLayer(RegionGraphLayer):
    """The Gaussian distributions input layer class."""
    def __init__(self, in_features, out_channels, regions, rg_depth, dropout, uniform_loc=None, optimize_scale=True):
        """
        Initialize a Gaussian distributions input layer.

        :param in_features: The number of input features.
        :param out_channels: The number of channels for each base distribution layer.
        :param regions: The regions of the distributions.
        :param rg_depth: The depth of the region graph.
        :param dropout: The leaf nodes dropout rate. It can be None.
        :param uniform_loc: The optional uniform distribution parameters for location initialization.
        :param optimize_scale: Whether to optimize scale and location jointly.
        """
        super(GaussianLayer, self).__init__(in_features, out_channels, regions, rg_depth, dropout)
        self.uniform_loc = uniform_loc
        self.optimize_scale = optimize_scale

        # Instantiate the location variable
        if uniform_loc is None:
            self.loc = torch.nn.Parameter(
                torch.randn(self.in_regions, self.out_channels, self.dimension),
                requires_grad=True
            )
        else:
            a, b = uniform_loc
            self.loc = torch.nn.Parameter(
                a + (b - a) * torch.rand(self.in_regions, self.out_channels, self.dimension),
                requires_grad=True
            )

        # Instantiate the scale variable
        if self.optimize_scale:
            self.scale = torch.nn.Parameter(
                0.5 + 0.1 * torch.tanh(torch.randn(self.in_regions, self.out_channels, self.dimension)),
                requires_grad=True
            )
        else:
            self.scale = torch.nn.Parameter(
                torch.ones(self.in_regions, self.out_channels, self.dimension),
                requires_grad=False
            )

        # Instantiate the multi-batch normal distribution
        self.distribution = torch.distributions.Normal(self.loc, self.scale)


class BernoulliLayer(RegionGraphLayer):
    """The Bernoulli distributions input layer class."""
    def __init__(self, in_features, out_channels, regions, rg_depth, dropout):
        """
        Initialize a Bernoulli distributions input layer.

        :param in_features: The number of input features.
        :param out_channels: The number of channels for each base distribution layer.
        :param regions: The regions of the distributions.
        :param rg_depth: The depth of the region graph.
        :param dropout: The leaf nodes dropout rate. It can be None.
        """
        super(BernoulliLayer, self).__init__(in_features, out_channels, regions, rg_depth, dropout)

        # Instantiate the logit variabel
        self.logits = torch.nn.Parameter(
            torch.randn(self.in_regions, self.out_channels, self.dimension),
            requires_grad=True
        )

        # Instantiate the multi-batch Bernoulli distribution
        self.distribution = torch.distributions.Bernoulli(logits=self.logits)


class ProductLayer(torch.nn.Module):
    """Product node layer class."""
    def __init__(self, in_regions, in_nodes):
        """
        Initialize the Product layer.

        :param in_regions: The number of input regions.
        :param in_nodes: The number of input nodes per region.
        """
        super(ProductLayer, self).__init__()
        self.in_regions = in_regions
        self.in_nodes = in_nodes
        self.out_partitions = in_regions // 2
        self.out_nodes = in_nodes ** 2

        # Initialize the mask used to compute the outer product
        mask = [True, False] * self.out_partitions
        self.register_buffer('mask', torch.tensor(mask))

    def forward(self, x):
        """
        Evaluate the layer given some inputs.

        :param x: The inputs.
        :return: The tensor result of the layer.
        """
        # Compute the outer product (the "outer sum" in log domain)
        x1 = x[:,  self.mask]                                # (-1, out_partitions, in_nodes)
        x2 = x[:, ~self.mask]                                # (-1, out_partitions, in_nodes)
        x1 = torch.unsqueeze(x1, dim=3)                      # (-1, out_partitions, in_nodes, 1)
        x2 = torch.unsqueeze(x2, dim=2)                      # (-1, out_partitions, 1, in_nodes)
        x = x1 + x2                                          # (-1, out_partitions, in_nodes, in_nodes)
        x = x.view(-1, self.out_partitions, self.out_nodes)  # (-1, out_partitions, out_nodes)
        return x

    @torch.no_grad()
    def mpe(self, x, idx_group, idx_offset):
        """
        Evaluate the layer given some inputs for maximum at posteriori estimation.

        :param x: The inputs (not used here).
        :param idx_group: The group indices.
        :param idx_offset: The offset indices.
        :return: The group and offset indices.
        """
        return self.sample(idx_group, idx_offset)

    @torch.no_grad()
    def sample(self, idx_group, idx_offset):
        """
        Sample from a product layer.

        :param idx_group: The group indices.
        :param idx_offset: The offset indices.
        :return: The group and offset indices.
        """
        # Compute the corresponding group and offset indices
        idx_group = torch.flatten(
            torch.stack([idx_group * 2, idx_group * 2 + 1], dim=2),
            start_dim=1
        )
        idx_offset = torch.flatten(
            torch.stack([idx_offset // self.in_nodes, idx_offset % self.in_nodes], dim=2),
            start_dim=1
        )
        return idx_group, idx_offset


class SumLayer(torch.nn.Module):
    """Sum node layer."""
    def __init__(self, in_partitions, in_nodes, out_nodes, dropout=None):
        """
        Initialize the sum layer.

        :param in_partitions: The number of input partitions.
        :param in_nodes: The number of input nodes per partition.
        :param out_nodes: The number of output nodes per region.
        :param dropout: The input nodes dropout rate. It can be None.
        """
        super(SumLayer, self).__init__()
        self.in_partitions = in_partitions
        self.in_nodes = in_nodes
        self.out_regions = in_partitions
        self.out_nodes = out_nodes
        self.dropout = dropout

        # Instantiate the weights
        self.weight = torch.nn.Parameter(
            torch.normal(0.0, 1e-1, [self.out_regions, self.out_nodes, self.in_nodes]),
            requires_grad=True
        )

    def forward(self, x):
        """
        Evaluate the layer given some inputs.

        :param x: The inputs.
        :return: The tensor result of the layer.
        """
        # Apply the dropout, if specified
        if self.training and self.dropout is not None:
            x = x + torch.log(torch.floor(1.0 - self.dropout + torch.rand_like(x)))

        # Calculate the log likelihood using the "logsumexp" trick
        w = torch.log_softmax(self.weight, dim=2)  # (out_regions, out_nodes, in_nodes)
        x = torch.unsqueeze(x, dim=2)              # (-1, in_partitions, 1, in_nodes) with in_partitions = out_regions
        x = torch.logsumexp(x + w, dim=-1)         # (-1, out_regions, out_nodes)
        return x

    @torch.no_grad()
    def mpe(self, x, idx_group, idx_offset):
        """
        Evaluate the layer given some inputs for maximum at posteriori estimation.

        :param x: The inputs.
        :param idx_group: The group indices.
        :param idx_offset: The offset indices.
        :return: The group and offset indices.
        """
        # Compute the offset indices evaluating the sum nodes as an argmax
        x = x[torch.unsqueeze(torch.arange(x.size(0)), dim=1), idx_group]
        w = torch.log_softmax(self.weight[idx_group, idx_offset], dim=2)
        idx_offset = torch.argmax(x + w, dim=2)
        return idx_group, idx_offset

    @torch.no_grad()
    def sample(self, idx_group, idx_offset):
        """
        Sample from a sum layer.

        :param idx_group: The group indices.
        :param idx_offset: The offset indices.
        :return: The group and offset indices.
        """
        # Compute the indices by sampling from a categorical distribution that is parametrized by sum layer's weights
        w = torch.log_softmax(self.weight[idx_group, idx_offset], dim=2)
        idx_offset = torch.distributions.Categorical(logits=w).sample()
        return idx_group, idx_offset


class RootLayer(torch.nn.Module):
    """Root sum node layer."""
    def __init__(self, in_partitions, in_nodes, out_classes):
        """
        Initialize the root layer.

        :param in_partitions: The number of input partitions.
        :param in_nodes: The number of input nodes per partition.
        :param out_classes: The number of output nodes.
        """
        super(RootLayer, self).__init__()
        self.in_partitions = in_partitions
        self.in_nodes = in_nodes
        self.out_classes = out_classes

        # Instantiate the weights
        self.weight = torch.nn.Parameter(
            torch.normal(0.0, 1e-1, [self.out_classes, self.in_partitions * self.in_nodes]),
            requires_grad=True
        )

    def forward(self, x):
        """
        Evaluate the layer given some inputs.

        :param x: The inputs.
        :return: The tensor result of the layer.
        """
        # Calculate the log likelihood using the "logsumexp" trick
        x = torch.flatten(x, start_dim=1)          # (-1, in_partitions * in_nodes)
        w = torch.log_softmax(self.weight, dim=1)  # (out_classes, in_partitions * in_nodes)
        x = torch.unsqueeze(x, dim=1)              # (-1, 1, in_partitions * in_nodes)
        x = torch.logsumexp(x + w, dim=-1)         # (-1, out_classes)
        return x

    @torch.no_grad()
    def mpe(self, x, y):
        """
        Evaluate the layer given some inputs for maximum at posteriori estimation.

        :param x: The inputs.
        :param y: The target classes.
        :return: The group and offset indices.
        """
        # Compute the layer top-down and get the group and offset indices
        x = torch.flatten(x, start_dim=1)
        w = torch.log_softmax(self.weight, dim=1)
        idx = torch.argmax(x + w[y], dim=1)
        idx_group = idx // self.in_nodes
        idx_offset = idx % self.in_nodes
        return idx_group, idx_offset

    @torch.no_grad()
    def sample(self, y):
        """
        Sample from the root layer.

        :param y: The target classes.
        :return: The group and offset indices.
        """
        # Compute the layer top-down and get the indices
        w = torch.log_softmax(self.weight, dim=1)
        idx = torch.distributions.Categorical(logits=w[y]).sample().unsqueeze(dim=1)
        idx_group = idx // self.in_nodes
        idx_offset = idx % self.in_nodes
        return idx_group, idx_offset
