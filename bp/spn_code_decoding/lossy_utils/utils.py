import numpy as np
import torch

def quantize_source(
    input,
    width,
):

    return np.floor(input / width)

def quant_to_source(
    num_bins, 
    width,
    message,
):
    """
    num_bins:  Number of bins to use for the quantizer
    width:  Width of each quantizer bin
    message: Message from the translator of shape (1 x (h*w) x alphabet_size)
    """

    # Store the quantized image bin numbers.  
    # Divide the interval [0, 255] into num_bins subintervals of length 256/num_bins.
    bins = torch.arange(num_bins).reshape(1, 1, -1).to(message.device)

    # Compute the mean and variance of the best fit Gaussian
    mean = width * (torch.sum(bins * message, dim=-1) + 1/2)
    var = width**2 / 12 + width**2 * (torch.sum(bins**2 * message, dim=-1) - torch.sum(bins * message, dim=-1)**2)

    return mean, var

def source_to_quant(
    quant_mean,
    quant_var,
    spn,
    num_bins,
    width,
):

    # Get the height and width of the image
    num_components, _, h, w = spn.base_layer.mean.shape

    # The SPN base distribution leaves are gaussian, so compute the new means and variances
    # of the scaled gaussian by incorporating external messages
    base_std = torch.exp(spn.base_layer.log_scale)
    base_var = torch.exp(2 * spn.base_layer.log_scale)
    base_mean = spn.base_layer.mean

    # The leaves of the SPN are Gaussian distributions.  To marginalize them out with
    # external beliefs we need to compute the expected value of the external beliefs
    # w.r.t. the leaf distributions
    z = -0.5 * torch.log(2 * np.pi * (base_var + quant_var)) - 0.5 * ((base_mean - quant_mean)**2 / (base_var + quant_var))
    z = z.squeeze(2)  # Remove in_channels dimension

    y = z
    for layer in spn.layers:
        y = layer(y)

    y = spn.root_layer(y)

    # Compute gradients to compute the mixture component probabilities
    (z_grad, ) = torch.autograd.grad(y, z, grad_outputs=torch.ones_like(y))  # (1, num_components, h, w)

    # Get the logits by computing the probability of each bin
    bins = torch.arange(1, num_bins).reshape(1, -1, 1, 1, 1, 1).to(base_mean.device)
    inv_width = 1 / width

    def cdf(u):
        return spn.base_layer.cdf(-(inv_width * base_mean - u) / (inv_width * base_std))
    probs = torch.cat([torch.zeros(1, 1, num_components, 1, h, w).to(quant_mean.device), cdf(bins), torch.ones(1, 1, num_components, 1, h, w).to(quant_mean.device)], dim=1)
    # probs = cdf(bins + 1) - cdf(bins)  # (1, num_bins, num_components, 1, h, w)
    probs = probs[:, 1:, ...] - probs[:, :-1, ...]
    probs = probs.squeeze(3)

    # Compute the marginal of each pixel now.  Note that this message is unfiltered
    # and must be corrected using the doping probabilities
    # z_grad = z_grad / z
    z_grad = z_grad.softmax(dim=1).unsqueeze(1)  # (1, 1, num_components, h, w)

    message = z_grad * probs  # (1, num_bins, num_components, h, w)
    message = message.sum(dim=2).softmax(dim=1)  # (1, num_bins, h, w)

    return message, z_grad.squeeze(1)



