import sys

from markov_test.utils import msg_graycode_to_int
sys.path.append('..')
sys.path.append('../..')

import os
import time
import argparse
import math
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib as mpl
from scipy.io import loadmat
from datetime import datetime
from functools import partial
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn.functional as F
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST

from torch_parallel.code_bp_torch_v2 import CodeBP
from dgcspn import DgcSpn
from spnflow.utils.data import compute_mean_quantiles
from spnflow.torch.transforms import Reshape
from my_experiments.datasets import MarkovDataset
from ldpc_generate import pyldpc_generate

from spn_code_decoding.markov_test.utils import *
from spn_code_decoding.markov_test.markov_source import *
from lossy_utils.utils import *

from tensorboardX import SummaryWriter

# Define a class for DataParallel that can access attributes
class MyDataParallel(torch.nn.DataParallel):
    """
    Allow nn.DataParallel to call model's attributes.
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

##################################################################################
# SOURCE-CODE BELIEF PROPAGATION USING SUM-PRODUCT NETWORK (SPN)
##################################################################################

class Source():

    def __init__(self, 
                in_size=(1, 32, 32),
                args=None):

        # Specify the device
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        # Store some parameters
        self.h = in_size[1]
        self.w = in_size[2]
        
        # Load the model
        self.model = DgcSpn(
                        in_size,
                        dequantize=False,
                        logit=None,
                        out_classes=1,
                        n_batch=args.n_batches,
                        sum_channels=args.sum_channels,
                        depthwise=args.depthwise,
                        n_pooling=0,
                        optimize_scale=True,
                        in_dropout=None,
                        sum_dropout=None,
                        quantiles_loc=None,
                        uniform_loc=None,
                        rand_state=np.random.RandomState(42),
                        leaf_distribution=args.leaf_distribution,
                    ).to(self.device)

        # TODO:  Please look into this 12/30
        if args.data_parallel:
            self.model = MyDataParallel(self.model, device_ids=[0, 1])
            self.model.to(self.device)

        # Restore the checkpoint and set to evaluation mode
        model_checkpoint = torch.load(args.checkpoint)
        self.model.load_state_dict(model_checkpoint['model_state_dict'])
        self.model.eval()

    def message(self, x, dope_mask):

        # Expect non log beliefs and convert them to log beliefs
        external_log_probs = torch.log(x) - torch.logsumexp(torch.log(x), dim=1, keepdim=True)

        input = float('nan') * torch.ones(1, 1, self.h, self.w).to(self.device)

        # Compute the base distribution log-likelihoods
        z = self.model.base_layer(input)
        z.requires_grad = True

        # If there are external probabilities per node add them here
        z = z + external_log_probs.reshape(1, -1, self.h, self.w)
        z = z - torch.logsumexp(z, dim=1, keepdim=True)

        # TODO: Uncomment for original version
        # Forward through the inner layers
        # y = z
        # for layer in self.model.layers:
        #     y = layer(y)

        # TODO: New version -- Comment for old version
        z = self.model.layers[0](z)
        y = z
        for i in range(1, len(self.model.layers)):
            y = self.model.layers[i](y)

        # Forward through the root layer
        y = self.model.root_layer(y)

        # Compute the gradients at distribution leaves
        (z_grad,) = torch.autograd.grad(y, z, grad_outputs=torch.ones_like(y))  # 1 x alphabet_size x (h*w)

        # TODO: Comment to revert to old version
        # Compute probabilities
        logits = self.model.layers[0].weight.log_softmax(dim=1).unsqueeze(0)
        z_grad = z_grad.unsqueeze(2) * torch.exp(logits) #* torch.exp(external_log_probs.reshape(1, -1, self.h, self.w)) / torch.exp(z.unsqueeze(2))
        z_grad = z_grad.sum(dim=1)

        # Reshape to get message
        message = z_grad.flatten(start_dim=2)

        # If you calculate the derivative the result must bust divided by the external log prob
        # Remember that the derivative at an indicator enforces that the pixel is either 0 or 1
        # But it was actually scaled by the external prob, so remove it to resemble slow message passing
        message = torch.log(message) #- external_log_probs
        message = message.permute(0, 2, 1)  # 1 x (h*w) x alphabet_size
        # Replace doped probabilities with correct label
        message = torch.where(dope_mask == 1, external_log_probs.permute(0, 2, 1), message)
        # Remove nans after division
        message = torch.where(torch.isnan(message), external_log_probs.permute(0, 2, 1), message)
        message = torch.exp(message)

        return message

    def message_fast(self, quant_mean, quant_var, num_bins, width, dope_mask=None, dope_prob=None, sample_doping=False):

        # Get the unfiltered message that will need to be corrected with 
        # doping probabilities in the next step
        message, mixture_probs = source_to_quant(
            quant_mean=quant_mean,
            quant_var=quant_var,
            spn=self.model,
            num_bins=num_bins,
            width=width,
        )  # (1, num_bins, h, w)

        # Flatten this message
        message = message.flatten(start_dim=2).permute(0, 2, 1)  # (1, num_bins, h * w)

        # Filter the messages if using sample doping
        if sample_doping:
            message = torch.where(dope_mask == 1, dope_prob, message)  # (1, h * w, num_bins)

        return message, mixture_probs


class SourceCodeBP():

    def __init__(self,
                 H,
                 h=1,
                 w=1000,
                 p=0.5, 
                 doperate=0.4,
                 args=None):

        # Specify the device
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        # Store the parameters
        self.h = h
        self.w = w
        self.p = p
        self.doperate = doperate
        self.alphabet_size = args.num_bins
        self.width = args.width
        self.doping_type = args.doping_type

        # Number of bits per alphabet
        self.bits = math.ceil(np.log2(self.alphabet_size))

        # Store the parity check matrix
        self.H = torch.FloatTensor(np.array(H.todense()).astype('float32')).to(self.device)
        self.K, self.N = self.H.shape

        # Setup the transforms
        in_size = (1, self.h, self.w)
        self.transform = torchvision.transforms.Compose([
                            torchvision.transforms.Grayscale(),
                            lambda x: torch.tensor(np.array(x)),
                            Reshape(in_size),
                            lambda x: x.float(),
                        ])

        if args.dataset == 'mnist':
            self.dataset = MNIST('../../../MNIST', train=False, transform=self.transform)
        elif args.dataset == 'fashion-mnist':
            self.dataset = FashionMNIST('../../../FashionMNIST', train=False, transform=self.transform)
        elif args.dataset == 'cifar10':
            self.dataset = CIFAR10('../../../CIFAR10', train=False, transform=self.transform)
        else:
            NotImplementedError("No other datasets supported currently")

        # Setup the source graph
        self.source = Source(in_size=(1, self.h, self.w), args=args)

        # Setup the code graph
        self.code = CodeBP(self.H, self.device).to(self.device)

        # Store a matrix for doping probabilities
        self.npot = torch.ones(1, self.h * self.w, self.alphabet_size).to(self.device) / self.alphabet_size
        # Store a matrix for masking
        self.mask = torch.zeros(1, self.h * self.w, self.alphabet_size).to(self.device)
        
        # Input image
        self.s = None
        self.samp = None

        # Encoded image
        self.c = None
        self.codeword = None

        # Initialize the messages
        self.M_to_code = None
        self.M_to_grid = None
        self.B = None

    def sample_doping(self):

        # Choose the doping indices based on the doping rate
        indices = np.random.choice(self.h*self.w, size=int(self.h*self.w*self.doperate), replace=False)

        # Get the value of the doped bits from the quantized samples
        vals = self.q[indices].flatten().astype(int)

        self.npot[:, indices, :] = 0.0
        self.npot[:, indices, vals] = 1.0
        self.mask[:, indices, :] = 1.0

        # Update the node potential for code graph BP after doping
        self.ps = msg_int_to_graycode(self.npot)

    def lattice_doping(self, lattice_sites):

        # Choose the doping indices periodically in the translated domain
        indices = np.arange(0, self.h * self.w * self.bits, self.bits) + lattice_sites.reshape(-1, 1)
        indices = indices.flatten()
        vals = self.g[indices].flatten()

        self.ps = 0.5 * torch.ones(self.h * self.w * self.bits, 2).to(self.device)
        self.mask = torch.zeros(self.h * self.w * self.bits, 2).to(self.device)
        self.ps[indices, :] = 0.0
        self.ps[indices, vals] = 1.0
        self.mask[indices, :] = 1.0

        self.npot = msg_graycode_to_int(self.ps, 1, self.h * self.w, self.bits)

    def doping(self):

        if self.doping_type == 'sample_doping':
            self.sample_doping()
        elif self.doping_type == 'lattice_doping':
            self.lattice_doping(lattice_sites=np.array([1]))
        else:
            raise NotImplementedError('Other forms of doping not implemented!')

    def generate_sample(self):

        idx = np.random.randint(0, len(self.dataset))
        self.s, _ = self.dataset[idx]
        self.s = self.s.reshape(-1, 1).cpu().numpy() / 256
        # self.samp = torch.FloatTensor(self.samp.reshape(-1, 1)).to(self.device)
        # # Works well with Numpy so just convert it to be safe
        # self.graycoded_samp = torch.FloatTensor(convert_to_graycode(self.samp.cpu().numpy().astype('uint8'), bits=self.bits)).to(self.device)

    def set_sample(self, x):
        
        self.s = x.reshape(-1, 1) / 256
        # self.samp = torch.FloatTensor(self.samp.reshape(-1, 1)).to(self.device)
        # # Works well with Numpy so just convert it to be safe
        # self.graycoded_samp = torch.FloatTensor(convert_to_graycode(self.samp.cpu().numpy().astype('uint8'), bits=self.bits)).to(self.device)

    def quantize(self):

        self.q = np.maximum(np.minimum(np.floor(self.s / self.width), 0), self.alphabet_size-1)

    def translate(self):

        self.g = convert_to_graycode(self.q.astype('uint8'), bits=self.bits)

    def hash(self):

        self.c = (self.H.cpu().numpy() @ self.g) % 2

    def encode(self):

        # First quantize the sample
        self.quantize()

        # Translate the sample
        self.translate()

        # Hash the translated sample
        self.hash()

        # Push the different vectors to the device
        self.samp = torch.FloatTensor(self.s).to(self.device)
        self.quantized_samp = torch.FloatTensor(self.q).to(self.device)
        self.graycoded_samp = torch.FloatTensor(self.g).to(self.device)
        self.codeword = torch.FloatTensor(self.c).to(self.device)

    def encode_full(self, x):

        x = np.floor(x / self.width)
        x = convert_to_graycode(x.astype('uint8'), bits=self.bits)
        x = (self.H.cpu().numpy() @ x) % 2

        return x

    def decode_step(self):

        # Perform one step of code graph belief propagation
        self.code(self.ps, self.codeword, self.M_to_code)
        self.M_from_code = self.code.M_out  # (h * w * bits, 2)

        # Convert the message over the graycode to a message
        # over the quantizer bins
        self.M_to_quant = msg_graycode_to_int(
            M_in=self.M_from_code,
            height=1,
            width=self.h * self.w,
            bits=self.bits,
        )  # (1, h * w, alphabet_size)

        # Convert the discrete messages over the quantizer bins
        # to a Gaussian message by computing the mean and variance
        self.quant_mean, self.quant_var = quant_to_source(
            num_bins=self.alphabet_size,
            width=self.width,
            message=(self.M_to_quant * self.npot) / torch.sum(self.M_to_quant * self.npot, dim=-1, keepdim=True),  #TODO: Might not need to multiply here
        )  # (1, h * w)

        # Reshape for input to SPN source -- (1, out_channels, in_channels, h, w)
        self.quant_mean = self.quant_mean.reshape(1, 1, 1, self.h, self.w)
        self.quant_var = self.quant_var.reshape(1, 1, 1, self.h, self.w)

        # Perform one step of source graph belief propagation
        self.M_from_grid, mixture_probs = self.source.message_fast(
            quant_mean=self.quant_mean,
            quant_var=self.quant_var,
            num_bins=self.alphabet_size,
            width=self.width,
            dope_mask=self.mask,
            dope_prob=self.npot,
            sample_doping=(self.doping_type == 'sample_doping'),
        )  # (1, h * w, num_bins), (1, num_components, h, w)

        # Convert to messages over graycode
        self.M_to_code = msg_int_to_graycode(self.M_from_grid)  # (h * w * bits, 2)

        if self.doping_type == 'lattice_doping':
            self.M_to_code = torch.where(self.mask == 1, self.ps, self.M_to_code)

        # Compute the marginal image
        base_mean = self.source.model.base_layer.mean
        base_var = torch.exp(2 * self.source.model.base_layer.log_scale)

        # Get the mixture probabilities (derivatives) and multiply with scaling factor
        scaling_factor = torch.exp(-0.5 * (base_mean - self.quant_mean)**2 / (base_var + self.quant_var)) / torch.sqrt(2 * np.pi * (base_var + self.quant_var))
        mixture_probs = mixture_probs * scaling_factor.squeeze(2)
        mixture_probs /= torch.sum(mixture_probs, dim=1)  # Normalize here

        # Get expected value of the base distribution x messages from code 
        marginal = (base_mean * self.quant_var + self.quant_mean * base_var) / (base_var + self.quant_var)  # (1, num_components, 1, h, w)
        self.marginal = torch.sum(mixture_probs * marginal.squeeze(2), dim=1, keepdim=True)  # (1, 1, h, w)

    def decode(self, num_iter=1, verbose=False, writer=None):

        # Set the initial beliefs to all nans
        max_ll_old = torch.tensor(float('nan') * np.ones((1, 1, self.h, self.w))).to(self.device)

        self.video = []
        self.video = [torch.argmax(self.npot, -1).reshape(1, 1, 1, self.h, self.w)]

        # Perform multiple iterations of belief propagation
        for i in range(num_iter):

            # Perform a step of message passing/decoding
            self.decode_step()

            # Calculate the marginal image
            marginal_image = self.marginal

            # Add the video for logging
            self.video.append(marginal_image.reshape(1, 1, 1, self.h, self.w))  # TODO: Check this for image intensity values

            # Compute the MSE loss
            mse = torch.mean((marginal_image - self.samp.reshape(1, 1, self.h, self.w))**2).item()

            if verbose:
                print(f"Iteration {i} :- MSE = {mse}")

            # Termination condition to end belief propagation
            if torch.max(torch.abs(marginal_image - max_ll_old)).item() < 0.01:
                break
            max_ll_old = marginal_image

        # Check if the marginal image satisfies the constraints
        z = self.codeword.detach().cpu().numpy()
        z_hat = self.encode_full(marginal_image.detach().cpu().numpy().reshape(-1, 1))
        errs = np.count_nonzero(z - z_hat)

        print(f"Errors in constraint satisfaction :- {errs} / {len(z_hat)}")

        return mse, errs, torch.cat(self.video, dim=1)

def test_source_code_bp_spn():

    parser = argparse.ArgumentParser(description='Belief propagation training arguments')
    parser.add_argument('--ldpc_mat', type=str, default='../H_28.mat', help="Path to LDPC matrix")
    parser.add_argument('--device', type=str, default='cuda:0', help="Device to run the code on")
    parser.add_argument('--num_iter', type=int, default=100, help="Number of bp iterations")
    parser.add_argument('--doperate', type=float, default=0.04, help="Dope rate")
    parser.add_argument('--rate', type=float, default=0.5, help='Compression rate')
    parser.add_argument('--num_bins', type=int, default=256, help='Number of bins to use in quantizer')
    parser.add_argument('--inv_width', type=float, default=1, help='Inverse width of each quantizer bin')
    parser.add_argument('--width', type=float, default=1, help='Width of each quantizer bin')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file')
    parser.add_argument('--phase', type=str, default='test', help='Phase option for Ising dataset')
    parser.add_argument('--doping_type', type=str, default='sample_doping', choices=['sample_doping', 'lattice_doping'], help="Type of doping to use")
    # DGC-SPN arguments
    parser.add_argument('--n-batches', type=int, default=128, help='The number of input distribution layer batches.')
    parser.add_argument('--sum-channels', type=int, default=64, help='The number of channels at sum layers.')
    parser.add_argument('--depthwise', action='store_true', help='Whether to use depthwise convolution layers.')
    parser.add_argument('--leaf_distribution', type=str, default='gaussian', help="Type of leaf distribution")
    parser.add_argument('--dataset', type=str, choices=['mnist', 'fashion-mnist', 'cifar10'], default='cifar10', help='Dataset to use for training')
    parser.add_argument('--root_dir', type=str, default='/fs/data/tejasj/Masters_Thesis/deepgen_compress/bp/spn_code_decoding/markov_test/markov_hf_001',
                    help='Dataset root directory')
    parser.add_argument('--gpu_id', type=int, default=0, help="GPU device to use")
    parser.add_argument("--log_video", action="store_true", help="Whether to log results in a video")
    parser.add_argument('--data_parallel', action='store_true', default=False, help="Whether to use DataParallel while training.")
    args = parser.parse_args()

    if args.dataset in  ['mnist', 'fashion-mnist']:
        h = w = 28
    elif args.dataset == 'cifar10':
        h = w = 32
    else:
        raise NotImplementedError("Decoding not implemented for this dataset")

    # Store the decoding rate
    rate = args.rate
    # Store the alphabet size/number of quantizer bins
    M = args.num_bins
    # Store the number of bits per bin and total bits of the quantized representation
    bits = math.ceil(np.log2(M))
    N_bits = h * w * bits

    # Set the inverse width from width
    args.width = 1 / args.inv_width

    source_code_bp = SourceCodeBP(
                        H=pyldpc_generate.generate(int(rate * N_bits), N_bits, 3.0, 2, 123),
                        h=h,
                        w=w,
                        doperate=args.doperate,
                        args=args,
                    )

    # Generate a sample
    source_code_bp.generate_sample()

    # Encode the sample
    source_code_bp.encode()

    # Perform doping
    source_code_bp.doping()

    writer = None
    if args.log_video:
        # Create the writer
        timestamp = time.strftime("%Y-%m-%d_%H:%M:%S")
        writer = SummaryWriter(f'bp_results_lossy/spn/tensorboard/{timestamp}')
        # Write the args to tensorboard
        writer.add_text('config', str(args.__dict__))

    # Decode the sample
    _, _, video = source_code_bp.decode(num_iter=100, verbose=True, writer=writer)

    if args.log_video:
        os.makedirs(f'bp_results_images_gray_lossy/{args.dataset}/{timestamp}/spn')
        numpy_image = video.squeeze().detach().cpu().numpy().repeat(4, axis=-1).repeat(4, axis=-2)
        for i in range(numpy_image.shape[0]):
            im = Image.fromarray((255*numpy_image[i, ...]).astype('uint8')).convert('RGB')
            im.save(f'bp_results_images_gray_lossy/{args.dataset}/{timestamp}/spn/{i}.png')

    if args.log_video:
        writer.add_video(f'convergence_video', video, fps=1, global_step=0)
        plot_samp = source_code_bp.samp.reshape(1, h, w)
        plot_samp = (plot_samp - torch.min(plot_samp)) / torch.max((plot_samp - torch.min(plot_samp)))
        writer.add_image('original_sample', plot_samp, global_step=0)

if __name__ == "__main__":

    test_source_code_bp_spn()