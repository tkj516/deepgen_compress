import sys
sys.path.append('..')
sys.path.append('../..')

import os
import time
import argparse
import math
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
from torchvision.datasets import MNIST, CIFAR10

from torch_parallel.code_bp_torch_v2 import CodeBP
from dgcspn import DgcSpn
from spnflow.utils.data import compute_mean_quantiles
from spnflow.torch.transforms import Reshape
from my_experiments.datasets import MarkovDataset
from ldpc_generate import pyldpc_generate

from spn_code_decoding.markov_test.utils import *
from spn_code_decoding.markov_test.markov_source import *

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

def convert_bin_to_graycode(x, bitplane=8):

    x = x.astype('uint8')
    graycode = np.bitwise_xor(np.right_shift(x, 1), x)

    return np.bitwise_and(np.right_shift(graycode, bitplane-1), 1)

##################################################################################
# SOURCE-CODE BELIEF PROPAGATION USING SUM-PRODUCT NETWORK (SPN)
##################################################################################

class Source():

    def __init__(self, 
                in_size=(1, 32, 32),
                out_classes=1,
                args=None):

        # Specify the device
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        # Specify the leaf distribution
        assert args.binary is True
        leaf_distribution = "indicator"

        # Compute mean quantiles, if specified
        assert args.quantiles_loc == False
        quantiles_loc = None

        # Store some parameters
        self.h = in_size[1]
        self.w = in_size[2]
        
        # Load the model
        self.model = DgcSpn(
                        in_size,
                        dequantize=args.dequantize,
                        logit=args.logit,
                        out_classes=out_classes,
                        n_batch=args.n_batches,
                        sum_channels=args.sum_channels,
                        depthwise=args.depthwise,
                        n_pooling=args.n_pooling,
                        optimize_scale=args.optimize_scale,
                        in_dropout=args.in_dropout,
                        sum_dropout=args.sum_dropout,
                        quantiles_loc=quantiles_loc,
                        uniform_loc=args.uniform_loc,
                        rand_state=np.random.RandomState(42),
                        leaf_distribution=leaf_distribution
                    ).to(self.device)

        # TODO:  Please look into this 12/30
        if args.data_parallel:
            self.model = MyDataParallel(self.model, device_ids=[0, 1])
            self.model.to(self.device)

        # Restore the checkpoint and set to evaluation mode
        model_checkpoint = torch.load(args.checkpoint)
        self.model.load_state_dict(model_checkpoint['model_state_dict'])
        self.model.eval()

    def message_fast(self, x, dope_mask):

        # Expect non log beliefs and convert them to log beliefs
        external_log_probs = torch.log(x) - torch.logsumexp(torch.log(x), dim=1, keepdim=True)

        input = float('nan') * torch.ones(1, 1, self.h, self.w).to(self.device)

        # Compute the base distribution log-likelihoods
        z = self.model.base_layer(input)
        z.requires_grad = True

        # If there are external probabilities per node add them here
        z = z + external_log_probs.reshape(1, -1, self.h, self.w)
        z = z - torch.logsumexp(z, dim=1, keepdim=True)

        # Forward through the inner layers
        y = z
        for layer in self.model.layers:
            y = layer(y)

        # Forward through the root layer
        y = self.model.root_layer(y)

        # Compute the gradients at distribution leaves
        (z_grad,) = torch.autograd.grad(y, z, grad_outputs=torch.ones_like(y))  # 1 x alphabet_size x (h*w)

        # Reshape to get message
        message = z_grad.flatten(start_dim=2)

        # If you calculate the derivative the result must bust divided by the external log prob
        # Remember that the derivative at an indicator enforces that the pixel is either 0 or 1
        # But it was actually scaled by the external prob, so remove it to resemble slow message passing
        message = torch.log(message) - external_log_probs
        message = message.permute(0, 2, 1)  # 1 x (h*w) x alphabet_size
        # Replace doped probabilities with correct label
        message = torch.where(dope_mask == 1, external_log_probs.permute(0, 2, 1), message)
        # Remove nans after division
        message = torch.where(torch.isnan(message), external_log_probs.permute(0, 2, 1), message)
        message = torch.exp(message)

        return message

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
        self.alphabet_size = args.n_batches

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
                        lambda x: torch.tensor(convert_bin_to_graycode(x.numpy(), args.bitplane)),
                        lambda x: x.float(),
                    ])

        if args.dataset == 'mnist':
            self.dataset = MNIST('../../../MNIST', train=False, transform=self.transform)
        elif args.dataset == 'cifar10':
            self.dataset = CIFAR10('../../../CIFAR10', train=False, transform=self.transform)
        else:
            NotImplementedError("No other datasets supported currently")

        # Setup the source graph
        self.source = Source(in_size=(1, self.h, self.w), out_classes=1, args=args)

        # Setup the code graph
        self.code = CodeBP(self.H, self.device).to(self.device)

        # Store a matrix for doping probabilities
        self.npot = torch.ones(1, self.h * self.w, self.alphabet_size).to(self.device) / self.alphabet_size
        # Store a matrix for masking
        self.mask = torch.zeros(1, self.h * self.w, self.alphabet_size).to(self.device)
        # Input image
        self.samp = None

        # Encoded image
        self.x = None

        # Initialize the messages
        self.M_to_code = None
        self.M_to_grid = None
        self.B = None

    def doping(self):

        indices = np.random.choice(self.h*self.w, size=int(self.h*self.w*self.doperate), replace=False)
        vals = self.samp.cpu().numpy()[indices].flatten().astype(int)
        self.npot[:, indices, :] = 0.0
        self.npot[:, indices, vals] = 1.0
        self.mask[:, indices, :] = 1.0
        # Update the node potential after doping
        self.ps = msg_int_to_graycode(self.npot)

    def generate_sample(self):

        idx = np.random.randint(0, len(self.dataset))
        # idx = 3756
        self.samp, _ = self.dataset[idx]
        self.samp = torch.FloatTensor(self.samp.reshape(-1, 1)).to(self.device)
        # Works well with Numpy so just convert it to be safe
        self.graycoded_samp = torch.FloatTensor(convert_to_graycode(self.samp.cpu().numpy().astype('uint8'), bits=self.bits)).to(self.device)

    def set_sample(self, x):
        
        self.samp = x
        self.samp = torch.FloatTensor(self.samp.reshape(-1, 1)).to(self.device)
        # Works well with Numpy so just convert it to be safe
        self.graycoded_samp = torch.FloatTensor(convert_to_graycode(self.samp.cpu().numpy().astype('uint8'), bits=self.bits)).to(self.device)

    def encode(self):

        self.x = (self.H @ self.graycoded_samp) % 2

    def decode_step(self, fast_message=True):

        # Perform one step of code graph belief propagation
        self.code(self.ps, self.x, self.M_to_code)
        self.M_from_code = self.code.M_out
        # Convert to graycode and reshape to send to grid
        self.M_to_grid = msg_graycode_to_int(
            M_in=self.M_from_code,
            height=1,
            width=self.h * self.w,
            bits=self.bits,
        )  # 1 x (h*w) x alphabet_size

        # Perform one step of source graph belief propagation
        # TODO: Make sure that there is no conflict in the probabilities.  For example if npot
        # says [1, 0] and code graph says [0, 1] we need to make sure it is [1, 0] before it is passed
        # to the source graph, otherwise just multiplying will result in [0, 0] probability.  To do this 
        # we just change every doped pixel row back to [0, 1] or [1, 0] as specified in npot
        external_prob = self.M_to_grid * self.npot  # 1 x (h*w) x 256
        external_prob = torch.where(self.mask == 1, self.npot, external_prob)  # 1 x (h*w)x 256
        external_prob = external_prob.permute(0, 2, 1) # 1 x 256 x (h*w)
        self.M_from_grid = self.source.message_fast(external_prob, self.mask)
        # Convert to int and reshape this output
        self.M_to_code = msg_int_to_graycode(self.M_from_grid)

    def decode(self, num_iter=1, verbose=False, writer=None):

        # Set the initial beliefs to all nans
        max_ll_old = torch.tensor(float('nan') * np.ones((1, self.h * self.w))).to(self.device)

        self.video = []

        # Perform multiple iterations of belief propagation
        for i in range(num_iter):

            # Perform a step of message passing/decoding
            self.decode_step()

            # Calculate the belief
            self.B = self.M_from_grid * self.M_to_grid * self.npot
            self.B /= torch.sum(self.B, -1, keepdims=True)
            self.max_ll = torch.argmax(self.B, -1).reshape(-1, 1)

            # Add the video for logging
            plot_max_ll = (self.max_ll - torch.min(self.samp)) / torch.max((self.samp - torch.min(self.samp)))
            self.video.append(plot_max_ll.reshape(1, 1, 1, self.h, self.w))

            # Compute the number of errors and print some information
            errs = torch.mean(torch.abs(self.max_ll - self.samp)).item()
            devs = torch.sum(1 - (self.max_ll == self.samp).float()).item()

            if verbose:
                print(f"Iteration {i} :- Errors = {errs}, Deviations = {devs}")

            # Termination condition to end belief propagation
            if torch.sum(torch.abs(self.max_ll - max_ll_old)).item() < 0.5:
                break
            max_ll_old = self.max_ll

        return errs, int(devs), torch.cat(self.video, dim=1)

def test_source_code_bp_spn():

    parser = argparse.ArgumentParser(description='Belief propagation training arguments')
    parser.add_argument('--ldpc_mat', type=str, default='../H_28.mat', help="Path to LDPC matrix")
    parser.add_argument('--device', type=str, default='cuda:0', help="Device to run the code on")
    parser.add_argument('--num_iter', type=int, default=100, help="Number of bp iterations")
    parser.add_argument('--doperate', type=float, default=0.04, help="Dope rate")
    parser.add_argument('--rate', type=float, default=0.5, help='Compression rate')
    parser.add_argument('--console_display', action='store_true', default=False, help="Visualize results in matplotlib")
    parser.add_argument('--num_experiments', type=int, default=1, help="Number of bp experiments")
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file')
    parser.add_argument('--num_avg', type=int, default=1000, help='Number of examples to use for averaging')
    parser.add_argument('--phase', type=str, default='test', help='Phase option for Ising dataset')
    parser.add_argument('--source_type', choices=['pgm', 'spn'], default='spn', help='The type of source model to use')
    # DGC-SPN arguments
    parser.add_argument('--dequantize', action='store_true', help='Whether to use dequantization.')
    parser.add_argument('--logit', type=float, default=None, help='The logit value to use for vision datasets.')
    parser.add_argument('--discriminative', action='store_true', help='Whether to use discriminative settings.')
    parser.add_argument('--n-batches', type=int, default=256, help='The number of input distribution layer batches.')
    parser.add_argument('--sum-channels', type=int, default=32, help='The number of channels at sum layers.')
    parser.add_argument('--depthwise', action='store_true', help='Whether to use depthwise convolution layers.')
    parser.add_argument('--n-pooling', type=int, default=0, help='The number of initial pooling product layers.')
    parser.add_argument(
        '--no-optimize-scale', dest='optimize_scale',
        action='store_false', help='Whether to optimize scale in Gaussian layers.'
    )
    parser.add_argument(
        '--quantiles-loc', action='store_true', default=False,
        help='Whether to use mean quantiles for leaves initialization.'
    )
    parser.add_argument(
        '--uniform-loc', nargs=2, type=float, default=None,
        help='Use uniform location for leaves initialization.'
    )
    parser.add_argument('--in-dropout', type=float, default=None, help='The input distributions layer dropout to use.')
    parser.add_argument('--sum-dropout', type=float, default=None, help='The sum layer dropout to use.')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='The learning rate.')
    parser.add_argument('--batch-size', type=int, default=128, help='The batch size.')
    parser.add_argument('--epochs', type=int, default=100, help='The number of epochs.')
    parser.add_argument('--patience', type=int, default=30, help='The epochs patience used for early stopping.')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='L2 regularization factor.')
    parser.add_argument('--binary', action='store_true', default=False, help='Use binary model and binarize dataset')
    parser.add_argument('--continue_checkpoint', default=None, help='Checkpoint to continue training from')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset to use for training')
    parser.add_argument('--root_dir', type=str, default='/fs/data/tejasj/Masters_Thesis/deepgen_compress/bp/spn_code_decoding/markov_test/markov_hf_001',
                    help='Dataset root directory')
    parser.add_argument('--gpu_id', type=int, default=0, help="GPU device to use")
    parser.add_argument("--log_video", action="store_true", help="Whether to log results in a video")
    parser.add_argument('--data_parallel', action='store_true', default=False, help="Whether to use DataParallel while training.")
    parser.add_argument('--bitplane', type=int, default=8, help="Bitplane to compress")
    args = parser.parse_args()

    if args.dataset == 'mnist':
        h = w = 28
    elif args.dataset == 'cifar10':
        h = w = 32
    else:
        raise NotImplementedError("Decoding not implemented for this dataset")

    rate = args.rate
    M = args.n_batches
    bits = math.ceil(np.log2(M))
    N_bits = h * w * bits

    # Set some default values
    args.depthwise = True
    args.binary = True

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
        writer = SummaryWriter(f'markov_source_bp_results/spn/tensorboard/{timestamp}')
        # Write the args to tensorboard
        writer.add_text('config', str(args.__dict__))

    # Decode the sample
    _, _, video = source_code_bp.decode(num_iter=100, verbose=True, writer=writer)

    if args.log_video:
        writer.add_video(f'convergence_video', video, fps=1, global_step=0)
        plot_samp = source_code_bp.samp.reshape(1, h, w)
        plot_samp = (plot_samp - torch.min(plot_samp)) / torch.max((plot_samp - torch.min(plot_samp)))
        writer.add_image('original_sample', plot_samp, global_step=0)

if __name__ == "__main__":

    test_source_code_bp_spn()