import sys
sys.path.append('..')
sys.path.append('../..')

import os
import time
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib as mpl
from scipy.io import loadmat
from datetime import datetime
from functools import partial
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST, CIFAR10

from torch_parallel.code_bp_torch_v2 import CodeBP
from torch_parallel.grid_bp_torch import GridBP
from torch_parallel.grid_gibbs import GibbsSampler
from my_experiments.dgcspn import DgcSpn
from spnflow.torch.transforms import Reshape
from spnflow.utils.data import compute_mean_quantiles
from my_experiments.datasets import IsingDataset
from ldpc_generate import pyldpc_generate

from .utils import generate_sample

from tensorboardX import SummaryWriter

##################################################################################
# SOURCE-CODE BELIEF PROPAGATION USING SUM-PRODUCT NETWORK (SPN)
##################################################################################

class Source():

    def __init__(self, 
                in_size=(1, 28, 28),
                out_classes=1,
                args=None):

        # Specify the device
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        # Specify the leaf distribution
        leaf_distribution = 'gaussian'
        if args.binary:
            leaf_distribution = 'indicator'

        # Compute mean quantiles, if specified
        if args.quantiles_loc:
            quantiles_loc = compute_mean_quantiles(data_train, args.n_batches)
        else:
            quantiles_loc = None
        
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

        # Restore the checkpoint and set to evaluation mode
        model_checkpoint = torch.load(args.checkpoint)
        self.model.load_state_dict(model_checkpoint['model_state_dict'])
        self.model.eval()

        # Store the input tensor for calculating 0 and 1 probabilities
        one_hot_input = torch.tensor(np.arange(784)).reshape(28, 28)
        one_hot_input = F.one_hot(one_hot_input, num_classes=784).permute(2, 0, 1).unsqueeze(1) # 784, 1, 28, 28
        self.zero_input = torch.where(one_hot_input == 1, torch.tensor(0.0), torch.tensor(float('nan'))).to(self.device)
        self.one_input = torch.where(one_hot_input == 1, torch.tensor(1.0), torch.tensor(float('nan'))).to(self.device)

    def message_fast(self, x, dope_mask):

        # Expect non log beliefs and convert them to log beliefs
        external_log_probs = torch.log(x) - torch.logsumexp(torch.log(x), dim=1, keepdim=True)

        input = float('nan') * torch.ones(1, 1, 28, 28).to(self.device)

        # Compute the base distribution log-likelihoods
        z = self.model.base_layer(input)
        z.requires_grad = True

        # If there are external probabilities per node add them here
        z = z + external_log_probs
        z = z - torch.logsumexp(z, dim=1, keepdim=True)

        # Forward through the inner layers
        y = z
        for layer in self.model.layers:
            y = layer(y)

        # Forward through the root layer
        y = self.model.root_layer(y)

        # Compute the gradients at distribution leaves
        (z_grad,) = torch.autograd.grad(y, z, grad_outputs=torch.ones_like(y))

        # Reshape to get message
        message = z_grad.squeeze(0).reshape(2, 784).permute(1, 0)
        # If you calculate the derivative the result must bust divided by the external log prob
        # Remember that the derivative at an indicator enforces that the pixel is either 0 or 1
        # But it was actually scaled by the external prob, so remove it to resemble slow message passing
        message = torch.log(message) - external_log_probs.squeeze(0).permute(1, 2, 0).reshape(-1, 2)
        # Replace doped probabilities with correct label
        message = torch.where(dope_mask == 1, external_log_probs.squeeze(0).permute(1, 2, 0).reshape(-1, 2), message)
        # Remove nans after division
        message = torch.where(torch.isnan(message), external_log_probs.squeeze(0).permute(1, 2, 0).reshape(-1, 2), message)
        message = torch.exp(message)

        return message

class SourceCodeBP():

    def __init__(self,
                 H,
                 h=28,
                 w=28,
                 p=0.5, 
                 doperate=0.04,
                 args=None):

        # Specify the device
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        # Store the parameters
        self.h = h
        self.w = w
        self.p = p
        self.doperate = doperate

        # Store the parity check matrix
        self.H = torch.FloatTensor(np.array(H.todense()).astype('float32')).to(self.device)
        self.K, self.N = self.H.shape

        # Setup the transforms
        in_size = (1, h, w)
        self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                         Reshape(in_size),
                                                         lambda x: (x > 0.5).float() if args.binary else x
                                                        ])

        if args.dataset == 'mnist':
            # Setup the MNIST dataset
            self.dataset = MNIST('../../../MNIST', train=False, transform=self.transform)
        elif args.dataset == 'ising':
            self.dataset = IsingDataset(root_dir=args.root_dir, phase=args.phase)
        else:
            NotImplementedError(f'Model is not yet supported for {args.dataset}')

        # Setup the source graph
        self.source = Source(in_size=(1, self.h, self.w), out_classes=1, args=args)

        # Setup the code graph
        self.code = CodeBP(self.H, self.device).to(self.device)

        # Store a matrix for doping probabilities
        self.ps = torch.FloatTensor(np.tile(np.array([1-p, p]), (h*w, 1))).to(self.device)
        # Store a matrix for masking
        self.mask = torch.zeros(h*w, 2).to(self.device)

        # Input image
        self.samp = None

        # Encoded image
        self.x = None

        # Initialize the messages
        self.M_to_code = None
        self.M_to_grid = None
        self.B = None

    def doping(self):

        indices = np.random.choice(self.N, size=int(self.N*self.doperate), replace=False)
        self.ps[indices, 0], self.ps[indices, 1] = (self.samp[indices, 0] == 0).float(), (self.samp[indices, 0] == 1).float()
        self.mask[indices, 0], self.mask[indices, 1] = 1.0, 1.0
        # Update the node potential after doping
        self.npot = self.ps.reshape(self.h, self.w, 2)

    def generate_sample(self):

        idx = np.random.randint(0, len(self.dataset))
        self.samp, _ = self.dataset[idx]
        self.samp = torch.FloatTensor(self.samp.reshape(-1, 1)).to(self.device)

    def set_sample(self, x):
        
        self.samp = x
        self.samp = torch.FloatTensor(self.samp.reshape(-1, 1)).to(self.device)

    def encode(self):

        self.x = (self.H @ self.samp) % 2

    def decode_step(self, fast_message=True):

        # Perform one step of code graph belief propagation
        self.code(self.ps, self.x, self.M_to_code)
        self.M_from_code = self.code.M_out
        # Reshape to send to grid
        self.M_to_grid = self.M_from_code.reshape(self.h, self.w, 2)

        # Perform one step of source graph belief propagation
        # TODO: Make sure that there is no conflict in the probabilities.  For example if npot
        # says [1, 0] and code graph says [0, 1] we need to make sure it is [1, 0] before it is passed
        # to the source graph, otherwise just multiplying will result in [0, 0] probability.  To do this 
        # we just change every doped pixel row back to [0, 1] or [1, 0] as specified in npot
        external_prob = self.M_to_grid*self.npot
        external_prob = torch.where((external_prob.sum(-1, keepdim=True) == 0).repeat(1, 1, external_prob.shape[-1]), self.npot, external_prob)
        # external_prob = (1 + self.M_to_grid)*self.npot
        external_prob = external_prob.unsqueeze(0).permute(0, 3, 1, 2) # b, 2, h, w
        self.M_to_code = self.source.message_fast(external_prob, self.mask)
        # Reshape this output
        self.M_from_grid = self.M_to_code.reshape(self.h, self.w, 2)

    def decode(self, num_iter=1):

        # Set the initial beliefs to all nans
        B_old = torch.tensor(float('nan') * np.ones((self.h, self.w))).to(self.device)
        start = time.time()

        # Perform multiple iterations of belief propagation
        for i in range(num_iter):

            # Perform a step of message passing/decoding
            self.decode_step(fast_message=True)

            # Calculate the belief
            self.B = self.M_from_grid * self.M_to_grid * self.npot
            self.B /= torch.sum(self.B, -1).unsqueeze(-1)

            # Termination condition to end belief propagation
            if torch.sum(torch.abs(self.B[..., 1] - B_old)).item() < 0.5:
                break
            B_old = self.B[..., 1]

            # Compute the number of errors and print some information
            errs = torch.sum(torch.abs((self.B[..., 1] > 0.5).float() - self.samp.reshape(self.h, self.w))).item()

        return int(errs)

##################################################################################
# SOURCE-CODE BELIEF PROPAGATION USING PROBABILISTIC GRAPHICAL MODEL (PGM)
##################################################################################

class SourceCodeBPPGM():

    def __init__(self,
                 H,
                 h=28,
                 w=28,
                 p=0.5, 
                 stay=0.9,
                 alpha=0.8,
                 doperate=0.04,
                 args=None):

        # Specify the device
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        # Store the parameters
        self.h = h
        self.w = w
        self.p = p
        self.stay = stay
        self.alpha = alpha
        self.doperate = doperate

        # Store the parity check matrix
        self.H = torch.FloatTensor(np.array(H.todense()).astype('float32')).to(self.device)
        self.K, self.N = self.H.shape

        # Setup the Gibbs sampler
        self.sampler = GibbsSampler(self.h, self.w, self.p, self.stay)

        # Setup the source graph
        self.source = GridBP(self.h, self.w, self.p, self.stay, self.alpha, self.device).to(self.device)

        # Setup the code graph
        self.code = CodeBP(self.H, self.device).to(self.device)

        # Store a matrix for doping probabilities
        self.ps = torch.FloatTensor(np.tile(np.array([1-p, p]), (h*w, 1))).to(self.device)

        # Input image
        self.samp = None

        # Encoded image
        self.x = None

        # Initialize the messages
        self.M_to_code = None
        self.M_to_grid = None
        self.B = None

    def doping(self):

        indices = np.random.choice(self.N, size=int(self.N*self.doperate), replace=False)
        self.ps[indices, 0], self.ps[indices, 1] = (self.samp[indices, 0] == 0).float(), (self.samp[indices, 0] == 1).float()
        # Update the node potential after doping
        self.source.npot.data = self.ps.reshape(self.h, self.w, 2)

    def generate_sample(self):

        self.samp = generate
        self.samp = torch.FloatTensor(self.sampler.samp.reshape(-1, 1)).to(self.device)

    def set_sample(self, x):
        
        self.samp = x
        self.samp = torch.FloatTensor(self.samp.reshape(-1, 1)).to(self.device)

    def encode(self):

        self.x = (self.H @ self.samp) % 2

    def decode_step(self):

        # Perform one step of code graph belief propagation
        self.code(self.ps, self.x, self.M_to_code)
        self.M_from_code = self.code.M_out
        # Reshape to send to grid
        self.M_to_grid = self.M_from_code.reshape(self.h, self.w, 2)

        # Perform one step of source graph belief propagation
        self.source(self.M_to_grid)
        self.M_from_grid = self.source.Mout
        # Reshape to send to code
        self.M_to_code = self.M_from_grid.reshape(self.N, 2)

    def decode(self, num_iter=1):

        # Set the initial beliefs to all nans
        B_old = torch.tensor(float('nan') * np.ones((self.h, self.w))).to(self.device)

        # Perform multiple iterations of belief propagation
        for i in range(num_iter):

            # Perform a step of message passing/decoding
            self.decode_step()

            # Calculate the belief
            self.B = self.M_from_grid * self.M_to_grid * self.source.npot
            self.B /= torch.sum(self.B, -1).unsqueeze(-1)

            # Termination condition to end belief propagation
            if torch.sum(torch.abs(self.B[..., 1] - B_old)).item() < 0.5:
                break
            B_old = self.B[..., 1]

            # Compute the number of errors and print some information
            errs = torch.sum(torch.abs((self.B[..., 1] > 0.5).float() - self.samp.reshape(self.h, self.w))).item()

        return int(errs)