import sys
sys.path.append('..')

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

from tensorboardX import SummaryWriter

mpl.rc('image', cmap='gray')

parser = argparse.ArgumentParser(description='Belief propagation training arguments')
parser.add_argument('--ldpc_mat', type=str, default='../H_28.mat', help="Path to LDPC matrix")
parser.add_argument('--device', type=str, default='cuda:0', help="Device to run the code on")
parser.add_argument('--num_iter', type=int, default=100, help="Number of bp iterations")
parser.add_argument('--doperate', type=float, default=0.04, help="Dope rate")
parser.add_argument('--console_display', action='store_true', default=False, help="Visualize results in matplotlib")
parser.add_argument('--num_experiments', type=int, default=1, help="Number of bp experiments")
# DGC-SPN arguments
parser.add_argument('--dequantize', action='store_true', help='Whether to use dequantization.')
parser.add_argument('--logit', type=float, default=None, help='The logit value to use for vision datasets.')
parser.add_argument('--discriminative', action='store_true', help='Whether to use discriminative settings.')
parser.add_argument('--n-batches', type=int, default=2, help='The number of input distribution layer batches.')
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
parser.add_argument('--epochs', type=int, default=1000, help='The number of epochs.')
parser.add_argument('--patience', type=int, default=30, help='The epochs patience used for early stopping.')
parser.add_argument('--weight-decay', type=float, default=0.0, help='L2 regularization factor.')
parser.add_argument('--binary', action='store_true', help='Use binary model and binarize dataset')
parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
parser.add_argument('--dataset', type=str, default='mnist', help='Dataset to use for training')
args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.enabled = True

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")

class MyDataParallel(torch.nn.DataParallel):
    """
    Allow nn.DataParallel to call model's attributes.
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

class Source():

    def __init__(self, 
                in_size=(1, 28, 28),
                out_classes=1):

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
                    ).to(device)

        # Restore the checkpoint and set to evaluation mode
        model_checkpoint = torch.load(args.checkpoint)
        self.model.load_state_dict(model_checkpoint['model_state_dict'])
        self.model.eval()

    def message(self, x):

        # Expect non log beliefs and convert them to log beliefs
        external_log_probs = torch.log(x) - torch.logsumexp(torch.log(x), dim=1, keepdim=True)

        input = float('nan') * torch.ones(1, 1, 28, 28).to(device)

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
        # If nan then this was doped pixel, replace nan with 0 prob => -inf log prob
        message = z_grad.squeeze(0).reshape(2, 784).permute(1, 0)
        message = torch.where(torch.isnan(message), torch.tensor(float('-inf')).to(device), message)

        return message

class SourceCodeBP():

    def __init__(self,
                 H,
                 h=28,
                 w=28,
                 p=0.5, 
                 stay=0.9,
                 alpha=0.8,
                 doperate=0.04
                 ):

        # Store the parameters
        self.h = h
        self.w = w
        self.p = p
        self.stay = stay
        self.alpha = alpha
        self.doperate = doperate

        # Store the parity check matrix
        self.H = H
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
            self.dataset = IsingDataset(phase='test')
        else:
            NotImplementedError(f'Model is not yet supported for {args.dataset}')

        print("[Setup the sampler ...]")

        # Setup the source graph
        self.source = Source(in_size=(1, self.h, self.w), out_classes=1)
        print("[Setup the source graph ...]")

        # Setup the code graph
        self.code = CodeBP(self.H, device).to(device)
        print("[Setup the code graph ...]")

        # Store a matrix for doping probabilities
        self.ps = torch.FloatTensor(np.tile(np.array([1-p, p]), (h*w, 1))).to(device)

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
        self.npot = self.ps.reshape(self.h, self.w, 2)

    def generate_sample(self):

        idx = np.random.randint(0, len(self.dataset))
        self.samp, _ = self.dataset[idx]
        self.samp = torch.FloatTensor(self.samp.reshape(-1, 1)).to(device)

        # self.samp = (torch.rand(28, 28) > 0.5).float().reshape(-1, 1).to(device)

    def encode(self):

        self.x = (self.H @ self.samp) % 2

    # @torch.no_grad()
    def decode_step(self):

        # Perform one step of code graph belief propagation
        self.code(self.ps, self.x, self.M_to_code)
        self.M_from_code = self.code.M_out
        # Reshape to send to grid
        self.M_to_grid = self.M_from_code.reshape(self.h, self.w, 2)

        # Calculate the belief for visualization purposes
        if self.M_to_code is None:
            self.B = self.M_to_grid * self.npot
        else:
            self.B = self.M_from_grid * self.M_to_grid * self.npot
        self.B /= torch.sum(self.B, -1).unsqueeze(-1)
        # Add frames to the video
        self.video.append(self.B[..., 1:].permute(2, 0, 1).unsqueeze(0).unsqueeze(0))

        # Perform one step of source graph belief propagation
        # TODO: Make sure that there is no conflict in the probabilities.  For example if npot
        # says [1, 0] and code graph says [0, 1] we need to make sure it is [1, 0] before it is passed
        # to the source graph, otherwise just multiplying will result in [0, 0] probability.  To do this 
        # we just change every doped pixel row back to [0, 1] or [1, 0] as specified in npot
        external_prob = self.M_to_grid*self.npot
        external_prob = torch.where((external_prob.sum(-1, keepdim=True) == 0).repeat(1, 1, external_prob.shape[-1]), self.npot, external_prob) # b, 2, h, w
        external_prob = external_prob.unsqueeze(0).permute(0, 3, 1, 2)
        self.M_to_code = self.source.message(external_prob)
        # Reshape this output
        self.M_from_grid = self.M_to_code.reshape(self.h, self.w, 2)

    # @torch.no_grad()
    def decode(self, num_iter=1):

        # Set the initial beliefs to all nans
        B_old = torch.tensor(float('nan') * np.ones((self.h, self.w))).to(device)
        start = time.time()

        # Let's create a nice video and log it
        self.video = [self.npot[..., 1:].permute(2, 0, 1).unsqueeze(0).unsqueeze(0)]

        # Perform multiple iterations of belief propagation
        for i in range(num_iter):

            # Perform a step of message passing/decoding
            self.decode_step()

            # Calculate the belief
            self.B = self.M_from_grid * self.M_to_grid * self.npot
            self.B /= torch.sum(self.B, -1).unsqueeze(-1)

            # Add frames to the video
            self.video.append(self.B[..., 1:].permute(2, 0, 1).unsqueeze(0).unsqueeze(0))

            # Termination condition to end belief propagation
            if torch.sum(torch.abs(self.B[..., 1] - B_old)).item() < 0.5:
                break
            B_old = self.B[..., 1]

            # Compute the number of errors and print some information
            errs = torch.sum(torch.abs((self.B[..., 1] > 0.5).float() - self.samp.reshape(self.h, self.w))).item()
            print(f'Iteration {i}: {errs} errors')

        end = time.time()
        print(f'Total time taken for decoding is {end - start}s')

        return torch.cat(self.video, dim=1)

def test_source_code_bp(console_display=False, writer=None, experiment_number=0):

    h = 28
    w = 28

    # Load the LDPC matrix
    H = torch.FloatTensor(loadmat(args.ldpc_mat)['Hf']).to(device)

    # Intialize the source-code decoding graph
    source_code_bp = SourceCodeBP(H, h=h, w=w, doperate=args.doperate)

    # Either load a sample image or generate one using Gibb's sampling
    print("[Generating the sample ...]")
    source_code_bp.generate_sample()
    
    # Encode the sample using the LDPC matrix
    print("[Encoding the sample ...]")
    source_code_bp.encode()

    # Dope it to update our initial beliefs
    print("[Doping ...]")
    source_code_bp.doping()

    # Decode the code using belief propagation
    print("[Decoding ...]")
    video = source_code_bp.decode(num_iter=args.num_iter)

    # Log images to tensorboard
    writer.add_image(f'{experiment_number}/original_image', source_code_bp.samp.reshape(1, h, w), global_step=experiment_number)
    writer.add_video(f'{experiment_number}/convergence_video', video, fps=1, global_step=experiment_number)

    if console_display:
        # Visualize the decoded image
        fig, ax = plt.subplots(3, 1)
        ax[0].imshow(source_code_bp.samp.cpu().numpy().reshape(28, 28), vmin=0, vmax=1)
        ax[0].set_title("Source Image")
        ax[1].imshow(source_code_bp.npot.cpu()[..., 1].numpy(), vmin=0, vmax=1)
        ax[1].set_title("Doping samples")
        ax[2].imshow((source_code_bp.B.cpu()[..., 1] > 0.5).float().numpy(), vmin=0, vmax=1)
        ax[2].set_title("Reconstructed Image")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":

    # Create the writer
    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S")
    writer = SummaryWriter(f'bp_results/{args.dataset}/tensorboard/' + timestamp)

    # Write the args to tensorboard
    writer.add_text('config', str(args.__dict__))

    for i in range(args.num_experiments):
        test_source_code_bp(console_display=args.console_display, writer=writer, experiment_number=i)
