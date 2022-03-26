'''
Compute the average rate across different datasets.
'''

import sys
sys.path.append('..')

import os
import time
import json
import math
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
from torchvision.datasets import FashionMNIST, MNIST, CIFAR10

from torch_parallel.code_bp_torch_v2 import CodeBP
from torch_parallel.grid_bp_torch import GridBP
from torch_parallel.grid_gibbs import GibbsSampler
from dgcspn import DgcSpn
from spnflow.torch.transforms import Reshape
from ldpc_generate import pyldpc_generate
from demo_bp_gray_lossy import SourceCodeBP
from gauss_markov.utils import GaussMarkovDataset

from tensorboardX import SummaryWriter

mpl.rc('image', cmap='gray')

parser = argparse.ArgumentParser(description='Belief propagation training arguments')
parser.add_argument('--ldpc_mat', type=str, default='../H_28.mat', help="Path to LDPC matrix")
parser.add_argument('--device', type=str, default='cuda:0', help="Device to run the code on")
parser.add_argument('--num_iter', type=int, default=50, help="Number of bp iterations")
parser.add_argument('--doperate', type=float, default=0.04, help="Dope rate")
parser.add_argument('--rate', type=float, default=0.5, help='Compression rate')
parser.add_argument('--num_bins', type=int, default=256, help='Number of bins to use in quantizer')
parser.add_argument('--inv_width', type=float, default=1, help='Inverse width of each quantizer bin')
parser.add_argument('--width', type=float, default=1, help='Width of each quantizer bin')
parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file')
parser.add_argument('--phase', type=str, default='test', help='Phase option for Ising dataset')
parser.add_argument('--doping_type', type=str, default='sample_doping', choices=['sample_doping', 'lattice_doping'], help="Type of doping to use")
parser.add_argument('--min_bin', type=int, default=0, help="Minimum value of the bins")
parser.add_argument('--num_avg', type=int, default=1000, help='Number of examples to use for averaging')
parser.add_argument('--source_type', choices=['pgm', 'spn'], default='spn', help='The type of source model to use')
# DGC-SPN arguments
parser.add_argument('--n-batches', type=int, default=128, help='The number of input distribution layer batches.')
parser.add_argument('--sum-channels', type=int, default=64, help='The number of channels at sum layers.')
parser.add_argument('--depthwise', action='store_true', help='Whether to use depthwise convolution layers.')
parser.add_argument('--leaf_distribution', type=str, default='gaussian', help="Type of leaf distribution")
parser.add_argument('--dataset', type=str, choices=['mnist', 'fashion-mnist', 'cifar10', 'gauss-markov'], default='gauss-markov', help='Dataset to use for training')
parser.add_argument('--root_dir', type=str, default='/fs/data/tejasj/Masters_Thesis/deepgen_compress/bp/gauss_markov_07_071_0_1',
                help='Dataset root directory')
parser.add_argument('--gpu_id', type=int, default=0, help="GPU device to use")
parser.add_argument("--log_video", action="store_true", help="Whether to log results in a video")
parser.add_argument('--data_parallel', action='store_true', default=False, help="Whether to use DataParallel while training.")
args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.enabled = True

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")

class Demo():

    def __init__(self):

        # Load the dataset
        image_size = None
        if args.dataset == 'mnist':
            image_size = (1, 28, 28)
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Grayscale(),
                lambda x: torch.tensor(np.array(x)),
                Reshape(image_size),
                lambda x: x.float(),
                lambda x: x / 256,
            ])
            # Setup the MNIST dataset
            self.dataset = MNIST('../../../MNIST', train=args.phase == 'train', transform=transform)
        if args.dataset == 'fashion-mnist':
            image_size = (1, 28, 28)
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Grayscale(),
                lambda x: torch.tensor(np.array(x)),
                Reshape(image_size),
                lambda x: x.float(),
                lambda x: x / 256,
            ])
            # Setup the MNIST dataset
            self.dataset = FashionMNIST('../../../FashionMNIST', train=args.phase == 'train', transform=transform)
        elif args.dataset == 'cifar10':
            image_size = (1, 32, 32)
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Grayscale(),
                lambda x: torch.tensor(np.array(x)),
                Reshape(image_size),
                lambda x: x.float(),
                lambda x: x / 256,
            ])
            self.dataset = CIFAR10('../../../CIFAR10', train=False, transform=transform)
        elif args.dataset == 'gauss-markov':
            image_size = (1, 32, 32)
            transform = torchvision.transforms.Compose([
                Reshape(image_size),
                lambda x: x.float(),
            ])
            self.dataset = GaussMarkovDataset(phase='test')
        else:
            NotImplementedError(f'Model is not yet supported for {args.dataset}')

        self.h, self.w = image_size[1], image_size[2]
        self.doperate = args.doperate
        self.num_iter = args.num_iter
        self.source_type = args.source_type

    # Generate the LDPC matrices for various rates and cache them
    # An LDPC matrix with rate 0.02 is represented by H[2]
    def generate_LDPC_matrices(self):
        print('[Generating and caching LDPC matrices ...]')
        self.H = {}
        for num_bins in [2, 4, 8, 16, 32, 64, 128]:
            for i in range(1, 151):
                rate = i/100.0
                N_bits = self.h * self.w * math.ceil(np.log2(num_bins))
                self.H[(num_bins, i)] = pyldpc_generate.generate(int(rate*N_bits), N_bits, 3.0, 2, 123)

    def compute_sample_rate(self, x, num_bins, width):

        def successful_decoding(rate):

            # Intialize the source-code decoding graph
            if self.source_type == 'spn':
                source_code_bp = SourceCodeBP(self.H[(num_bins, rate)], 
                                                h=self.h, 
                                                w=self.w, 
                                                doperate=self.doperate, 
                                                args=args)
            else:
                raise NotImplementedError('This demo only works for SPN')

            # Either load a sample image or generate one using Gibb's sampling
            source_code_bp.set_sample(x)
            
            # Encode the sample using the LDPC matrix
            source_code_bp.encode()

            # Dope it to update our initial beliefs
            source_code_bp.doping()

            # Decode the code using belief propagation
            mse, errs, _ = source_code_bp.decode(num_iter=self.num_iter)

            # Successfully decoded if the number of errors is 0
            if errs == 0:
                return mse, -10 * np.log10(mse), True
            return -1, -1, False
            
        # Compute the lowest rate using divide and conquer
        low_rate = 1
        high_rate = 150
        mse_opt = 0
        sqnr_opt = 0

        # Set the number of bins and the width of each bin
        args.num_bins = num_bins
        args.width = width
        args.inv_width = 1 / width
        args.min_bin = - num_bins // 2

        while low_rate < high_rate:

            mid_rate = (low_rate + high_rate) // 2

            # If decoding works at the mid rate store the mse,
            # sqnr and rate
            mse, sqnr, flag = successful_decoding(mid_rate)
            if flag:
                high_rate = mid_rate
                mse_opt = mse
                sqnr_opt = sqnr
            else:
                low_rate = mid_rate + 1

        assert(low_rate == high_rate)

        return mse_opt, sqnr_opt, low_rate

    def compute_dataset_rate(self):
        '''
        Given a dataset compute the average, minimum and maximum rate
        required for decoding.
        '''

        # Generate the LDPC matrices
        self.generate_LDPC_matrices()

        # Create the dataloader
        dataloader = torch.utils.data.DataLoader(
                    self.dataset, batch_size=1, shuffle=False, num_workers=4
                    )

        # Loop through all the images in the dataset log its rate
        rates = []

        if args.num_avg == -1:
            args.num_avg = len(self.dataset)

        with tqdm(total=args.num_avg) as pbar:
            count = 0
            for sample, _ in dataloader:
                for num_bins in [2, 4, 8, 16, 32, 64, 128]:
                    for width in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:

                        print(num_bins, width)
                
                        x = sample
                        mse, sqnr, rate = self.compute_sample_rate(x, num_bins, width)

                        # Perform rate logging on the dataset
                        rates.append((num_bins, width, mse, sqnr, rate))

                pbar.update(1)

                count += 1
                if count == args.num_avg:
                    print("here")
                    break

        # Log into files here
        results = {}
        results['rates'] = rates

        with open('temp.json', 'w') as file:
            json.dump(results, file, indent=4)

        filepath = os.path.join('demo_gray_lossy', args.dataset, args.source_type + '_' + args.phase, os.path.basename(args.root_dir) + '.json')
        os.makedirs(os.path.join('demo_gray_lossy', args.dataset, args.source_type + '_' + args.phase), exist_ok=True)
        with open(filepath, 'w') as file:
            json.dump(results, file, indent=4)

def main():

    demo = Demo()
    demo.compute_dataset_rate()

if __name__ == '__main__':
    main()
