'''
Compute the average rate across different datasets.
'''

import sys
sys.path.append('..')

import os
import time
import json
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
from demo_bp import SourceCodeBP, SourceCodeBPPGM

from tensorboardX import SummaryWriter

mpl.rc('image', cmap='gray')

parser = argparse.ArgumentParser(description='Belief propagation training arguments')
parser.add_argument('--ldpc_mat', type=str, default='../H_28.mat', help="Path to LDPC matrix")
parser.add_argument('--device', type=str, default='cuda:0', help="Device to run the code on")
parser.add_argument('--num_iter', type=int, default=100, help="Number of bp iterations")
parser.add_argument('--stay', type=float, default=None, help='Edge strength for Ising model')
parser.add_argument('--doperate', type=float, default=0.04, help="Dope rate")
parser.add_argument('--rate', type=float, default=0.5, help='Compression rate')
parser.add_argument('--console_display', action='store_true', default=False, help="Visualize results in matplotlib")
parser.add_argument('--num_experiments', type=int, default=1, help="Number of bp experiments")
parser.add_argument('--binary', action='store_true', help='Use binary model and binarize dataset')
parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file')
parser.add_argument('--dataset', type=str, default='mnist', help='Dataset to use for training')
parser.add_argument('--root_dir', type=str, default='/fs/data/tejasj/Masters_Thesis/deepgen_compress/bp/datasets/ising_28_05_09_75000',
                    help='Dataset root directory')
parser.add_argument('--num_avg', type=int, default=1000, help='Number of examples to use for averaging')
parser.add_argument('--phase', type=str, default='test', help='Phase option for Ising dataset')
parser.add_argument('--source_type', choices=['pgm', 'spn'], required=True, help='The type of source model to use')
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
            # Setup the MNIST dataset
            image_size = (1, 28, 28)
            transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                         Reshape(image_size),
                                                         lambda x: (x > 0.5).float() if args.binary else x
                                                        ])
            self.dataset = MNIST('../../../MNIST', train=args.phase == 'train', transform=transform)
        elif args.dataset == 'ising':
            image_size = (1, 28, 28)
            self.dataset = IsingDataset(root_dir=args.root_dir, phase=args.phase)
        else:
            NotImplementedError(f'Model is not yet supported for {args.dataset}')

        self.h, self.w = image_size[1], image_size[2]
        self.doperate = args.doperate
        self.num_iter = args.num_iter
        self.source_type = args.source_type

        # Only needed if PGM source
        self.stay = args.stay
        if self.source_type == 'pgm':
            assert(self.stay is not None)

    # Generate the LDPC matrices for various rates and cache them
    # An LDPC matrix with rate 0.02 is represented by H[2]
    def generate_LDPC_matrices(self):
        print('[Generating and caching LDPC matrices ...]')
        self.H = {}
        for i in tqdm(range(1, 151)):
            rate = i/100.0
            self.H[i] = pyldpc_generate.generate(int(rate*self.h*self.w), self.h*self.w, 3.0, 2, 123)

    def compute_sample_rate(self, x):

        def successful_decoding(rate):

            # Intialize the source-code decoding graph
            if self.source_type == 'spn':
                source_code_bp = SourceCodeBP(self.H[rate], h=self.h, w=self.w, doperate=self.doperate, args=args)
            else:
                source_code_bp = SourceCodeBPPGM(self.H[rate], h=self.h, w=self.w, stay=self.stay, doperate=self.doperate, args=args)

            # Either load a sample image or generate one using Gibb's sampling
            source_code_bp.set_sample(x)
            
            # Encode the sample using the LDPC matrix
            source_code_bp.encode()

            # Dope it to update our initial beliefs
            source_code_bp.doping()

            # Decode the code using belief propagation
            errs = source_code_bp.decode(num_iter=self.num_iter)

            # Successfully decoded if the number of errors is 0
            if errs == 0:
                return True
            return False
            
        # Compute the lowest rate using divide and conquer
        low_rate = 1
        high_rate = 150

        while low_rate < high_rate:

            mid_rate = (low_rate + high_rate) // 2

            # If decoding works at the mid rate
            if successful_decoding(mid_rate):
                high_rate = mid_rate
            else:
                low_rate = mid_rate + 1

        assert(low_rate == high_rate)

        return low_rate

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
        min_rate = 150
        max_rate = 0

        if args.num_avg == -1:
            args.num_avg = len(self.dataset)

        with tqdm(total=args.num_avg) as pbar:
            for sample, _ in dataloader:
                
                x = sample
                rate = self.compute_sample_rate(x)

                # Perform rate logging on the dataset
                rates.append(rate)
                min_rate = min(min_rate, rates[-1])
                max_rate = max(max_rate, rates[-1])

                pbar.update(1)

                if len(rates) == args.num_avg:
                    break

        # Log into files here
        results = {}
        avg_rate = np.average(rates)
        results['rates'] = rates
        results['avg_rate'] = avg_rate
        results['min_rate'] = min_rate
        results['max_rate'] = max_rate

        with open('temp.json', 'w') as file:
            json.dump(results, file, indent=4)

        print(f'Avg Rate: {avg_rate/100.0}, Min Rate: {min_rate/100.0}, Max Rate: {max_rate/100.0}')

        filepath = os.path.join('demo', args.dataset, args.source_type + '_' + args.phase, os.path.basename(args.root_dir) + '.json')
        os.makedirs(os.path.join('demo', args.dataset, args.source_type + '_' + args.phase), exist_ok=True)
        with open(filepath, 'w') as file:
            json.dump(results, file, indent=4)

        

def main():

    demo = Demo()
    demo.compute_dataset_rate()

if __name__ == '__main__':
    main()
