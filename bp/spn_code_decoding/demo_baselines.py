'''
Compute the average rate across different datasets.
'''

import sys
sys.path.append('..')

import os
import json
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib as mpl
from scipy.io import loadmat
from datetime import datetime

# Import compressors
import gzip
import bz2

import torch
import torchvision
from torchvision.datasets import FashionMNIST, MNIST, CIFAR10

from spnflow.torch.transforms import Reshape
from my_experiments.datasets import IsingDataset

mpl.rc('image', cmap='gray')

parser = argparse.ArgumentParser(description='Belief propagation training arguments')
parser.add_argument('--dataset', type=str, default='mnist', help='Dataset to use for training')
parser.add_argument('--root_dir', type=str, default='/fs/data/tejasj/Masters_Thesis/deepgen_compress/bp/datasets/ising_28_05_09_75000',
                    help='Dataset root directory')
parser.add_argument('--num_avg', type=int, default=1000, help='Number of examples to use for averaging')
parser.add_argument('--phase', type=str, default='test', help='Phase option for Ising dataset')
parser.add_argument('--compressor', type=str, choices=['gzip', 'bz2'], required=True, help='Choice of baseline compression system')
parser.add_argument('--binary', action='store_true', default=False, help='Whether to binarize dataset')
args = parser.parse_args()

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")

class Demo():

    def __init__(self):

        # Load the dataset
        image_size = None
        if args.dataset == 'mnist':
            image_size = (1, 28, 28)
            if args.binary:
                transform = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    Reshape(image_size),
                    lambda x: (x > 0.5).float() if args.binary else x
                ])
            else:
                transform = torchvision.transforms.Compose([
                    torchvision.transforms.Grayscale(),
                    lambda x: torch.tensor(np.array(x)),
                    Reshape(image_size),
                    lambda x: x.float(),
                ])
            # Setup the MNIST dataset
            self.dataset = MNIST('../../../MNIST', train=args.phase == 'train', transform=transform)
        elif args.dataset == 'ising':
            image_size = (1, 28, 28)
            self.dataset = IsingDataset(root_dir=args.root_dir, phase=args.phase)
        elif args.dataset == 'fashion-mnist':
            image_size = (1, 28, 28)
            if args.binary:
                transform = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    Reshape(image_size),
                    lambda x: (x > 0.5).float() if args.binary else x
                ])
            else:
                transform = torchvision.transforms.Compose([
                    torchvision.transforms.Grayscale(),
                    lambda x: torch.tensor(np.array(x)),
                    Reshape(image_size),
                    lambda x: x.float(),
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
            ])
            self.dataset = CIFAR10('../../../CIFAR10', train=False, transform=transform)
        else:
            NotImplementedError(f'Model is not yet supported for {args.dataset}')

        self.h, self.w = image_size[1], image_size[2]

        # Choose the compressor
        if args.compressor == 'gzip':
            self.compressor = gzip
        elif args.compressor == 'bz2':
            self.compressor = bz2
        else:
            raise NotImplementedError("Basline compressor not implemented.")

    def compute_dataset_rate(self):
        '''
        Given a dataset compute the average, minimum and maximum rate
        required for decoding.
        '''

        # Create the dataloader
        dataloader = torch.utils.data.DataLoader(
                    self.dataset, batch_size=1, shuffle=False, num_workers=4
                    )

        # Loop through all the images in the dataset log its rate
        rates = []
        min_rate = 2
        max_rate = 0

        if args.num_avg == -1:
            args.num_avg = len(self.dataset)

        # Get the size of empty file
        empty_size = len(self.compressor.compress(bytes()))

        with tqdm(total=args.num_avg) as pbar:
            for sample, _ in dataloader:
                
                # Flatten input and convert to bytes object
                if args.binary:
                    x = ''.join(str(i) for i in list(sample.numpy().reshape(-1, ).astype('uint8')))
                    x = [int(x[i: i+8], 2) for i in range(0, len(x), 8)]
                    x = bytes(x)
                else:
                    x = bytes(list(sample.numpy().reshape(-1, ).astype('uint8')))
                rate = (len(self.compressor.compress(x)) - empty_size) / len(x)

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

        print(f'Avg Rate: {avg_rate}, Min Rate: {min_rate}, Max Rate: {max_rate}')

        # Create binary suffix
        suffix = '_bin' if args.binary else ''
        filepath = os.path.join('demo_gray_baseline', args.dataset, args.compressor + '_' + args.phase, os.path.basename(args.root_dir) + suffix + '.json')
        os.makedirs(os.path.join('demo_gray_baseline', args.dataset, args.compressor + '_' + args.phase), exist_ok=True)
        with open(filepath, 'w') as file:
            json.dump(results, file, indent=4)

def main():

    demo = Demo()
    demo.compute_dataset_rate()

if __name__ == '__main__':
    main()
