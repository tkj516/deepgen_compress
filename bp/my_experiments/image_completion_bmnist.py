import sys
sys.path.append('..')

import os
import time
import json
import argparse
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

from dgcspn import DgcSpn
from spnflow.torch.transforms import Reshape
from spnflow.utils.data import compute_mean_quantiles
from routines import torch_train, torch_test

from tensorboardX import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def omit_pixels(x, omittion_type='right'):

    x = x.float()
    b, c, h, w = x.shape

    if omittion_type == 'right':
        x[..., w//2:] = float('nan')
    elif omittion_type == 'left':
        x[..., :w//2] = float('nan')
    elif omittion_type == 'up':
        x[..., :h//2, :] = float('nan')
    elif omittion_type == 'down':
        x[..., h//2:, :] = float('nan') 
    else:
        random_mask = torch.FloatTensor(x.shape).uniform_() > 0.7
        x.masked_fill_(random_mask, float('nan'))

    return x
    

if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(
        description='Deep Generalized Convolutional Sum-Product Networks (DGC-SPNs) experiments'
    )
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
    parser.add_argument('--binary', action='store_true', default=False, help='Use binary model and binarize dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    args = parser.parse_args()

    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    assert args.quantiles_loc is False or args.uniform_loc is None, \
        'Only one between --quantiles-loc and --uniform-loc can be defined'

    # Load the MNIST dataset
    in_size = (1, 28, 28)
    transform = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    Reshape(in_size),
                    lambda x: (x > 0.5).float() if args.binary else x
                ])
    data_train = torchvision.datasets.MNIST('../examples/dataset', train=True, transform=transform, download=True)
    data_test = torchvision.datasets.MNIST('../examples/dataset', train=False, transform=transform, download=True)
    n_val = int(0.1 * len(data_train))
    n_train = len(data_train) - n_val
    data_train, data_val = torch.utils.data.random_split(data_train, [n_train, n_val])

    out_classes = 1

    # Compute mean quantiles, if specified
    if args.quantiles_loc:
        quantiles_loc = compute_mean_quantiles(data_train, args.n_batches)
    else:
        quantiles_loc = None

    # Specify the leaf distribution
    leaf_distribution = 'gaussian'
    if args.binary:
        leaf_distribution = 'indicator'

    # Build the model
    model = DgcSpn(
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
        rand_state=rand_state,
        leaf_distribution=leaf_distribution
    ).to(device)

    # Load the model
    cp = torch.load(args.checkpoint)
    model.load_state_dict(cp['model_state_dict'])

    # Perform image completion
    train_loader = torch.utils.data.DataLoader(
        data_train, batch_size=1, shuffle=True, num_workers=8
    )
    test_loader = torch.utils.data.DataLoader(
        data_test, batch_size=1, shuffle=True, num_workers=8
    )

    # Create the writer
    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S")
    writer = SummaryWriter('../bp_results/dgcspn/tensorboard/image_completion/' + timestamp)

    # Write the args to tensorboard
    writer.add_text('config', str(args.__dict__))
    
    omittion_order = ['right', 'left', 'up', 'down', 'random']
    
    print("Performing test on training images...")
    # Do random completions on 20 training images
    count = 0
    for x, targets in train_loader: 

        # Original Image
        orig = x.clone().detach().float()

        # Perform omittion
        x = omit_pixels(x, omittion_type=omittion_order[0])
        x = x.to(device)

        # Get the estimated completion using MPE
        sample, _ = model.mpe(x)

        image_grid = torchvision.utils.make_grid(torch.cat([orig, x.cpu(), sample.cpu()], dim=0))
        writer.add_image(f'train/{count}_{omittion_order[0]}', image_grid, count)

        # Increment count and perform circular rotation on omittion order
        count += 1
        old = omittion_order.pop(0)
        omittion_order.append(old)

        if count == 20:
            break

    print("Performing tests on test images...")
    # Do random completions on 20 test images
    count = 0
    for x, targets in test_loader: 

        # Original Image
        orig = x.clone().detach().float()

        # Perform omittion
        x = omit_pixels(x, omittion_type=omittion_order[0])
        x = x.to(device)

        # Get the estimated completion using MPE
        sample, _ = model.mpe(x)

        image_grid = torchvision.utils.make_grid(torch.cat([orig, 
                                                 torch.where(torch.isnan(x.cpu(), torch.tensor(0.5), x.cpu())), 
                                                 sample.cpu()], dim=0))
        writer.add_image(f'test/{count}_{omittion_order[0]}', image_grid, count)

        # Increment count and perform circular rotation on omittion order
        count += 1
        old = omittion_order.pop(0)
        omittion_order.append(old)

        if count == 20:
            break
