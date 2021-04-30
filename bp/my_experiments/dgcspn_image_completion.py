import sys
sys.path.append('..')

import os
import time
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from spnflow.torch.models.dgcspn import DgcSpn
from spnflow.utils.data import compute_mean_quantiles

from experiments.datasets import load_vision_dataset
from experiments.datasets import VISION_DATASETS
from experiments.utils import collect_results_generative, collect_results_discriminative

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(
        description='Deep Generalized Convolutional Sum-Product Networks (DGC-SPNs) experiments'
    )
    parser.add_argument(
        'dataset', choices=VISION_DATASETS, help='The vision dataset used in the experiment.'
    )
    parser.add_argument('--dequantize', action='store_true', help='Whether to use dequantization.')
    parser.add_argument('--logit', type=float, default=None, help='The logit value to use for vision datasets.')
    parser.add_argument('--discriminative', action='store_true', help='Whether to use discriminative settings.')
    parser.add_argument('--n-batches', type=int, default=8, help='The number of input distribution layer batches.')
    parser.add_argument('--sum-channels', type=int, default=8, help='The number of channels at sum layers.')
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
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    args = parser.parse_args()

    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    assert args.quantiles_loc is False or args.uniform_loc is None, \
        'Only one between --quantiles-loc and --uniform-loc can be defined'

    # Load the dataset
    if args.discriminative:
        (data_train, label_train), (data_valid, label_valid), (data_test, label_test) = load_vision_dataset(
            'datasets', args.dataset, unsupervised=False
        )
    else:
        data_train, data_valid, data_test = load_vision_dataset('../datasets', args.dataset, unsupervised=True)
    in_size = data_train.shape[1:]

    if args.discriminative:
        out_classes = len(np.unique(label_train))
        data_train = list(zip(data_train, label_train))
        data_valid = list(zip(data_valid, label_valid))
        data_test = list(zip(data_test, label_test))
    else:
        out_classes = 1

    # Compute mean quantiles, if specified
    if args.quantiles_loc:
        quantiles_loc = compute_mean_quantiles(data_train, args.n_batches)
    else:
        quantiles_loc = None

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
        rand_state=rand_state
    ).to(device)

    # Load the model
    model.load_state_dict(torch.load(args.checkpoint))

    # Perform image completion
    train_loader = torch.utils.data.DataLoader(
        data_train, batch_size=1, shuffle=True, num_workers=8
    )
    
    for x in train_loader: 
        orig = x.float()
        x = x.float()  
        # Cover the right side of the image
        b, c, h, w = x.shape
        x[..., w//2:] = float('nan')
        x = x.to(device)

        # Get the estimated completion using MPE
        sample, _ = model.mpe(x)

        fig, ax = plt.subplots(3, 1)
        ax[0].imshow(orig.detach().cpu().squeeze(0).squeeze(0).numpy())
        ax[1].imshow(sample.detach().cpu().squeeze(0).squeeze(0).numpy())
        ax[2].imshow((sample.detach().cpu().squeeze(0).squeeze(0).numpy() > 128).astype('float'))
        plt.show()

        break

