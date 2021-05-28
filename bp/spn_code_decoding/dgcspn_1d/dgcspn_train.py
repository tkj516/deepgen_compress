import sys
sys.path.append('..')
sys.path.append('../..')

import os
import csv
import time
import json
import argparse
import numpy as np
import torch
import torchvision

from models.dgcspn import DgcSpn
from spnflow.torch.transforms import Reshape
from spnflow.utils.data import compute_mean_quantiles
from routines import torch_train, torch_test
from my_experiments.datasets import UCIDNADataset

from tensorboardX import SummaryWriter

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
    parser.add_argument('--epochs', type=int, default=100, help='The number of epochs.')
    parser.add_argument('--patience', type=int, default=30, help='The epochs patience used for early stopping.')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='L2 regularization factor.')
    parser.add_argument('--binary', action='store_true', default=False, help='Use binary model and binarize dataset')
    parser.add_argument('--continue_checkpoint', default=None, help='Checkpoint to continue training from')
    parser.add_argument('--dataset', type=str, default='uci-dna', help='Dataset to use for training')
    parser.add_argument('--root_dir', type=str, default='/fs/data/tejasj/Masters_Thesis/deepgen_compress/bp/datasets/ising_28_05_09_75000',
                    help='Dataset root directory')
    parser.add_argument('--gpu_id', type=int, default=0, help="GPU device to use")
    args = parser.parse_args()

    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    assert args.quantiles_loc is False or args.uniform_loc is None, \
        'Only one between --quantiles-loc and --uniform-loc can be defined'

    if args.dataset == 'uci-dna':
        in_size = (1, 180)
        data_train = UCIDNADataset(phase='train')
        data_val = UCIDNADataset(phase='val')
        data_test = UCIDNADataset(phase='test')
    else:
        NotImplementedError(f'Model is not yet supported for {args.dataset}')

    # Set the number of output classes
    if args.discriminative:
        out_classes = 10
    else:
        out_classes = 1

    # Create the results directory
    directory = os.path.join('dgcspn_1d', args.dataset)
    os.makedirs(directory, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S")
    if args.discriminative:
        directory = os.path.join(directory, 'discriminative')
        os.makedirs(directory, exist_ok=True)
    else:
        directory = os.path.join(directory, 'generative')
        os.makedirs(directory, exist_ok=True)

    # Open the results JSON of the chosen dataset
    filepath = os.path.join(directory, f'{args.dataset}_{timestamp}.json')
    if os.path.isfile(filepath):
        with open(filepath, 'r') as file:
            results = json.load(file)
    else:
        results = dict()

    # Create the writer
    writer = SummaryWriter(os.path.join(directory, 'tensorboard', timestamp))

    # Write the arguments to tensorboard
    writer.add_text('config', str(args.__dict__))

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
    )

    # Continue training if specified
    if args.continue_checkpoint is not None:
        checkpoint_name = args.continue_checkpoint
    else:    
        checkpoint_name = os.path.join(directory, f'model_{timestamp}.pt')

    # Set the device
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    # Train the model and collect the results
    if args.discriminative:
        
        # Train the model
        torch_train(model, 
                    data_train, 
                    data_val, 
                    setting='discriminative', 
                    lr=args.learning_rate,
                    batch_size=args.batch_size,
                    patience=args.patience,
                    weight_decay=args.weight_decay,
                    writer=writer,
                    epochs=args.epochs,
                    continue_checkpoint=args.continue_checkpoint,
                    device=device)

        (nll, accuracy) = torch_test(model, data_test, setting='discriminative', device=device)
        
        results[timestamp] = {
            'nll': nll,
            'accuracy': accuracy,
            'settings': args.__dict__
        }
        with open(filepath, 'w') as file:
            json.dump(results, file, indent=4)
    else:

        # Train the model
        torch_train(model, 
                    data_train, 
                    data_val, 
                    setting='generative', 
                    lr=args.learning_rate,
                    batch_size=args.batch_size,
                    patience=args.patience,
                    weight_decay=args.weight_decay,
                    writer=writer,
                    epochs=args.epochs,
                    checkpoint_name=checkpoint_name,
                    continue_checkpoint=args.continue_checkpoint,
                    gpu_id=args.gpu_id,
                    device=device)

        # Test the model
        (mu_ll, sigma_ll) = torch_test(model, data_test, setting='generative', device=device)

        # Compute the bits per pixel, if specified
        dims = np.prod(list(in_size))
        bpp = -(mu_ll / np.log(2)) / dims

        print(f'mu_ll: {mu_ll}, sigma_ll: {sigma_ll}')

        results[timestamp] = {
            'log_likelihood': {
                'mean': mu_ll,
                'stddev': sigma_ll
            },
            'bpp': bpp,
            'settings': args.__dict__
        }
        with open(filepath, 'w') as file:
            json.dump(results, file, indent=4)
