import sys
sys.path.append('..')

import os
import time
import json
import argparse
import numpy as np
import torch
import torchvision

from spnflow.torch.transforms import Reshape
from spnflow.utils.data import compute_mean_quantiles
from variational_spn.dgcspn.routines import torch_train, torch_test
from gauss_markov.utils import GaussMarkovDataset
from variational_spn.models.factorized_prior import VAE

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

if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(
        description='Deep Generalized Convolutional Sum-Product Networks (DGC-SPNs) experiments'
    )
    parser.add_argument('--network_channels', type=int, default=128, help="Number of channels in the encoder and decoder")
    parser.add_argument('--compression_channels', type=int, default=256, help="Number of channels at the bottleneck layer")
    parser.add_argument('--n-batches', type=int, default=32, help='The number of input distribution layer batches.')
    parser.add_argument('--sum-channels', type=int, default=32, help='The number of channels at sum layers.')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='The learning rate.')
    parser.add_argument('--batch-size', type=int, default=128, help='The batch size.')
    parser.add_argument('--epochs', type=int, default=1000, help='The number of epochs.')
    parser.add_argument('--patience', type=int, default=30, help='The epochs patience used for early stopping.')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='L2 regularization factor.')
    parser.add_argument('--continue_checkpoint', default=None, help='Checkpoint to continue training from')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'fashion-mnist', 'cifar10', 'gauss-markov'], default='mnist', help='Dataset to use for training')
    parser.add_argument('--root_dir', type=str, default='/fs/data/tejasj/Masters_Thesis/deepgen_compress/bp/datasets/ising_28_05_09_75000',
                    help='Dataset root directory')
    parser.add_argument('--gpu_id', type=int, default=0, help="GPU device to use")
    parser.add_argument('--data_parallel', action='store_true', default=False, help="Whether to use DataParallel while training.")
    args = parser.parse_args()

    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    if args.dataset == 'mnist':
        # Load the MNIST dataset
        in_size = (1, 28, 28)
        transform = torchvision.transforms.Compose([
                        torchvision.transforms.Grayscale(),
                        lambda x: torch.tensor(np.array(x)),
                        Reshape(in_size),
                        lambda x: x.float(),
                        lambda x: x / 256,
                        lambda x: 2 * x - 1,
                    ])
        data_train = torchvision.datasets.MNIST('../../../MNIST', train=True, transform=transform, download=True)
        data_test = torchvision.datasets.MNIST('../../../MNIST', train=False, transform=transform, download=True)
        n_val = int(0.1 * len(data_train))
        n_train = len(data_train) - n_val
        data_train, data_val = torch.utils.data.random_split(data_train, [n_train, n_val])
    if args.dataset == 'fashion-mnist':
        # Load the Fashion MNIST dataset
        in_size = (1, 28, 28)
        transform = torchvision.transforms.Compose([
                        torchvision.transforms.Grayscale(),
                        lambda x: torch.tensor(np.array(x)),
                        Reshape(in_size),
                        lambda x: x.float(),
                        lambda x: x / 256,
                        lambda x: 2 * x - 1,
                    ])
        data_train = torchvision.datasets.FashionMNIST('../../../FashionMNIST', train=True, transform=transform, download=True)
        data_test = torchvision.datasets.FashionMNIST('../../../FashionMNIST', train=False, transform=transform, download=True)
        n_val = int(0.1 * len(data_train))
        n_train = len(data_train) - n_val
        data_train, data_val = torch.utils.data.random_split(data_train, [n_train, n_val])        
    elif args.dataset == 'cifar10':
        in_size = (1, 32, 32)
        transform = torchvision.transforms.Compose([
                        torchvision.transforms.Grayscale(),
                        lambda x: torch.tensor(np.array(x)),
                        Reshape(in_size),
                        lambda x: x.float(),
                        lambda x: x / 256,
                        # lambda x: 2 * x - 1,
                    ])
        data_train = torchvision.datasets.CIFAR10('../../../CIFAR10', train=True, transform=transform, download=True)
        data_test = torchvision.datasets.CIFAR10('../../../CIFAR10', train=False, transform=transform, download=True)
        n_val = int(0.1 * len(data_train))
        n_train = len(data_train) - n_val
        data_train, data_val = torch.utils.data.random_split(data_train, [n_train, n_val])
    elif args.dataset == 'gauss-markov':
        in_size = (1, 32, 32)
        transform = torchvision.transforms.Compose([
                        Reshape(in_size),
                        lambda x: x.float(),
                    ])
        data_train = GaussMarkovDataset(phase='train')
        data_test = GaussMarkovDataset(phase='test')
        n_val = int(0.1 * len(data_train))
        n_train = len(data_train) - n_val
        data_train, data_val = torch.utils.data.random_split(data_train, [n_train, n_val])
    else:
        NotImplementedError(f'Model is not yet supported for {args.dataset}')

    # Create the results directory
    if args.dataset in ['mnist', 'fashion-mnist', 'cifar10', 'gauss-markov']:
        directory = os.path.join('vae', args.dataset)
    else:
        directory = os.path.join('vae', os.path.basename(args.root_dir))
    os.makedirs(directory, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S")
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

    # Build the model
    model = VAE(
        network_channels=args.network_channels,
        compression_channels=args.compression_channels,
        num_base_distributions=args.n_batches,
        in_size=in_size,
        sum_channels=args.sum_channels,
    )

    # Continue training if specified
    if args.continue_checkpoint is not None:
        checkpoint_name = args.continue_checkpoint
    else:    
        checkpoint_name = os.path.join(directory, f'model_{timestamp}.pt')

    # Set the device
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    # Set specific setting when using multiple GPUs
    if args.data_parallel:
        print("Using Data Parallel ...")
        assert args.gpu_id == 0
        model = MyDataParallel(model, device_ids=[0, 1])

    # Train the model and collect the results
    torch_train(model, 
                data_train, 
                data_val, 
                lr=args.learning_rate,
                batch_size=args.batch_size,
                patience=args.patience,
                weight_decay=args.weight_decay,
                writer=writer,
                epochs=args.epochs,
                checkpoint_name=checkpoint_name,
                continue_checkpoint=args.continue_checkpoint,
                gpu_id=args.gpu_id,
                device=device,
                data_parallel=args.data_parallel)

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
