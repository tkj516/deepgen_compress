import sys
sys.path.append('..')

import os
import argparse
import numpy as np
import torch

from variational_spn.models.factorized_prior import VAE

device = torch.device('cuda:0')

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


def approximate_image_prior(
    args,
    in_size=(1, 32, 32),
    num_samples=512,
):

    # Load the model from checkpoint
    model = VAE(
        network_channels=args.network_channels,
        compression_channels=args.compression_channels,
        num_base_distributions=args.n_batches,
        in_size=in_size,
        sum_channels=args.sum_channels,
    )

    # Set specific setting when using multiple GPUs
    if args.data_parallel:
        print("Using Data Parallel ...")
        assert args.gpu_id == 0
        model = MyDataParallel(model, device_ids=[0, 1])

    model.to(device)

    # Get the latent size
    latent_shape = (args.compression_channels, in_size[1]//2**4, in_size[2]//2**4)

    # Approximate the mean and log_scales of the SPN
    # representing p(x)
    means_total, log_scales_total = compute_moments(model, latent_shape, num_samples=num_samples)

    return means_total, log_scales_total


def compute_moments(model, latent_shape, num_samples):

    if num_samples >= 64:
        print("Running a loop to compute moment estimates...")

    means_list = []
    log_scale_list = []

    for i in range(0, num_samples, 64):

        # Sample some latents using the prior distribution
        z = torch.zeros(min(i + 64, num_samples) - i, *latent_shape).to(device)

        # Decode the latents
        means, log_scales = model.decode(z)  # (bs, num_base_distributions, h, w)

        means_list.append(means)
        log_scale_list.append(log_scales)

    # Now compute the total mean and variance
    means_total = torch.cat(means_list, dim=0).mean(dim=0)
    second_moment = torch.exp(2 * torch.cat(log_scale_list, dim=0)) + torch.cat(means_list, dim=0)**2
    scales_total = second_moment.mean(dim=0) - means_total ** 2
    log_scales_total = torch.log(scales_total) / 2

    return means_total, log_scales_total


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(
        description='Deep Generalized Convolutional Sum-Product Networks (DGC-SPNs) experiments'
    )
    parser.add_argument('--network_channels', type=int, default=128, help="Number of channels in the encoder and decoder")
    parser.add_argument('--compression_channels', type=int, default=256, help="Number of channels at the bottleneck layer")
    parser.add_argument('--n-batches', type=int, default=32, help='The number of input distribution layer batches.')
    parser.add_argument('--sum-channels', type=int, default=32, help='The number of channels at sum layers.')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'fashion-mnist', 'cifar10', 'gauss-markov'], default='mnist', help='Dataset to use for training')
    parser.add_argument('--root_dir', type=str, default='/fs/data/tejasj/Masters_Thesis/deepgen_compress/bp/datasets/ising_28_05_09_75000',
                    help='Dataset root directory')
    parser.add_argument('--checkpoint', type=str, default='/fs/data/tejasj/Masters_Thesis/deepgen_compress/bp/variational_spn/vae/cifar10/generative/model_2022-04-25_23:24:08.pt')
    parser.add_argument('--gpu_id', type=int, default=0, help="GPU device to use")
    parser.add_argument('--data_parallel', action='store_true', default=False, help="Whether to use DataParallel while training.")
    args = parser.parse_args()

    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    if args.dataset == 'mnist':
        # Load the MNIST dataset
        in_size = (1, 28, 28)
    if args.dataset == 'fashion-mnist':
        # Load the Fashion MNIST dataset
        in_size = (1, 28, 28)     
    elif args.dataset == 'cifar10':
        in_size = (1, 32, 32)
    elif args.dataset == 'gauss-markov':
        in_size = (1, 32, 32)
    else:
        NotImplementedError(f'Model is not yet supported for {args.dataset}')

    means_total, log_scales_total = approximate_image_prior(
                                        args,
                                        in_size=in_size,
                                        num_samples=512,
                                    )
    directory_path = os.path.basename(args.checkpoint).split('.')[-1]
    os.makedirs(directory_path, exist_ok=True)
    save_dict = {
        'means': means_total,
        'log_scales': log_scales_total,
    }
    torch.save(save_dict, os.path.join(directory_path, 'moments.pt'))
