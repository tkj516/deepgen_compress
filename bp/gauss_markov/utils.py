import numpy as np
import argparse
import os
from torch.utils.data import Dataset
import torch

def generate_markov(mu_0=0, 
                    s_0=1, 
                    s=0.51, 
                    a=0.7, 
                    n=1024):

    samp = np.zeros(n)
    samp[0] = np.random.normal(mu_0, np.sqrt(s_0))
    for i in range(1, n):
        samp[i] = a * samp[i-1] + np.random.normal(0, np.sqrt(s))

    return samp


class GaussMarkovDataset(Dataset):
    """Dataset of Gibbs sampled Ising images"""

    def __init__(self, 
                root_dir='/fs/data/tejasj/Masters_Thesis/deepgen_compress/bp/gauss_markov_07_051_0_1',
                phase='train', 
                transform=None):
        # Root directory for the data
        self.root_dir = root_dir

        # Choose the phase
        self.phase = phase

        # Read the training files from the mat file
        self.files = sorted(os.listdir(self.root_dir))

        # Choose the number of files
        if self.phase == 'train':
            start_idx = 0
            end_idx = int(0.9*len(self.files))
        else:
            start_idx = int(0.9*len(self.files))
            end_idx = len(self.files)

        self.files = sorted(os.listdir(self.root_dir))[start_idx:end_idx]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        image = os.path.join(self.root_dir, self.files[idx])
        image = np.load(image).reshape(32, 32)

        sample = torch.FloatTensor(image).unsqueeze(0)

        return sample, torch.tensor([0])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Dataset generation arguments')
    parser.add_argument('--n', type=int, default=1024, help="Number of samples")
    parser.add_argument('--a', type=float, default=0.7, help="Autocorrelation parameter")
    parser.add_argument('--mu_0', type=float, default=0, help="Edge probability")
    parser.add_argument('--s_0', type=float, default=1, help="Edge probability")
    parser.add_argument('--s', type=float, default=0.51, help="Edge probability")
    parser.add_argument('--num_images', type=int, default=100000, help="Number of images in dataset")
    parser.add_argument('--root', type=str, default='../gauss_markov_07_051_0_1', help="Directory to store images")
    args = parser.parse_args()

    # Create root directory if it does not exist
    if not os.path.exists(args.root):
        os.makedirs(args.root)

    # Define the Gibbs sampler
    sampler = generate_markov

    # Define map to prevent duplicates
    check_map = {}
    count = 0

    # Fill count
    fill = len(str(args.num_images))

    # Generate samples
    while count < args.num_images:

        # Sample a new image
        sample = sampler(args.mu_0, args.s_0, args.s, args.a, args.n)
        sample_hash = hash(sample.data.tobytes())

        if sample_hash in check_map:
            pass
        else:
            filename = os.path.join(args.root, str(count).zfill(fill))
            np.save(filename, sample)
            check_map[sample_hash] = 1
            count += 1
        
        if count % 100 == 0:
            print(f'Number of samples generated = {count}')

    print(f'[Done!]')

