import numpy as np
import os
import argparse

from bp.torch_parallel.grid_gibbs import GibbsSampler

parser = argparse.ArgumentParser(description='Dataset generation arguments')
parser.add_argument('--height', type=int, default=28, help="Height of image")
parser.add_argument('--width', type=int, default=28, help="Width of image")
parser.add_argument('--p', type=float, default=0.5, help="Node probability")
parser.add_argument('--stay', type=float, default=0.9, help="Edge probability")
parser.add_argument('--num_images', type=int, default=21000, help="Number of images in dataset")
parser.add_argument('--root', type=str, default='data', help="Directory to store images")
args = parser.parse_args()

if __name__ == "__main__":

    # Create root directory if it does not exist
    if not os.path.exists(args.root):
        os.makedirs(args.root)

    # Define the Gibbs sampler
    sampler = GibbsSampler(args.height, args.width, args.p, args.stay)

    # Define map to prevent duplicates
    check_map = {}
    count = 0

    # Fill count
    fill = len(str(args.num_images))

    # Generate samples
    while count < args.num_images:

        # Sample a new image
        sampler.reset()
        sampler.sampler(1000)
        sample = sampler.samp
        sample_hash = tuple(sample.reshape(-1, ))

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

    


