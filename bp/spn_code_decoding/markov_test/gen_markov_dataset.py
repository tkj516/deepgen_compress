import numpy as np
import os
import argparse
import sys

sys.path.append('..')
sys.path.append('../..')

from spn_code_decoding.markov_test.utils import *
from spn_code_decoding.markov_test.markov_source import *

parser = argparse.ArgumentParser(description='Dataset generation arguments')
parser.add_argument('--height', type=int, default=1, help="Height of image")
parser.add_argument('--width', type=int, default=1000, help="Width of image")
parser.add_argument('--alphabet_size', type=int, default=256, help="Alphabet size")
parser.add_argument('--reference_entropy', type=float, default=0.01, help="Reference entropy")
parser.add_argument('--num_samples', type=int, default=200000, help="Number of images in dataset")
parser.add_argument('--root', type=str, default='data', help="Directory to store images")
args = parser.parse_args()

if __name__ == "__main__":

    h = args.height
    w = args.width
    M = args.alphabet_size
    hf = args.reference_entropy

    # Calculate the number of bits
    bits = int(np.log2(M))

    # Create root directory if it does not exist
    if not os.path.exists(args.root):
        os.makedirs(args.root)

    # Setup the Markov Source
    markov = MarkovSource(
        N=w,
        M=M,
        hf=hf,
    )

    # Define map to prevent duplicates
    check_map = {}
    count = 0

    # Fill count
    fill = len(str(args.num_samples))

    # Generate samples
    while count < args.num_samples:

        # Sample a new image
        sample, _ = generate_sample(
                        epot=markov.epot_src,
                        sta=markov.sta,
                        N=w,
                        bits=bits,
                    )
        
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

    


