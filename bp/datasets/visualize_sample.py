import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Dataset visualization arguments')
parser.add_argument('--root', type=str, default='data', help="Directory to store images")
args = parser.parse_args()

if __name__ == "__main__":

    # Create root directory if it does not exist
    if not os.path.exists(args.root):
        RuntimeError("Root directory not found")

    # Fill count
    files = os.listdir(args.root)
    num_images = len(files)

    # Choose a random image from the dataset
    idx = np.random.randint(0, num_images)

    fix, ax = plt.subplots()
    ax.imshow(np.load(os.path.join(args.root, files[idx])))
    ax.set_title(f'Sample {idx}')
    plt.show()

    


