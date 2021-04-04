import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.linalg import norm
import argparse

parser = argparse.ArgumentParser(description='Gibbs generation arguments')
parser.add_argument('--height', type=int, default=28, help="Height of image")
parser.add_argument('--width', type=int, default=28, help="Width of image")
parser.add_argument('--p', type=float, default=0.5, help="Node probability")
parser.add_argument('--stay', type=float, default=0.9, help="Edge probability")
args = parser.parse_args()

class GibbsSampler():

    def __init__(self, h, w, p=0.5, stay=0.9):

        # Get the height and width of the image
        self.h = h
        self.w = w
        self.N = h * w 
        
        # Dfine the node and edge potentials
        self.npot = np.tile(np.array([1-p, p]), (h*w, 1)).reshape(h, w, 2)
        self.epot = np.array([[stay, 1-stay],
                              [1-stay, stay]]).reshape(1, 1, 2, 2)
        self.epot_v = np.tile(self.epot, (h-1, w, 1, 1))
        self.epot_h = np.tile(self.epot, (h, w-1, 1, 1))

        # Calculate the log potentials
        self.log_npot = 0.5 * (np.log(self.npot[..., 0]) - np.log(self.npot[..., 1]))
        self.log_epot_v = 0.5 * (np.log(self.epot_v[..., 0, 0]) - np.log(self.epot_v[..., 0, 1]))
        self.log_epot_h = 0.5 * (np.log(self.epot_h[..., 0, 0]) - np.log(self.epot_h[..., 0, 1]))

        # Initial sample
        self.samp = None
        self.sdist = None

        # Helper variables for subtraction
        self.mask = np.tile(np.eye(2), (h//2, w//2))

    def generate_sample_sequential(self):

        # Generate an initial starting sample if none provided
        if self.samp is None:
            self.samp = np.random.rand(self.h, self.w) > 0.5
        # Initialize the output beliefs
        if self.sdist is None:
            self.sdist = np.zeros(self.samp.shape)
        # Perform belief propagation using a random ordering
        indices = np.random.permutation(self.N)

        # Loop over the image
        for i in indices:
            w = i // self.h
            h = i % self.h
            Ml = Mr = Md = Mu = 0

            if w > 0:
                Ml = self.log_epot_h[h, w-1] * (1 - 2*self.samp[h, w-1])
            if w < self.w-1:
                Mr = self.log_epot_h[h,w] * (1-2*self.samp[h,w+1])
            if h > 0:
                Md = self.log_epot_v[h-1, w] * (1 - 2*self.samp[h-1, w])
            if h < self.h-1:
                Mu = self.log_epot_v[h,w] * (1-2*self.samp[h+1,w])

            self.sdist[h, w] = self.log_npot[h, w] + Ml + Mr + Mu + Md
            self.samp[h, w] = (np.random.rand() < 0.5*(1 - np.tanh(self.sdist[h, w]))).astype(float)

        self.sdist = self.log_npot + Ml + Mr + Mu + Md
        self.samp = (np.random.rand(self.h, self.w) < 0.5 * (1 - np.tanh(self.sdist))).astype(float)

    def generate_sample_chromatic(self):

        # Generate an initial starting sample if none provided
        if self.samp is None:
            self.samp = np.random.rand(self.h, self.w) > 0.5
        # Initialize the output beliefs
        if self.sdist is None:
            self.sdist = np.zeros(self.samp.shape)

        Mu = np.pad(self.log_epot_v * (1 - 2 * self.samp[1:, :]), ((0, 1), (0, 0)))
        Md = np.pad(self.log_epot_v * (1 - 2 * self.samp[:-1, :]), ((1, 0), (0, 0)))
        Ml = np.pad(self.log_epot_h * (1 - 2 * self.samp[:, :-1]), ((0, 0), (1, 0)))
        Mr = np.pad(self.log_epot_h * (1 - 2 * self.samp[:, 1:]), ((0, 0), (0, 1)))

        self.sdist = self.log_npot + Ml + Mr + Mu + Md
        self.samp = (1 - self.mask)*self.samp + \
                    self.mask*(np.random.rand(self.h, self.w) < 0.5 * (1 - np.tanh(self.sdist))).astype(float)
        # Change the mask to update the other color nodes
        self.mask = 1 - self.mask

    def set_sample(self, samp):

        self.samp = samp

    def sampler(self, num_iter=1000):

        num_SEs_h = [0]
        num_SEs_v = [0]

        for i in range(num_iter):
            self.generate_sample_chromatic()

            num_SEs_h.append(num_SEs_h[-1] + np.sum(np.abs(np.diff(self.samp, axis=1))))
            num_SEs_v.append(num_SEs_v[-1] + np.sum(np.abs(np.diff(self.samp, axis=0))))

            if i > 50:
                v1 = num_SEs_v[-50] - num_SEs_v[-41] - num_SEs_v[-10] + num_SEs_v[-1]
                h1 = num_SEs_h[-50] - num_SEs_h[-41] - num_SEs_h[-10] + num_SEs_h[-1]

                if norm(np.array([v1, h1])) < 0.2 * np.sqrt(self.N):
                    break

def test_gibbs_sampler():

    gibbs_sampler = GibbsSampler(args.height, args.width, args.p, args.stay)
    gibbs_sampler.sampler(2000)

    fig, ax = plt.subplots()
    ax.imshow(gibbs_sampler.samp)
    plt.show()

if __name__ == "__main__":

    test_gibbs_sampler()

                
