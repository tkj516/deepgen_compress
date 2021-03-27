import torch
import numpy as np
from tqdm import tqdm

class GridBP():

    def __init__(self, h, w, p=0.5, stay=0.9, alpha=0.8):

        # Get the height and width of the image
        self.h = h
        self.w = w
        self.N = h * w 
        
        # Define the node and edge potentials
        self.npot = np.tile(np.array([1-p, p]), (h*w, 1)).reshape(h, w, 2)
        self.epot = np.array([[stay, 1-stay],
                              [1-stay, stay]]).reshape(1, 1, 2, 2)
        self.epot_v = np.tile(self.epot, (h-1, w, 1, 1))
        self.epot_h = np.tile(self.epot, (h, w-1, 1, 1))

        # Initialize the messages
        self.Md = np.ones((self.h-1, self.w, 2))/2
        self.Mu = np.ones((self.h-1, self.w, 2))/2
        self.Mr = np.ones((self.h, self.w-1, 2))/2
        self.Ml = np.ones((self.h, self.w-1, 2))/2
        self.Mout = np.ones((self.h, self.w, 2))/2

        # Initialize the learning rate
        self.alpha = alpha

    def update(self, Min):
        """
        Min : h x w x 2
        Mout: h x w x 2
        """

        # Resolve contradictory signals
        S = np.sum(self.npot * Min, -1)
        Min[..., 0][S == 0] = 0.5
        Min[..., 1][S == 0] = 0.5
       
        # Top to bottom
        Md1  = self.npot[:-1, ...] * Min[:-1, ...]
        Md1[1:, ...] *= self.Md[:-1, ...]
        Md1[:, 1:, :] *= self.Mr[:-1, ...]
        Md1[:, :-1, :] *= self.Ml[:-1, ...]
        Md1[..., 0], Md1[..., 1] = np.sum(self.epot_v[..., 0, :]*Md1, -1), np.sum(self.epot_v[..., 1, :]*Md1, -1)

        # Bottom to top
        Mu1  = self.npot[1:, ...] * Min[1:, ...]
        Mu1[:-1, ...] *= self.Mu[1:, ...]
        Mu1[:, 1:, :] *= self.Mr[1:, ...]
        Mu1[:, :-1, :] *= self.Ml[1:, ...]
        Mu1[..., 0], Mu1[..., 1] = np.sum(self.epot_v[..., 0]*Mu1, -1), np.sum(self.epot_v[..., 1]*Mu1, -1)

        # Left to right
        Mr1  = self.npot[:, :-1, :] * Min[:, :-1, :]
        Mr1[:-1, ...] *= self.Mu[:, :-1, :]
        Mr1[:, 1:, :] *= self.Mr[:, :-1, :]
        Mr1[1:, ...] *= self.Md[:, :-1, :]
        Mr1[..., 0], Mr1[..., 1] = np.sum(self.epot_h[..., 0, :]*Mr1, -1), np.sum(self.epot_h[..., 1, :]*Mr1, -1)

        # Right to left
        Ml1  = self.npot[:, 1:, :] * Min[:, 1:, :]
        Ml1[:-1, ...] *= self.Mu[:, 1:, :]
        Ml1[:, :-1, :] *= self.Ml[:, 1:, :]
        Ml1[1:, :, :] *= self.Md[:, 1:, :]
        Ml1[..., 0], Ml1[..., 1] = np.sum(self.epot_h[..., 0]*Ml1, -1), np.sum(self.epot_h[..., 1]*Ml1, -1)

        # Normalize the messages
        Md1 /= np.sum(Md1, -1).reshape(self.h-1, self.w, 1)
        Mu1 /= np.sum(Mu1, -1).reshape(self.h-1, self.w, 1)
        Mr1 /= np.sum(Mr1, -1).reshape(self.h, self.w-1, 1)
        Ml1 /= np.sum(Ml1, -1).reshape(self.h, self.w-1, 1)

        # Update the messages
        self.Md = self.alpha*Md1 + (1-self.alpha)*self.Md
        self.Mu = self.alpha*Mu1 + (1-self.alpha)*self.Mu
        self.Mr = self.alpha*Mr1 + (1-self.alpha)*self.Mr
        self.Ml = self.alpha*Ml1 + (1-self.alpha)*self.Ml

        # Compute Mout
        self.Mout = np.ones((self.h, self.w, 2))
        self.Mout[1:, :, :] *= self.Md
        self.Mout[:-1, ...] *= self.Mu
        self.Mout[:, 1:, :] *= self.Mr
        self.Mout[:, :-1, :] *= self.Ml
        self.Mout /= np.sum(self.Mout, -1)[..., np.newaxis]

    def bp(self, Min, num_iter=1):

        for i in tqdm(range(num_iter)):
            self.update(Min)

def test_grid_bp():
    grid_bp = GridBP(128, 128)
    Min = np.rand(128, 128, 2)
    grid_bp.bp(Min)

if __name__ == "__main__":
    test_grid_bp()



