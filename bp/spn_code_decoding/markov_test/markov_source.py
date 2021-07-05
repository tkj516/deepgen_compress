import torch
import torch.nn as nn
import numpy as np
from .utils import *
from tqdm import tqdm

class MarkovSource():
    """
    Class that defines operations on a Markov Chain
    """

    def __init__(
        self, 
        N=1000,
        M=256,
        hf=0.01,
        ):
        """Setup a Markov chain source.

        Args:
            N (int, optional): Length of chain. Defaults to 1000.
            M (int, optional): Alphabet size. Defaults to 256.
            hf (float, optional): Reference entropy. Defaults to 0.01.
        """

        # Length of the signal
        self.N = N
        # Number of bits per alphabet
        self.bits = np.log2(M)
        # Total number of bits
        self.Nbits = self.N * self.bits

        # How much neighboring values are the same
        self.hf = hf

        # Generate a markov chain
        (
            self.bleed_src,
            self.epot,
            self.sta
        ) = generate_transition_matrix(self.M, self.hf)

    def sample(self):
        """Generate a sample from the Markov chain.

        Returns:
            np.ndarray: Sample from the Markov chain.
        """

        samp, graycoded_samp = generate_sample(
                                    self.epot, 
                                    self.sta, 
                                    self.N, 
                                    self.bits
                                )

        return samp, graycoded_samp

    def entropy(self):
        """Compute the entropy of the Markov chain.

        Returns:
            float: Entropy of the chain.
        """

        hcond = [entropy(self.epot[:, i]) for i in range(self.M)]
        h = hcond @ self.sta / self.bits

        return h


class MarkovSourceBP(nn.Module):
    """
    Class that defines operations for performing BP on 
    Markov chains.
    """

    def __init__(
        self,
        npot,
        epot,
        alpha=0.9,
        device=torch.device("cuda:0"),
    ):
        """Setup the Markov source for BP.

        Args:
            npot (np.ndarray): Node potentials.
            epot (np.ndarray): Edge potentials
            alpha (float, optional): Message update rate. Defaults to 0.9.
            device ([type], optional): Device to perform operations on. Defaults to torch.device("cuda:0").
        """
        super(MarkovSourceBP, self).__init__()

        # Get the height and width of the image
        self.h = npot.shape[0]
        self.w = npot.shape[1]
        self.N = self.h * self.w 
        self.M = npot.shape[3]
        
        # Define the node and edge potentials
        # Register the potentials as buffers to indicate non-trainable members of the module
        self.register_buffer('npot', torch.Tensor(npot))
        self.register_buffer('epot', torch.Tensor(epot))

        # Initialize the messages
        # Set them as parameters for gradient backpropagation
        self.Mr = nn.Parameter(torch.ones(self.h, self.w-1, self.M) / self.M)
        self.Ml = nn.Parameter(torch.ones(self.h, self.w-1, self.M) / self.M)
        self.Mout = nn.Parameter(torch.ones(self.h, self.w, self.M) / self.M)

        # Initialize the learning rate
        self.alpha = alpha

        # Store the device name
        self.device = device

    def forward(self, Min):
        """
        Min : 1 x N x M
        Mout: 1 x N x M
        """
       
        # Resolve contradictory signals
        S = torch.sum(self.npot * Min, -1)
        for i in range(self.M):
            Min[..., i][S == 0] = 1 / self.M

        # Left to right
        temp  = self.npot[:, :-1, :] * Min[:, :-1, :]
        temp[:, 1:, :] *= self.Mr[:, :-1, :]
        Mr1 = torch.matmul(temp.reshape(-1, self.M), self.epot.permute(1, 0)).reshape(self.h, self.w-1, self.M)

        # Right to left
        temp  = self.npot[:, 1:, :] * Min[:, 1:, :]
        temp[:, :-1, :] *= self.Ml[:, 1:, :]
        Ml1 = torch.matmul(temp.reshape(-1, self.M), self.epot.permute(1, 0)).reshape(self.h, self.w-1, self.M)

        # Normalize the messages
        Mr1 /= torch.sum(Mr1, -1, keepdim=True)
        Ml1 /= torch.sum(Ml1, -1, keepdim=True)

        # Update the messages
        # Since the messages are parameters access using .data
        self.Mr.data = self.alpha*Mr1 + (1-self.alpha)*self.Mr.data
        self.Ml.data = self.alpha*Ml1 + (1-self.alpha)*self.Ml.data

        # Compute Mout
        self.Mout.data = torch.ones(self.h, self.w, self.M).to(self.device)
        self.Mout[:, 1:, :] *= self.Mr
        self.Mout[:, :-1, :] *= self.Ml
        self.Mout.data /= torch.sum(self.Mout.data, -1, keepdim=True)

    def reset_messages(self):

        # Reset the message parameters
        self.Mr = nn.Parameter(torch.ones(self.h, self.w-1, self.M) / self.M).to(self.device)
        self.Ml = nn.Parameter(torch.ones(self.h, self.w-1, self.M) / self.M).to(self.device)
        self.Mout = nn.Parameter(torch.ones(self.h, self.w, self.M) / self.M).to(self.device)

    def bp(self, Min, num_iter=1):

        for i in tqdm(range(num_iter)):
            self.forward(Min)






