import torch
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix

class CodeBP():

    def __init__(self, H):

        # Get the shape of the parity check matrix
        self.K, self.N = H.shape
        self.H = H

        # Set an epsilon for high precision calculations
        self.epsilon = 1e-20

        # Build the adjaceny list for variable and check nodes
        self.neighbors_of_factor = self.neighbors_of_variable = []
        self.build_adjacency_list()

        # Initialize the messages to all 0
        self.Hsx = torch.zeros(self.K, self.N)
        self.Hxs = torch.zeros(self.K, self.N)
        self.Mout = torch.zeros((self.N, 2))

    def build_adjacency_list(self):

        print("[Building adjacency list ...]")

        # Intialize the lists
        self.neighbors_of_factor = [[] for _ in range(self.K)]
        self.neighbors_of_variable = [[] for _ in range(self.N)]

        # Create a sparse matrix
        sp_H = csr_matrix(self.H)
        rows, cols = sp_H.nonzero()
        
        # Build the factor adjaceny list
        for a, i in tqdm(zip(rows, cols)):
            self.neighbors_of_variable[i].append(a)
            self.neighbors_of_factor[a].append(i)

    def update(self, ps, x, Min):
        """
        Min : N x 2
        ps  : N x 2
        Mout: N x 2
        """

        # Get the shape of the tensors
        _, c = ps.shape

        if Min is None:
            Min = torch.ones(self.N, 2)/2

        # Perturb tensors by epsilon
        Min += self.epsilon
        ps += self.epsilon
        Min /= torch.sum(Min, -1).unsqueeze(-1)
        ps /= torch.sum(ps, -1).unsqueeze(-1)

        # Calculate the log likelihood of the messages
        log_Min = 0.5 * (torch.log(Min[:, 0]) - torch.log(Min[:, 1]))
        log_ps = 0.5 * (torch.log(ps[:, 0]) - torch.log(ps[:, 1]))

        # Node to factor messages
        for i in range(self.N):
            factor_neighbors = self.neighbors_of_variable[i]

            self.Hsx[factor_neighbors, i] = log_Min[i] + log_ps[i]

            # Sum up all check node message incoming to a variable node
            ll_sum = torch.sum(self.Hxs[factor_neighbors, i], -1)

            # Now calculate the node to factor message for each factor
            self.Hsx[factor_neighbors, i] += ll_sum - self.Hxs[factor_neighbors, i]

        # Factor to node messages
        for a in range(self.K):
            variable_neighbors = self.neighbors_of_factor[a]
            
            # Multiply all the likelihoods
            # Initialize diff product as -1 or 1
            ll_diff_prod = 1 - 2*x[a]
            ll_diff_prod *= torch.prod(torch.tanh(self.Hsx[a, variable_neighbors]+self.epsilon), -1)

            # Now calculate the factor to node message for each variable
            self.Hxs[a, variable_neighbors] = torch.atanh(ll_diff_prod / torch.tanh(self.Hsx[a, variable_neighbors]+self.epsilon))

        # Compute outgoing super edge message
        self.M_out = torch.zeros((self.N, 2))
        for i in range(self.N):
            factor_neighbors = self.neighbors_of_variable[i]

            M_out_diff = torch.tanh(torch.sum(self.Hxs[factor_neighbors, i], -1)).unsqueeze(-1)
            self.M_out[i, :] = torch.cat([0.5 + M_out_diff/2, 0.5 - M_out_diff/2], -1)






